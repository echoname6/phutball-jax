#!/usr/bin/env python3
"""Self-stop warm start, step 1: build SFT examples from the model's OWN thinking, cut where it could have stopped.

The thinking-on model knows short answers early but never stops (12k-token probe: forced answers 31/32 correct, 100%
of rollouts cut off). This teaches the missing skill without writing any reasoning for it:
  1. sample a thinking rollout per puzzle (vLLM, thinking on, up to --max-think tokens);
  2. cut it at several lengths (snapped back to a line end) and force an answer after each cut
     (prefix + "</think>\\n\\nANSWER:", 16 greedy tokens);
  3. keep the SHORTEST cut whose forced answer the engine verifies AND stays verified at every longer cut (a lucky
     early guess that later changes is not kept). A rollout that finished on its own with a verified answer is kept
     whole;
  4. example = the model's own thinking up to that cut + "\\n</think>\\n\\n" + its own verified answer line.
No-win puzzles: a forced "NO WIN" is right at any length, so their cut is drawn from the distribution of the cuts
selected on positive puzzles (length must not become a cue for "no win"), and it must say NO WIN there and later.
Every token of reasoning and every answer is the model's own; the engine only chooses where it could have stopped.

  python -m llm_train.self_stop --out /content/drive/MyDrive/phutball/self_stop/sft.jsonl --n 900
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STOP = "\n</think>\n\n"
CUTS = (768, 1024, 1536, 2048, 3072, 4096)


def snap(text: str) -> str:
    """Cut back to the last line end, unless that would drop more than half the text."""
    k = text.rfind("\n")
    return text[:k] if k > len(text) // 2 else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--puzzles", type=Path, default=ROOT / "llm_train/puzzles/grpo_train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=900, help="puzzles to sample (one rollout each)")
    ap.add_argument("--mix", default="win_pos=0.4,win_neg=0.2,block=0.2,forced=0.2",
                    help="share of --n per subtask (prevent left out: one-rule shortcut)")
    ap.add_argument("--max-think", type=int, default=max(CUTS))
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); t0 = time.time(); rng = random.Random(a.seed)
    from vllm import LLM, SamplingParams
    from expert.engine import State
    from llm_bench.run import score
    from llm_bench.text import prompt

    rows = [json.loads(l) for l in open(a.puzzles)]; rng.shuffle(rows)
    share = {k: float(v) for k, v in (p.split("=") for p in a.mix.split(","))}
    picked = []
    for sub, f in share.items():
        picked += [r for r in rows if r["subtask"] == sub][: int(round(f * a.n))]
    print(f"{len(picked)} puzzles: {dict(Counter(r['subtask'] for r in picked))}", flush=True)

    llm = LLM(a.model, max_model_len=8192, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=a.seed)
    tok = llm.get_tokenizer()
    heads = []
    for r in picked:
        it = r["item"]; s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"])
        heads.append(tok.apply_chat_template([{"role": "user", "content": prompt(it["task"], s)}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True))
    gen = llm.generate(heads, SamplingParams(temperature=a.temperature, top_p=0.95, top_k=20, max_tokens=a.max_think,
                                             seed=a.seed))
    print(f"generated {len(gen)} rollouts in {time.time() - t0:.0f}s", flush=True)

    # forced answers after every cut (prefix caching makes the shared prefixes cheap)
    jobs, natural = [], {}
    for i, (r, o) in enumerate(zip(picked, gen)):
        ids, text = o.outputs[0].token_ids, o.outputs[0].text
        if "</think>" in text:                                          # finished on its own
            natural[i] = text
        for c in CUTS:
            if c <= len(ids):
                jobs.append((i, c, snap(tok.decode(ids[:c]))))
    forced = llm.generate([heads[i] + pre + STOP + "ANSWER:" for i, _, pre in jobs],
                          SamplingParams(temperature=0.0, max_tokens=16))
    hits = defaultdict(dict)                                            # i -> {cut: (ok, prefix, answer line)}
    for (i, c, pre), o in zip(jobs, forced):
        ans = "ANSWER:" + o.outputs[0].text.split("\n")[0]
        hits[i][c] = (bool(score(picked[i]["item"], ans)["correct"]), pre, ans)
    print(f"forced {len(jobs)} cuts in {time.time() - t0:.0f}s", flush=True)

    def stable_from(h: dict, allowed) -> int | None:
        cs = sorted(h)
        for k, c in enumerate(cs):
            if c in allowed and all(h[d][0] for d in cs[k:]): return c
        return None

    out, stats, pos_cuts = [], Counter(), []
    order = sorted(range(len(picked)), key=lambda i: picked[i]["subtask"] == "win_neg")   # positives first
    for i in order:
        r = picked[i]; sub = r["subtask"]; h = hits.get(i, {})
        if i in natural:
            th, ans = natural[i].split("</think>", 1)
            if score(r["item"], ans)["correct"]:
                out.append((i, th.rstrip("\n"), ans.strip().split("\n")[-1], "natural")); stats[f"{sub}: natural"] += 1
                continue
        if sub == "win_neg":
            if not pos_cuts: stats[f"{sub}: no positive cuts yet"] += 1; continue
            target = rng.choice(pos_cuts)
            c = stable_from(h, {target})
        else:
            c = stable_from(h, set(CUTS))
        if c is None: stats[f"{sub}: no stable verified cut"] += 1; continue
        ok, pre, ans = h[c]
        out.append((i, pre, ans, c)); stats[f"{sub}: cut {c}"] += 1
        if sub != "win_neg": pos_cuts.append(c)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for i, pre, ans, how in out:
            r = picked[i]
            f.write(json.dumps({"prompt": heads[i], "completion": pre + STOP + ans.strip() + tok.eos_token,
                                "id": r["id"], "subtask": r["subtask"], "cut": how,
                                "think_tokens": len(tok(pre, add_special_tokens=False)["input_ids"])}) + "\n")
    kept = Counter(picked[i]["subtask"] for i, *_ in out)
    print(f"{len(out)} examples from {len(picked)} puzzles in {time.time() - t0:.0f}s -> {a.out}")
    print("kept by subtask:", dict(kept))
    for k, v in sorted(stats.items()): print(f"  {k}: {v}")
    L = sorted(json.loads(l)["think_tokens"] for l in open(a.out))
    if L: print(f"thinking tokens kept: median {L[len(L) // 2]}, p90 {L[int(0.9 * (len(L) - 1))]}, max {L[-1]}")


if __name__ == "__main__":
    main()
