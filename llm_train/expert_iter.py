#!/usr/bin/env python3
"""Expert iteration (ReST-EM / rejection fine-tuning) round: sample with the current model, keep the model's own
engine-verified solutions, fine-tune on them. Two sampling arms with the same budget (k solves per puzzle):

  iid        k independent thinking samples per puzzle.
  strategic  approach-diverse sampling (Gurung, Whitammer & Lapata, "Strategically Diverse Sampling for Self-Training"):
             the model first lists k genuinely different strategies for the puzzle (thinking off, verbalized-sampling
             style, no solving), then solves once per strategy with the strategy as a hidden instruction. The training
             example keeps only the original puzzle and the solution; solutions that refer to their instruction are
             dropped (the instruction will not be there at test time).

Both arms keep, per sample: a solution that finished on its own with a verified answer, else (to keep the stopping
skill) the shortest cut whose forced answer verifies and stays verified at every longer cut (llm_train/self_stop.py).
Only verified answers are ever trained on, so a wrong "NO WIN" can never be reinforced. Up to --keep per puzzle
(shortest first; in the strategic arm from different strategies); no-win examples capped at --max-neg-ratio x wins.

  python -m llm_train.expert_iter --arm strategic --adapter DRIVE/self_stop/adapter/final --out DRIVE/ei/r1_strategic
writes OUT/sft.jsonl (prompt = the original puzzle only) and OUT/stats.json; then
  python -m llm_train.sft_stop --data OUT/sft.jsonl --out OUT/adapter     (fine-tunes from the base model)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm_train.self_stop import CUTS, STOP, snap  # noqa: E402

APPROACH_ASK = ("{puzzle}\n\nDo not solve this yet. List {k} genuinely different strategies someone could use to find the "
                "answer to this kind of position (different ways to organize the search, not different guesses). For each, "
                "give one line and your estimated probability of using it yourself. Format exactly:\n"
                "1. <strategy> (p=0.xx)\n2. ...")
HINTED = "{puzzle}\n\nSolve it using this approach: {approach}"
LEAK = re.compile(r"(as instructed|the (given|suggested|specified|provided|requested) (approach|strategy)|"
                  r"(follow|following|use|using) (this|the) (approach|strategy)|the approach (says|suggests|asks)|"
                  r"instructed approach|approach (i was|we were) (given|asked))", re.I)


def parse_approaches(text: str, k: int) -> list[str]:
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+?)\s*(\(p\s*=\s*[0-9.]+\))?\s*$", line)
        if m and len(m.group(1)) > 8: out.append(m.group(1).strip(" *"))
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA of the current model (none: the base model)")
    ap.add_argument("--arm", choices=["iid", "strategic"], required=True)
    ap.add_argument("--puzzles", type=Path, default=ROOT / "llm_train/puzzles/grpo_train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--keep", type=int, default=2, help="verified solutions kept per puzzle")
    ap.add_argument("--mix", default="win_pos=0.4,win_neg=0.2,block=0.2,forced=0.2")
    ap.add_argument("--max-think", type=int, default=6144); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-neg-ratio", type=float, default=0.6); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--select", choices=["random", "shortest"], default="random",
                    help="which verified solutions to keep per puzzle (random: natural finishes first, no length preference)")
    a = ap.parse_args(); t0 = time.time(); rng = random.Random(a.seed); a.out.mkdir(parents=True, exist_ok=True)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from expert.engine import State
    from llm_bench.run import score
    from llm_bench.text import prompt

    rows = [json.loads(l) for l in open(a.puzzles)]; rng.shuffle(rows)        # same seed in both arms: same puzzles
    share = {k_: float(v) for k_, v in (p.split("=") for p in a.mix.split(","))}
    picked = []
    for sub, f in share.items():
        picked += [r for r in rows if r["subtask"] == sub][: int(round(f * a.n))]
    texts = []
    for r in picked:
        it = r["item"]; s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"])
        texts.append(prompt(it["task"], s))
    print(f"{len(picked)} puzzles: {dict(Counter(r['subtask'] for r in picked))}; arm {a.arm}, k={a.k}", flush=True)

    llm = LLM(a.model, max_model_len=a.max_think + 2048, gpu_memory_utilization=0.90, enable_prefix_caching=True,
              enable_lora=a.adapter is not None, max_lora_rank=32, seed=a.seed)
    lora = LoRARequest("policy", 1, str(a.adapter)) if a.adapter else None
    tok = llm.get_tokenizer()
    tmpl = lambda content, think=True: tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                                add_generation_prompt=True, enable_thinking=think)
    heads = [tmpl(t) for t in texts]                                           # what the model is trained on

    # sampling: one job per (puzzle, slot); strategic slots carry an approach as a hidden instruction
    jobs = []                                                                  # (puzzle i, approach or None, prompt)
    if a.arm == "strategic":
        asks = llm.generate([tmpl(APPROACH_ASK.format(puzzle=t, k=a.k), think=False) for t in texts],
                            SamplingParams(temperature=1.0, top_p=0.95, max_tokens=500), lora_request=lora)
        n_short = 0
        for i, o in enumerate(asks):
            apps = parse_approaches(o.outputs[0].text, a.k)
            n_short += len(apps) < a.k
            for j in range(a.k):
                appr = apps[j] if j < len(apps) else None                     # too few strategies: plain sample
                jobs.append((i, appr, tmpl(HINTED.format(puzzle=texts[i], approach=appr)) if appr else heads[i]))
        print(f"approach lists in {time.time() - t0:.0f}s ({n_short} puzzles gave fewer than {a.k})", flush=True)
        (a.out / "approaches.jsonl").write_text("".join(
            json.dumps({"id": picked[i]["id"], "text": o.outputs[0].text}) + "\n" for i, o in enumerate(asks)))
    else:
        jobs = [(i, None, heads[i]) for i in range(len(picked)) for _ in range(a.k)]
    # no per-request seed: identical prompts (the iid arm's k samples) would get identical outputs; the engine seed
    # (LLM(seed=...)) keeps the run reproducible
    sp = SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=a.max_think)
    gen = llm.generate([p for _, _, p in jobs], sp, lora_request=lora)
    print(f"{len(gen)} samples in {time.time() - t0:.0f}s", flush=True)

    # verify: finished answers directly; run-on samples by forced answers at cuts (forcing uses the hinted prompt
    # the sample was generated under, so the cut is judged in its own context)
    cand = defaultdict(list)                                                  # i -> [(think_tokens, think, answer, approach)]
    stats = Counter(); force_jobs = []                                        # (sample index, cut, prefix)
    for si, ((i, appr, p), o) in enumerate(zip(jobs, gen)):
        out = o.outputs[0]; text = out.text
        if "</think>" in text:
            th, ans = text.split("</think>", 1)
            ans_line = ans.strip().split("\n")[-1] if ans.strip() else ""
            if score(picked[i]["item"], ans)["correct"]:
                if appr and LEAK.search(th): stats["dropped: refers to its instruction"] += 1; continue
                cand[i].append((len(out.token_ids), th.rstrip("\n"), ans_line, appr, True)); stats["kept: finished"] += 1
            else:
                stats["finished wrong"] += 1
            continue
        stats["cut off"] += 1
        for c in CUTS:
            if c <= len(out.token_ids): force_jobs.append((si, c, snap(tok.decode(out.token_ids[:c]))))
    forced = llm.generate([jobs[si][2] + pre + STOP + "ANSWER:" for si, _, pre in force_jobs],
                          SamplingParams(temperature=0.0, max_tokens=16), lora_request=lora)
    by_sample = defaultdict(dict)                                             # si -> {cut: (ok, prefix, answer)}
    for (si, c, pre), o in zip(force_jobs, forced):
        i = jobs[si][0]; ans = "ANSWER:" + o.outputs[0].text.split("\n")[0]
        by_sample[si][c] = (bool(score(picked[i]["item"], ans)["correct"]), pre, ans)
    for si, g in by_sample.items():
        i, appr, _ = jobs[si]; cs = sorted(g)
        for kk, c in enumerate(cs):
            if picked[i]["subtask"] == "win_neg" and c < 2048: continue      # no short NO WIN cuts (no search shown)
            if all(g[d][0] for d in cs[kk:]):
                ok, pre, ans = g[c]
                if appr and LEAK.search(pre): stats["dropped: refers to its instruction"] += 1; break
                cand[i].append((c, pre, ans, appr, False)); stats["kept: stable cut"] += 1
                break

    # select up to --keep per puzzle. random (default): natural finishes first, in random order, then stable cuts in
    # random order. shortest: the round-1 rule, which taught a length prior (r1 iid: forced 2-jump wins 74% -> 42%,
    # stopping early and saying NO WIN on harder wins). strategic: from different strategies.
    chosen = []
    for i, cs in cand.items():
        seen_appr = set(); got = 0
        if a.select == "shortest":
            order = sorted(cs, key=lambda x: x[0])
        else:
            nat = [x for x in cs if x[4]]; cut = [x for x in cs if not x[4]]; rng.shuffle(nat); rng.shuffle(cut)
            order = nat + cut
        for L, th, ans, appr, natural in order:
            if a.arm == "strategic" and appr is not None and appr in seen_appr: continue
            seen_appr.add(appr); chosen.append((i, L, th, ans, appr, natural)); got += 1
            if got >= a.keep: break
    pos = [x for x in chosen if picked[x[0]]["subtask"] == "win_pos"]
    neg = [x for x in chosen if picked[x[0]]["subtask"] == "win_neg"]
    cap = int(a.max_neg_ratio * len(pos))
    if len(neg) > cap:
        rng.shuffle(neg); drop = set(map(id, neg[cap:])); chosen = [x for x in chosen if id(x) not in drop]
    with open(a.out / "sft.jsonl", "w") as f:
        for i, L, th, ans, appr, natural in chosen:
            f.write(json.dumps({"prompt": heads[i], "completion": th + STOP + ans.strip() + tok.eos_token,
                                "id": picked[i]["id"], "subtask": picked[i]["subtask"], "think_tokens": L,
                                "approach": appr, "natural": natural}) + "\n")
    solved = Counter(picked[i]["subtask"] for i in cand)
    total = Counter(r["subtask"] for r in picked)
    summary = {"arm": a.arm, "select": a.select, "natural_share": round(sum(x[5] for x in chosen) / max(len(chosen), 1), 3),
               "k": a.k, "puzzles": len(picked), "examples": len(chosen),
               "examples_by_subtask": dict(Counter(picked[x[0]]["subtask"] for x in chosen)),
               "puzzles_with_a_verified_solution": {s: f"{solved[s]}/{total[s]}" for s in total},
               "distinct_strategies_kept": len({x[4] for x in chosen if x[4]}),
               "median_think_tokens": sorted(x[1] for x in chosen)[len(chosen) // 2] if chosen else None,
               "sample_outcomes": dict(stats), "seconds": round(time.time() - t0)}
    (a.out / "stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
