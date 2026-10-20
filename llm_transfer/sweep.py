#!/usr/bin/env python3
"""Budget sweep: accuracy as a function of the thinking budget, from ONE long rollout per item.

The prompt never states the cap, so a run capped at B tokens is the first B tokens of a run with a larger cap. Each item
is generated once at --max-tokens (24k by default); then, for every cut B in --cuts:
  finished@B  the model's own answer, if it finished within B tokens (else wrong);
  forced@B    the best answer within B: the finished answer if it finished within B, else the rollout cut at B tokens +
              "Time is up ..." + "ANSWER:" (greedy), as llm_transfer.run / llm_bench --truncate-at.
A curve that shifts LEFT across models is efficiency (the same accuracy for fewer tokens); a higher PLATEAU is skill
(solved at a budget where the other model never solves it). The fixed 6k cap could not tell these apart on the bridge:
every model was cut off on 92-100% of 3-4 jump backward traps.

Generations are appended to <out>.gen.jsonl as they arrive (a restarted run skips the ids already there), so a
disconnect loses only the requests in flight; the scored rows go to llm_transfer/results/<name>_sweep.jsonl.

  python -m llm_transfer.sweep --model grpo --name grpo --families bridge --gen-dir DRIVE/transfer/sweep
  python -m llm_transfer.sweep_report grpo r14 base warm
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm_bench.run import FORCE, call  # noqa: E402
from llm_transfer.run import answer_line, score  # noqa: E402

CUTS = (2048, 4096, 6144, 8192, 12288, 16384, 24576)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=ROOT / "llm_transfer/data/suite_v1.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000/v1"); ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--families", default="bridge"); ap.add_argument("--cuts", default=",".join(map(str, CUTS)))
    ap.add_argument("--max-tokens", type=int, default=0, help="generation cap (default: the largest cut)")
    ap.add_argument("--concurrency", type=int, default=96); ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-dir", type=Path, default=ROOT / "llm_transfer/results", help="where raw generations persist")
    a = ap.parse_args()
    cuts = sorted(int(c) for c in a.cuts.split(",")); cap = a.max_tokens or cuts[-1]
    items = [json.loads(l) for l in open(a.suite)]
    items = [it for it in items if it["family"] in a.families.split(",")]
    by_id = {it["id"]: it for it in items}

    # 1. one long rollout per item (resumable)
    a.gen_dir.mkdir(parents=True, exist_ok=True); gpath = a.gen_dir / f"{a.name}_sweep.gen.jsonl"
    done = {}
    if gpath.exists():
        for l in open(gpath):
            g = json.loads(l)
            if g["id"] in by_id and g.get("cap") == cap and not g.get("error"): done[g["id"]] = g
    todo = [it for it in items if it["id"] not in done]
    print(f"{a.name}: {len(items)} items ({len(done)} already generated), cap {cap}, cuts {cuts}", flush=True)
    gen = argparse.Namespace(claude_code=False, api_key_env="OPENAI_API_KEY", completions=False, model=a.model,
                             max_tokens=cap, temperature=0.6, top_p=0.95, top_k=20, seed=a.seed, thinking=True,
                             base_url=a.base_url, timeout=a.timeout)
    t0 = time.time()

    def one(it):
        out = call(gen, it["prompt"])
        return {"id": it["id"], "cap": cap, "prompt_sha": hashlib.sha1(it["prompt"].encode()).hexdigest()[:12],
                "reasoning": out.get("reasoning") or "", "reply": out.get("reply") or "",
                "finish_reason": out.get("finish_reason"), "tokens": out.get("usage", {}).get("completion_tokens"),
                "error": out.get("error")}

    with open(gpath, "a") as f, cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for i, g in enumerate(ex.map(one, todo), 1):
            f.write(json.dumps(g) + "\n"); f.flush()
            if not g.get("error"): done[g["id"]] = g
            if i % 25 == 0: print(f"  generated {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    missing = [i for i in by_id if i not in done]
    if missing: print(f"WARNING: {len(missing)} items failed to generate (rerun to retry): {missing[:5]}")

    # 2. score every cut: finished within B, or forced from the rollout cut at B
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    force = argparse.Namespace(**{**vars(gen), "completions": True, "max_tokens": 60, "temperature": 0.0})
    rows, jobs = [], []
    for iid, g in done.items():
        it = by_id[iid]
        reply = g["reply"]; ans_part = reply.rsplit("</think>", 1)[1] if "</think>" in reply else (reply if g["reasoning"] else "")
        own = answer_line(ans_part); ok, cat = score(it, own)
        body = g["reasoning"] + reply; ids = tok(body, add_special_tokens=False)["input_ids"]
        n = g["tokens"] or len(ids)
        r = {"id": iid, "prompt_sha": g["prompt_sha"], "family": it["family"], "dial": it["dial"], "absent": it["absent"],
             "tokens": n, "finish_reason": g["finish_reason"], "answer": own, "correct": ok, "category": cat, "at": {}}
        for c in cuts:
            if cat != "unparsed" and n <= c:                     # finished on its own within the budget
                r["at"][c] = {"finished": ok, "forced": ok, "fcat": cat}
            else:
                r["at"][c] = {"finished": False}
                cut_body = tok.decode(ids[:c]) if n > c else body
                jobs.append((r, c, it, cut_body))
        rows.append(r)

    def forced(job):
        r, c, it, cut_body = job
        head = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        out = call(force, None, raw=head + cut_body + FORCE["</think>" not in cut_body] + "ANSWER:")
        return job, (out.get("reply") or "").split("\n")[0].strip()

    print(f"forcing {len(jobs)} (item, cut) answers", flush=True)
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for (r, c, it, _), fa in ex.map(forced, jobs):
            ok, cat = score(it, fa); r["at"][c].update(forced=ok, fcat=cat, forced_answer=fa)

    out_dir = ROOT / "llm_transfer/results"; out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{a.name}_sweep.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} items scored at {len(cuts)} budgets in {time.time() - t0:.0f}s -> {out_dir / a.name}_sweep.jsonl")
    from llm_transfer.sweep_report import curves
    curves([a.name], group_by="all")


if __name__ == "__main__":
    main()
