#!/usr/bin/env python3
"""Build the stage-1 SFT set: benchmark-format prompts with engine-derived thinking traces (llm_train/traces.py).

Same puzzle families and generators as llm_bench/build.py, from disjoint seeds (--seed + worker; the benchmark used
9001, the network's training pools 0-3), and every position in the benchmark or expert_data/puzzle_eval.npz is
rejected. Each example's answer is checked with llm_bench.run.score against its engine labels before it is written.
forced: by default positions where a square next to the ball is a correct answer are dropped (--keep-adjacent-forced
to keep them): the one-rule shortcut "place next to the ball" solves a third of the raw forced family.

Output (llm_train/data/traces.jsonl), one example per line:
  {"id", "task", "prompt", "think", "answer", "messages": [user prompt, assistant {"reasoning_content": think,
   "content": answer}], "text_assistant": "<think>\\n...\\n</think>\\n\\nANSWER: ...", "meta", "item"}
"item" is a benchmark-format item (board, answers) so the same scorer and renderer apply.

  ~/Projects/phutball/venv/bin/python -m llm_train.build_traces --per-task 50 --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TASKS = ("win_pos", "win_neg", "forced", "block", "prevent")


def key_of(board, ball, player) -> tuple:
    return (tuple(int(v) for v in board), int(ball), int(player))


def bench_keys() -> set:
    keys = set()
    for l in open(ROOT / "llm_bench/data/bench.jsonl"):
        it = json.loads(l); keys.add(key_of(it["board"], it["ball"], it["player"]))
    for r in np.load(ROOT / "expert_data/puzzle_eval.npz", allow_pickle=True)["recs"]:
        keys.add(key_of(r["board"], r["ball"], r["player"]))
    return keys


def worker(args):
    wid, seed, quota, keep_adj, deadline = args
    from expert.engine import State
    from expert.puzzle_data import CELLS, Generator
    from expert.puzzle_place import CapHit, make_block, make_forced, make_prevent, make_threat, threat_placements
    from llm_bench.run import score
    from llm_bench.text import prompt, sq
    from llm_train import traces as T

    gen = Generator(seed=seed); rng = random.Random(seed); avoid = bench_keys()
    win_cells = list(CELLS); place_cells = [c for c in CELLS if c[1] <= 3]
    need = {t: quota for t in TASKS}; out = []; stats = Counter(); k = 0

    def emit(task, s, answers, meta, res):
        if res is None: stats[f"{task}: skipped (search cap / label mismatch)"] += 1; return False
        think, ans = res
        btask = "win" if task.startswith("win") else task
        item = {"id": f"{task}-{wid}-{k}", "task": btask, "rows": s.rows, "cols": s.cols, "board": list(map(int, s.board)),
                "ball": int(s.ball), "player": int(s.player), "answers": answers, "meta": meta}
        sc = score(item, ans)
        if not sc["correct"]: stats[f"{task}: answer failed the scorer ({sc['outcome']})"] += 1; return False
        p = prompt(btask, s)
        out.append({"id": item["id"], "task": btask, "subtask": task, "prompt": p, "think": think, "answer": ans,
                    "messages": [{"role": "user", "content": p},
                                 {"role": "assistant", "reasoning_content": think, "content": ans}],
                    "text_assistant": f"<think>\n{think}\n</think>\n\n{ans}", "meta": meta, "item": item})
        need[task] -= 1; stats[f"{task}: ok"] += 1; return True

    while any(v > 0 for v in need.values()) and time.time() < deadline:
        k += 1
        try:
            if need["win_pos"] > 0 and k % 3 == 0:
                cell = win_cells[rng.randrange(len(win_cells))]
                s = gen.puzzle(cell)
                if s is None or key_of(s.board, s.ball, s.player) in avoid: continue
                emit("win_pos", s, {"win": True}, {"family": cell[0], "J": cell[1]}, T.trace_win(s))
                continue
            base = gen.puzzle(place_cells[k % len(place_cells)])
            if base is None: continue
            if need["win_neg"] > 0 and k % 3 == 1:
                r = make_threat(base, rng)
                if r and key_of(r[0].board, r[0].ball, r[0].player) not in avoid:
                    emit("win_neg", r[0], {"win": False}, {"source": "near-miss threat"}, T.trace_win(r[0]))
                continue
            if need["block"] > 0:
                r = make_block(base)
                if r and key_of(r[0].board, r[0].ball, r[0].player) not in avoid:
                    s, tw, meta = r
                    emit("block", s, {"placements": sorted(sq(p, s.cols) for p in tw)}, meta, T.trace_block(s, set(tw)))
                    continue
            if need["prevent"] > 0 and k % 2 == 0:
                r = make_prevent(base, rng)
                if r and key_of(r[0].board, r[0].ball, r[0].player) not in avoid:
                    s, pl, jl, meta = r
                    emit("prevent", s, {"placements": sorted(sq(p, s.cols) for p in pl),
                                        "saving_first_jumps": sorted(sq(l, s.cols) for l in jl)},
                         {k_: int(v) for k_, v in meta.items()}, T.trace_prevent(s, set(pl), set(jl)))
                    continue
            if need["forced"] > 0:
                r = make_forced(base, rng)
                if r and key_of(r[0].board, r[0].ball, r[0].player) not in avoid:
                    s, tw, meta = r
                    if not keep_adj and set(tw) & set(T.neighbours(s.ball, s.rows, s.cols)):
                        stats["forced: dropped (adjacent answer)"] += 1; continue
                    threats = sorted(threat_placements(s))
                    emit("forced", s, {"placements": sorted(sq(p, s.cols) for p in tw)},
                         {k_: int(v) for k_, v in meta.items()}, T.trace_forced(s, set(tw), threats))
        except CapHit:
            stats["generator search cap"] += 1
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=50, help="examples per subtask (win_pos, win_neg, forced, block, prevent)")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--keep-adjacent-forced", action="store_true"); ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--out", type=Path, default=ROOT / "llm_train/data/traces.jsonl")
    ap.add_argument("--tokenizer", default=None, help="path to a tokenizer.json to count tokens (else ~chars/3.5)")
    a = ap.parse_args(); t0 = time.time()
    quota = -(-a.per_task // a.workers); deadline = t0 + a.minutes * 60
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(worker, [(w, a.seed + w, quota, a.keep_adjacent_forced, deadline) for w in range(a.workers)])
    rows = [r for out, _ in res for r in out]; stats = sum((st for _, st in res), Counter())
    seen, uniq = set(), []
    for r in rows:
        kk = key_of(r["item"]["board"], r["item"]["ball"], r["item"]["player"])
        if kk not in seen: seen.add(kk); uniq.append(r)
    count = (lambda t: len(t) / 3.5)
    if a.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer); count = lambda t: len(tok.encode(t).ids)
    for r in uniq: r["think_tokens"] = int(count(r["think"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in uniq: f.write(json.dumps(r) + "\n")
    print(f"{len(uniq)} examples ({len(rows) - len(uniq)} duplicates dropped) in {time.time() - t0:.0f}s -> {a.out}")
    for k_, v in sorted(stats.items()): print(f"  {k_}: {v}")
    print("thinking tokens" + (" (tokenizer)" if a.tokenizer else " (approx chars/3.5)") + ": median / p90 / max")
    for t in TASKS:
        L = sorted(r["think_tokens"] for r in uniq if r["subtask"] == t)
        if L: print(f"  {t:8s} n={len(L):4d}  {statistics.median(L):7.0f} {L[int(0.9 * (len(L) - 1))]:7d} {L[-1]:7d}")


if __name__ == "__main__":
    main()
