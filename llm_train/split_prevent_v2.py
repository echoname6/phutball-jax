#!/usr/bin/env python3
"""Finish the regenerated prevent puzzles (llm_train/make_prevent_v2.py): measure what simple rules still solve, then
split them into test / validation / training.

Shortcut report (the goal: no simple rule above ~33%):
  best fixed square     the single square at a fixed offset from the ball (in the defender's frame, up to 3 away) that
                        saves the most puzzles; a model that always plays there gets this score
  any single jump       the first legal jump of the ball (stop after one jump); jump toward own goal likewise
  random placement      expected score of a uniformly random legal placement

Split (seeded): --test to llm_bench/data/prevent_v2_test.jsonl (llm_bench format; report with
llm_bench.run --bench), --val appended to the curriculum validation set (+ its id list), the rest added to the pool
(pool_v3 + them -> pool_v4).

  python -m llm_train.split_prevent_v2 --src llm_train/puzzles/prevent_v2.jsonl.gz
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(src: Path) -> list[dict]:
    """The merged file, or (run still going) its per-worker part files."""
    from llm_train.build_traces import key_of
    if src.exists(): rows = [json.loads(l) for l in gzip.open(src, "rt")]
    else: rows = [json.loads(l) for p in sorted(glob.glob(f"{src}.part*.jsonl")) for l in open(p)]
    seen, out = set(), []
    for r in rows:
        k = key_of(r["item"]["board"], r["item"]["ball"], r["item"]["player"])
        if k not in seen: seen.add(k); out.append(r)
    return out


def shortcuts(items: list[dict]) -> dict:
    from expert.engine import BALL, MAN, State
    from llm_bench.run import jump_landings, score
    from llm_bench.text import sq
    n = len(items); rel = Counter(); res = Counter(); rand = 0.0
    for it in items:
        s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"]); C = s.cols
        br, bc = divmod(s.ball, C); fw = -1 if s.player == 1 else 1
        saves = {int(x[1:]) * C + ord(x[0]) - 97 for x in it["answers"]["placements"]}
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r, c = br + dr * fw, bc + dc * fw                            # defender's frame (mirrored for player 2)
                if 1 <= r <= s.rows - 2 and 0 <= c < C and r * C + c in saves: rel[(dr, dc)] += 1
        js = jump_landings(s.board, s.ball, s.rows, s.cols)
        if js:
            res["any single jump"] += score(it, f"ANSWER: JUMP {sq(js[0][0], C)}")["correct"]
            toward = [l for l, _ in js if (l // C - br) * fw > 0]
            res["single jump toward own goal"] += bool(toward) and score(it, f"ANSWER: JUMP {sq(toward[0], C)}")["correct"]
        empty = [i for i in range(C, (s.rows - 1) * C) if s.board[i] not in (BALL, MAN)]
        rand += len(saves & set(empty)) / len(empty)
    (off, k), = rel.most_common(1) or [((None, None), 0)]
    return {"puzzles": n, "best fixed square (rows toward own goal, cols)": [list(off), round(k / n, 3)],
            "any single jump": round(res["any single jump"] / n, 3),
            "single jump toward own goal": round(res["single jump toward own goal"] / n, 3),
            "random placement": round(rand / n, 3),
            "top fixed squares": [[list(o), round(v / n, 3)] for o, v in rel.most_common(5)]}


def mirror(row: dict) -> dict:
    """The left-right mirror of a puzzle: the rules are symmetric under reflecting the columns, so the labels mirror
    exactly (each mirrored answer is re-checked with the engine in main)."""
    it = row["item"]; R, C = it["rows"], it["cols"]
    f = lambda i: (i // C) * C + (C - 1 - i % C)
    fs = lambda t: chr(ord("a") + C - 1 - (ord(t[0]) - ord("a"))) + t[1:]
    b = [0] * (R * C)
    for i, v in enumerate(it["board"]): b[f(i)] = v
    mid = row["id"] + "-m"
    item = {**it, "id": mid, "board": b, "ball": f(it["ball"]),
            "answers": {"placements": sorted(fs(t) for t in it["answers"]["placements"]),
                        "saving_first_jumps": sorted(fs(t) for t in it["answers"].get("saving_first_jumps", []))},
            "meta": {**it["meta"], "mirror_of": row["id"]}}
    return {**row, "id": mid, "item": item}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "llm_train/puzzles/prevent_v2.jsonl.gz", help="strict puzzles")
    ap.add_argument("--relaxed", type=Path, default=ROOT / "llm_train/puzzles/prevent_v2_relaxed.jsonl.gz",
                    help="puzzles where only the three defender-side squares are ruled out (a sideways escape may save)")
    ap.add_argument("--total", type=int, default=400, help="puzzles after mirroring (strict first, then relaxed)")
    ap.add_argument("--test", type=int, default=200); ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--report-only", action="store_true"); ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    from expert.engine import State
    from llm_bench.run import score
    from llm_bench.text import prompt
    from llm_train.build_traces import key_of
    from llm_train.make_prevent_v2 import neighbours
    strict = load(a.src)
    keys = {key_of(r["item"]["board"], r["item"]["ball"], r["item"]["player"]) for r in strict}
    relaxed = [r for r in (load(a.relaxed) if (a.relaxed.exists() or list(a.relaxed.parent.glob(a.relaxed.name + ".part*"))) else [])
               if key_of(r["item"]["board"], r["item"]["ball"], r["item"]["player"]) not in keys]
    rng = random.Random(a.seed)
    def saving_neighbours(r):
        it = r["item"]; s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"]); C = s.cols
        saves = {int(x[1:]) * C + ord(x[0]) - 97 for x in it["answers"]["placements"]}
        return len(saves & set(neighbours(s)))
    rng.shuffle(relaxed); relaxed.sort(key=saving_neighbours)               # fewest saving squares next to the ball first
    need = max(0, a.total // 2 - len(strict))
    for r in relaxed: r["item"]["meta"]["source"] = r["item"]["meta"].get("source", "") + " (relaxed)"
    base = strict + relaxed[:need]
    print(f"strict {len(strict)}, relaxed candidates {len(relaxed)}, taking {min(need, len(relaxed))} relaxed -> {len(base)} base puzzles")
    pairs, bad = [], 0
    for r in base:
        m = mirror(r)
        mk = key_of(m["item"]["board"], m["item"]["ball"], m["item"]["player"])
        if mk in keys or mk == key_of(r["item"]["board"], r["item"]["ball"], r["item"]["player"]): continue   # symmetric
        if not all(score(m["item"], f"ANSWER: PLACE {t}")["correct"] for t in m["item"]["answers"]["placements"]): bad += 1; continue
        keys.add(mk); pairs.append((r, m))
    print(f"{len(pairs)} mirror pairs ({2 * len(pairs)} puzzles); mirrored labels failing the engine check: {bad}")
    allrows = [x for p_ in pairs for x in p_]
    rep = shortcuts([r["item"] for r in allrows])
    print(json.dumps({k: v for k, v in rep.items() if k != "top fixed squares"}, indent=1)); print("top fixed squares:", rep["top fixed squares"])
    if a.report_only: return
    rng.shuffle(pairs)
    tp, vp = a.test // 2, a.val // 2
    test = [x for p_ in pairs[:tp] for x in p_]; val = [x for p_ in pairs[tp: tp + vp] for x in p_]
    train = [x for p_ in pairs[tp + vp:] for x in p_]
    bench_row = lambda it: {**{k: it[k] for k in ("id", "task", "rows", "cols", "board", "ball", "player", "answers")},
                            "meta": {**it["meta"], "bucket": "prevent"},
                            "prompt": prompt("prevent", State(it["rows"], it["cols"], it["board"], it["ball"], it["player"]), "ascii")}
    with open(ROOT / "llm_bench/data/prevent_v2_test.jsonl", "w") as f:
        for r in test: f.write(json.dumps(bench_row(r["item"])) + "\n")
    vpath, vids = ROOT / "llm_train/puzzles/val_v3.jsonl", ROOT / "llm_train/puzzles/val_v3_ids.json"
    kept = [l for l in open(vpath) if json.loads(l)["task"] != "prevent"]               # re-running replaces prevent
    with open(vpath, "w") as f:
        for l in kept: f.write(l)
        for r in val: f.write(json.dumps(bench_row(r["item"])) + "\n")
    old = [i for i in json.loads(vids.read_text()) if not i.startswith("prevent")]
    vids.write_text(json.dumps(sorted(set(old) | {r["id"] for r in val})))
    with gzip.open(ROOT / "llm_train/puzzles/pool_v3.jsonl.gz", "rt") as f, \
            gzip.open(ROOT / "llm_train/puzzles/pool_v4.jsonl.gz", "wt") as g:
        for l in f: g.write(l)
        for r in train: g.write(json.dumps({**r, "bucket": "prevent"}) + "\n")
    print(f"test {len(test)} -> llm_bench/data/prevent_v2_test.jsonl | val {len(val)} -> val_v3 | "
          f"train {len(train)} -> pool_v4 (= pool_v3 + them); mirror pairs never split")


if __name__ == "__main__":
    main()
