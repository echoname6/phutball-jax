#!/usr/bin/env python3
"""Transfer suite v1: does phutball training carry over to multi-hop chains, state tracking and code dependence?

Every item is one prompt that ends "End your reply with one line: ANSWER: ...", one exact answer, a family, a difficulty
dial (hops, operations, context length, ...) and `absent` (the right answer is "there is none": NO WIN, NONE, NO). The
suite is a ladder of distance from the training task (phutball win puzzles, ascii+men board format):

  bridge     the SAME validation win puzzles (val_v3, never trained on) rewritten without the board: named lanes and
             numbered tiers instead of a grid and squares, "stones / token / leap / home tiers" instead of
             "men / ball / jump / goal", tiers counted from the other side. Same logic, same answers (scored by the
             engine), new surface: does the skill survive a change of format?
  pointer    follow k links ("<a> leads to <b>") through shuffled lines with decoy chains; some chains are broken
             (answer NONE).
  path       a directed graph as text; give a path from S to T, or NONE when there is none (exhaustive search).
  boxes      7 boxes, n put / remove / move operations (half the sets REMOVE-heavy: the operation a 2026 mechanistic
             study found fragile in LMs, and the one phutball jumps train - every jump removes the men it crosses).
  ruler_vt   RULER-style variable tracking: an assignment chain (VAR X = 41527, VAR Y = VAR X, ...) inside filler text;
             list every variable that holds the value.
  babilong   BABILong qa2 / qa3 (2 / 3 supporting facts) at 0k-8k tokens of distractor text, qa19 path finding.
  cogniload  CogniLoad state-update puzzles at 20 / 50 / 100 statements.
  core       CoRe static analysis (C / Java / Python): does a data / control / information-flow dependence exist? (yes/no;
             the released file has no ground truth for the trace or the source list, so only existence is scored).

  python -m llm_transfer.build --out llm_transfer/data/suite_v1.jsonl      (downloads BABILong / CogniLoad / CoRe once)
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from expert.engine import BALL, MAN, State, jump_landings, winner_at  # noqa: E402

CACHE = ROOT / "llm_transfer/data/cache"
END = "\n\nEnd your reply with one line, exactly:\n"


def fetch(url: str, name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True); p = CACHE / name
    if not p.exists():
        print("download", url, flush=True); urllib.request.urlretrieve(url, p)
    return p


# --- bridge: phutball without the board ----------------------------------------------------------------------------
LANES = ["cedar", "onyx", "maple", "quartz", "willow", "garnet", "heron", "lagoon", "basalt", "fennel", "kestrel",
         "ivory", "dune", "nectar", "tundra", "saffron", "violet", "raven", "plume", "zephyr"]


def tier_of(r: int, rows: int) -> int: return rows - r                 # tiers counted from the south: row 0 -> tier R


def place(i: int, s: State) -> str:
    r, c = divmod(i, s.cols); return f"{LANES[c]}-{tier_of(r, s.rows)}"


def parse_place(t: str, s: State) -> int | None:
    t = t.strip().strip(",.;").lower()
    if "-" not in t: return None
    lane, tier = t.rsplit("-", 1)
    if lane not in LANES[:s.cols] or not tier.isdigit(): return None
    r = s.rows - int(tier); c = LANES.index(lane)
    return r * s.cols + c if 0 <= r < s.rows else None


def bridge_prompt(it: dict) -> str:
    s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"])
    R = s.rows; me = s.player
    home = (R, R - 1) if me == 1 else (2, 1)                            # rows 0,1 (player 1) / R-2,R-1 (player 2)
    away = (2, 1) if me == 1 else (R, R - 1)
    stones = [place(i, s) for i, v in enumerate(s.board) if v == MAN and i != s.ball]
    leaps = jump_landings(s.board, s.ball, s.rows, s.cols)
    first = "; ".join(f"{place(l, s)} (over {', '.join(place(j, s) for j in jd)})" for l, jd in leaps) or "none"
    return (
        f"A field has {s.cols} lanes, from west to east: {', '.join(LANES[:s.cols])}; and {R} tiers, numbered 1 (south) to "
        f"{R} (north). A place is written lane-tier, e.g. {LANES[2]}-7. Some places hold a stone; one place holds the token.\n"
        f"Rules of a turn: the mover may LEAP the token in a straight line in any of the 8 directions (along a lane, along a "
        f"tier, or diagonally), over one or more stones on consecutive adjacent places in that line, landing on the first "
        f"place after them that holds no stone (that place must exist on the field). Every stone leapt over is removed. "
        f"From the landing place the mover may leap again, in any direction, or stop. The mover wins the moment the token "
        f"lands in its home tiers ({home[0]} and {home[1]}); landing in the opposing home tiers ({away[0]} and {away[1]}) "
        f"loses at once.\n\nPosition: the token is at {place(s.ball, s)}. Stones ({len(stones)}): {', '.join(stones)}.\n"
        f"Leaps available from the token's place now: {first}.\n\n"
        f"Question: can the mover win THIS turn by leaping? If yes, give one complete winning sequence of leaps as its "
        f"landing places, in order. If no sequence of leaps wins this turn, say so.\nEnd your reply with one line, "
        f"exactly in one of these forms:\nANSWER: LEAPS <place> <place> ...\nANSWER: NONE")


def build_bridge(per_bucket: int):
    rows = [json.loads(l) for l in open(ROOT / "llm_train/puzzles/val_v3.jsonl")]
    out, n = [], Counter()
    for it in rows:
        if it["task"] != "win": continue
        b = it["meta"]["bucket"]
        if n[b] >= per_bucket: continue
        n[b] += 1
        out.append({"id": f"bridge-{it['id']}", "family": "bridge", "dial": {"bucket": b, "J": it["meta"].get("J", 0)},
                    "kind": "bridge", "prompt": bridge_prompt(it), "item": it, "absent": not it["answers"]["win"],
                    "source_id": it["id"]})
    return out


# --- pointer chasing --------------------------------------------------------------------------------------------------
def words(rng, n, k=6):
    seen = set()
    while len(seen) < n:
        seen.add("".join(rng.choice(string.ascii_lowercase) for _ in range(k)))
    return list(seen)


def build_pointer(rng, hops=(1, 2, 4, 6, 8), per=30, decoys=6):
    out = []
    for h in hops:
        for k in range(per):
            names = words(rng, (h + 1) * (decoys + 1) + 10)
            chain = names[:h + 1]; rest = names[h + 1:]
            lines = [f"{chain[i]} leads to {chain[i + 1]}." for i in range(h)]
            broken = k % 4 == 0 and h > 1                                   # a missing link: answer NONE
            if broken: lines.pop(rng.randrange(1, h))
            for d in range(decoys):                                         # decoy chains of the same length
                dl = rest[d * (h + 1):(d + 1) * (h + 1)]
                lines += [f"{dl[i]} leads to {dl[i + 1]}." for i in range(len(dl) - 1)]
            rng.shuffle(lines)
            ans = "NONE" if broken else chain[-1]
            out.append({"id": f"pointer-h{h}-{k}", "family": "pointer", "dial": {"hops": h}, "kind": "value",
                        "prompt": "Each line says where a name leads.\n\n" + "\n".join(lines) +
                        f"\n\nStart at {chain[0]} and follow the links exactly {h} time{'s' if h > 1 else ''}. Where do you end? "
                        f"If a link you need is missing, the answer is NONE." + END + "ANSWER: <name>  (or ANSWER: NONE)",
                        "answer": ans, "absent": broken})
    return out


# --- path finding -------------------------------------------------------------------------------------------------------
def build_path(rng, lengths=(2, 3, 4, 6), per=30, n_nodes=24, extra_edges=18):
    out = []
    for L in lengths:
        for k in range(per):
            names = words(rng, n_nodes, 5)
            path = names[:L + 1]; others = names[L + 1:]
            edges = {(path[i], path[i + 1]) for i in range(L)}
            none = k % 4 == 0
            if none: edges.discard((path[L - 1], path[L]))                  # cut the only route; decoys never reach T
            pool = names[:-1] if not none else [x for x in names if x != path[L]]
            while len(edges) < L + extra_edges:
                a, b = rng.sample(pool, 2)
                if b == path[L] or a == path[L]: continue                   # T reachable only via the planted path
                edges.add((a, b))
            T = path[L]
            reach = {path[0]}; frontier = [path[0]]                          # check: is T reachable? (exact answer)
            while frontier:
                x = frontier.pop()
                for a, b in edges:
                    if a == x and b not in reach: reach.add(b); frontier.append(b)
            exists = T in reach
            el = [f"{a} -> {b}" for a, b in edges]; rng.shuffle(el)
            out.append({"id": f"path-L{L}-{k}", "family": "path", "dial": {"length": L}, "kind": "path",
                        "prompt": "One-way roads between towns:\n" + "\n".join(el) +
                        f"\n\nQuestion: is there a route from {path[0]} to {T} along these one-way roads? If yes, give one "
                        f"route as the list of towns from {path[0]} to {T}. If there is no route, say NONE." + END +
                        f"ANSWER: {path[0]} <town> ... {T}  (or ANSWER: NONE)",
                        "edges": sorted(edges), "start": path[0], "target": T, "absent": not exists})
    return out


# --- boxes (entity tracking with removals) -----------------------------------------------------------------------------
ITEMS = ["apple", "bell", "coin", "drum", "egg", "flute", "glove", "hat", "ink", "jar", "key", "lamp", "map", "nut",
         "orb", "pen", "quill", "ring", "shoe", "toy"]


def build_boxes(rng, ops=(4, 8, 16, 24), per=30):
    out = []
    for n in ops:
        for k in range(per):
            heavy = k % 2 == 0                                              # REMOVE-heavy half
            boxes = {i: set(rng.sample(ITEMS, rng.randint(1, 3))) for i in range(1, 8)}
            init = " ".join(f"Box {i} contains {' and '.join(sorted(v)) or 'nothing'}." for i, v in boxes.items())
            lines = []
            for _ in range(n):
                full = [i for i in boxes if boxes[i]]
                op = rng.choices(["put", "remove", "move"], weights=[1, 3, 1] if heavy else [2, 1, 2])[0]
                if op != "put" and not full: op = "put"
                if op == "put":
                    i = rng.randint(1, 7); free = [x for x in ITEMS if not any(x in b for b in boxes.values())]
                    if not free: continue
                    x = rng.choice(free); boxes[i].add(x); lines.append(f"Put the {x} into Box {i}.")
                elif op == "remove":
                    i = rng.choice(full); x = rng.choice(sorted(boxes[i])); boxes[i].discard(x)
                    lines.append(f"Remove the {x} from Box {i}.")
                else:
                    i = rng.choice(full); j = rng.choice([b for b in boxes if b != i]); x = rng.choice(sorted(boxes[i]))
                    boxes[i].discard(x); boxes[j].add(x); lines.append(f"Move the {x} from Box {i} to Box {j}.")
            q = rng.randint(1, 7)
            out.append({"id": f"boxes-n{n}-{k}", "family": "boxes", "dial": {"ops": n, "remove_heavy": heavy}, "kind": "set",
                        "prompt": init + "\n\n" + "\n".join(lines) + f"\n\nQuestion: what does Box {q} contain now? List the "
                        f"items, or NOTHING if it is empty." + END + "ANSWER: <item>, <item>, ...  (or ANSWER: NOTHING)",
                        "answer": sorted(boxes[q]), "absent": not boxes[q]})
    return out


# --- RULER-style variable tracking ---------------------------------------------------------------------------------------
FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")


def build_ruler_vt(rng, hops=(2, 4, 8), lengths=(2000, 8000), per=20):
    out = []
    for h in hops:
        for L in lengths:
            for k in range(per):
                names = [n.upper() for n in words(rng, 2 * (h + 1) + 2, 5)]
                val = str(rng.randint(10000, 99999)); dval = str(rng.randint(10000, 99999))
                chain = names[:h + 1]; dec = names[h + 1:2 * (h + 1)]
                stmts = [f"VAR {chain[0]} = {val}."] + [f"VAR {chain[i]} = VAR {chain[i - 1]}." for i in range(1, h + 1)]
                dst = [f"VAR {dec[0]} = {dval}."] + [f"VAR {dec[i]} = VAR {dec[i - 1]}." for i in range(1, h + 1)]
                n_fill = max(1, L // 23)                                    # ~23 tokens per filler block
                slots = sorted(rng.sample(range(n_fill), len(stmts) + len(dst)))
                tags = [0] * len(stmts) + [1] * len(dst); rng.shuffle(tags)    # interleave, each chain in its own order
                cs, ds = iter(stmts), iter(dst); seq = [next(ds) if t else next(cs) for t in tags]
                body, si = [], 0
                for f in range(n_fill):
                    body.append(FILLER)
                    if si < len(slots) and slots[si] == f: body.append(seq[si] + " "); si += 1
                out.append({"id": f"ruler_vt-h{h}-L{L}-{k}", "family": "ruler_vt", "dial": {"hops": h, "tokens": L},
                            "kind": "set", "prompt": "Memorize and track the chain(s) of variable assignments hidden in the "
                            "following text.\n\n" + "".join(body) + f"\n\nQuestion: find all variables that are assigned the "
                            f"value {val} in the text above." + END + "ANSWER: <VAR>, <VAR>, ...",
                            "answer": sorted(chain), "absent": False})
    return out


# --- BABILong, CogniLoad, CoRe ---------------------------------------------------------------------------------------------
def build_babilong(rng, plan=(("qa2", ("0k", "2k", "8k")), ("qa3", ("0k", "2k", "8k")), ("qa19", ("0k",))), per=25):
    out = []
    for task, lens in plan:
        for L in lens:
            data = json.load(open(fetch(f"https://huggingface.co/datasets/RMT-team/babilong/resolve/main/data/{task}/{L}.json",
                                        f"babilong_{task}_{L}.json")))
            for k, d in enumerate(data[:per]):
                hint = (" Answer with the two directions as letters separated by a comma, e.g. n,w." if task == "qa19" else
                        " Answer with one word.")
                out.append({"id": f"babilong-{task}-{L}-{k}", "family": "babilong", "dial": {"task": task, "context": L},
                            "kind": "value", "prompt": d["input"] + "\n\nQuestion: " + d["question"].strip() + hint + END +
                            "ANSWER: <answer>", "answer": d["target"], "absent": False})
    return out


def build_cogniload(rng, lengths=(20, 50, 100), per=40):
    out = []
    for L in lengths:
        rows = [json.loads(l) for l in open(fetch(f"https://huggingface.co/datasets/cogniloadteam/cogniload/resolve/main/length_{L}.jsonl",
                                                  f"cogniload_{L}.jsonl"))]
        rng.shuffle(rows)
        for k, r in enumerate(rows[:per]):
            out.append({"id": f"cogniload-{L}-{r['puzzle_id']}", "family": "cogniload",
                        "dial": {"statements": L, "difficulty": int(r["difficulty"]), "needles": int(r["needles"])},
                        "kind": "value", "prompt": r["initialisations"] + "\n\n" + r["statements"] + "\n\nQuestion: " +
                        r["question"] + " Answer with one word." + END + "ANSWER: <answer>", "answer": r["solution"],
                        "absent": False})
    return out


def build_core(rng, per_type=50):
    rows = [json.loads(l) for l in open(fetch("https://huggingface.co/datasets/lt-asset/CoRe/resolve/main/data.jsonl", "core.jsonl"))]
    rows = [r for r in rows if r["category"] == "trace"]
    rng.shuffle(rows); out = []
    for typ in ("data", "control", "infoflow"):
        for gt in (True, False):
            sel = [r for r in rows if r["task_id"].startswith(typ) and str(r["groundtruth"]) == str(gt)][:per_type // 2]
            for r in sel:
                out.append({"id": f"core-{r['task_id']}", "family": "core", "dial": {"type": typ, "language": r["language"]},
                            "kind": "yesno", "prompt": r["prompt"] + "\n\n(After your analysis, also give the bare yes/no "
                            "verdict.)" + END + "ANSWER: YES  (or ANSWER: NO)", "answer": "YES" if gt else "NO",
                            "absent": not gt})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=ROOT / "llm_transfer/data/suite_v1.jsonl")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--small", action="store_true", help="a few per cell (tests)")
    a = ap.parse_args(); rng = random.Random(a.seed); s = a.small
    items = (build_bridge(3 if s else 25) + build_pointer(rng, per=4 if s else 30) + build_path(rng, per=4 if s else 30) +
             build_boxes(rng, per=4 if s else 30) + build_ruler_vt(rng, per=2 if s else 20) +
             build_babilong(rng, per=3 if s else 25) + build_cogniload(rng, per=3 if s else 40) +
             build_core(rng, per_type=6 if s else 50))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for it in items: f.write(json.dumps(it) + "\n")
    c = Counter(it["family"] for it in items); ab = Counter(it["family"] for it in items if it["absent"])
    print(f"{len(items)} items -> {a.out}"); print({k: f"{v} ({ab[k]} absent)" for k, v in c.items()})


if __name__ == "__main__":
    main()
