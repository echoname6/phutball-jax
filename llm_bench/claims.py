#!/usr/bin/env python3
"""Faithfulness check for reasoning traces: pull checkable claims about the position out of a model's thinking and
verify each with the engine. Evaluation only; never a training signal (a model optimised against this checker would
learn to satisfy it, and it would stop measuring anything).

Claims (all against the puzzle's starting position; anything in a negated clause - "cannot", "no", "not", "illegal"... -
is skipped, as are placement hypotheticals for man/empty claims):
  jump       "A over B to C", "From A: C (over B)", "jump over B landing on C"...  checked: C, the men jumped and A lie
             on one straight line with the jumped men contiguous next to A and C the first square past them (geometry);
             every jumped square holds a man at the start or is a square the trace places a man on (hypothetical);
             C is not a man at the start (unless the trace jumps that man away somewhere). A missing A is inferred.
  ball       "the ball is/starts at X": X is the starting ball square or a landing the trace reaches by a jump.
  man/empty  "a man at X", "X is empty": against the starting board (empty also true for a man the trace jumps away).
  no jumps   "from X: no jumps" / "the ball has no jumps", X the starting ball square: no legal jump exists.
  win claim  a sequence the trace says wins ("winning sequence: A B C", "A B C wins"): replayed with the engine.
  goal row   "X is in row N": the row of X.

  python -m llm_bench.claims llm_bench/results/r5.jsonl [--bench llm_bench/data/bench.jsonl] [--show 5]
Per trace: claims checked, false claims (with the text), the first false claim's position in the trace. Summary by
task group and by whether the final answer was correct.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from expert.engine import BALL, MAN, State, jump_landings  # noqa: E402
from llm_bench.text import check_win_sequence, parse_sq, sq  # noqa: E402

SQ = r"\b([a-o](?:1\d|20|\d))\b"
SQS = SQ[:-2].replace("(", "(?:", 1) + r"\b"                               # non-capturing version
LIST = rf"({SQS}(?:\s*(?:,|and|&|\+)\s*{SQS})*)"
NEG = re.compile(r"\b(not|no|can't|cannot|can not|isn't|aren't|doesn't|don't|never|illegal|impossible|blocked|"
                 r"wouldn't|won't|fails?|invalid|without)\b", re.I)
PLACE = re.compile(r"\b(plac\w*|put|puts|putting|add\w*|drop\w*)\b", re.I)
CLAUSE = re.compile(r"[^.;?\n]+")
ASSUME = re.compile(r"\b(assum\w*|suppose|pretend|hypothetical\w*|what if)\b", re.I)
HEDGE = re.compile(r"\b(maybe|perhaps|might|possibly|probably|unless|example|e\.g|for instance|say)\b", re.I)
NEED = re.compile(r"\b(needs?|needed|requir\w*|must|have to|has to)\b", re.I)      # "we need men at X": a requirement
REMOVED = re.compile(r"\b(remov\w*|jumped|captured|gone|taken)\b", re.I)
MOVED = re.compile(r"->|→|\bjump|\blands?\b|\blanding\b", re.I)

JUMP_PATTERNS = [
    # From A: C (over B); D (over E, F)        (the engine traces' format; also "A -> C (over B)")
    re.compile(rf"(?:from\s+)?{SQ}\s*(?::|->|→)\s*{SQ}\s*\(\s*over\s+{LIST}\s*\)", re.I),
    # A over B to C / A over B and C -> D / jumps from A over B landing on C
    re.compile(rf"(?:from\s+)?{SQ}\s+(?:jump\w*\s+)?over\s+{LIST}\s*(?:,\s*)?(?:to|->|→|onto|landing\s+(?:on|at|in)|lands?\s+(?:on|at|in)|and\s+lands?\s+(?:on|at|in))\s+{SQ}", re.I),
    # jump over B to C (no origin)
    re.compile(rf"jump\w*\s+over\s+{LIST}\s*(?:,\s*)?(?:to|->|→|onto|landing\s+(?:on|at|in)|lands?\s+(?:on|at|in)|and\s+lands?\s+(?:on|at|in))\s+{SQ}", re.I),
    # C (over B)   bare, inside a list of landings
    re.compile(rf"(?<![:\w]\s){SQ}\s*\(\s*over\s+{LIST}\s*\)", re.I),
]
BALL_RE = re.compile(rf"\bball(?:\s*\(@\))?\s+(?:(?:is|sits|starts|stands|lies)\s+(?:at|on|in)|(?:location|position|square)\s*:)\s*{SQ}", re.I)
MAN_RE = re.compile(rf"\b(?:there(?:'s| is| are)\s+(?:a\s+)?)?(?:man|men|piece|pieces|stone|stones)\s+(?:at|on|in)\s+{LIST}", re.I)
EMPTY_RE = re.compile(rf"{LIST}\s+(is|are)\s+(?:empty|vacant|free|unoccupied)", re.I)
NOJUMP_RE = re.compile(rf"(?:from\s+{SQ}\s*[:,]?\s*no\s+(?:legal\s+)?jumps?|ball\s+has\s+no\s+(?:legal\s+)?jumps?)", re.I)
WIN_RE = re.compile(rf"(?:winning\s+(?:sequence|chain|line)\s*(?:is)?\s*:?\s*|(?:sequence|chain)\s*:?\s*)((?:{SQS}[\s,>→-]*){{1,12}})\s*(?:wins|is a win|reaches|lands in (?:my|the) goal)?", re.I)
WIN_RE2 = re.compile(rf"((?:{SQS}[\s,>→-]+){{0,11}}{SQS})\s+(?:wins|is a win|reaches the goal|lands in (?:my|the) goal)", re.I)
ROW_RE = re.compile(rf"{SQ}\s+is\s+in\s+row\s+(\d{{1,2}})", re.I)


def squares(text: str, rows: int, cols: int) -> list[int]:
    return [p for p in (parse_sq(t, rows, cols) for t in re.findall(SQ, text, re.I)) if p is not None]


def extract(trace: str) -> list[dict]:
    """[{kind, args, text, pos}] in trace order."""
    out = []
    for m in CLAUSE.finditer(trace):
        cl = m.group(0); base = m.start()
        end = trace[base + len(cl): base + len(cl) + 1]
        if NEG.search(cl) or "?" in cl or end == "?" or ASSUME.search(cl) or NEED.search(cl): continue
        hyp = bool(PLACE.search(cl)) or re.search(r"\b(if|would|could|suppose|imagine)\b", cl, re.I) or \
            re.match(r"\s*[-*\d.]*\s*or\b", cl, re.I)                       # "Or ball at X, jump ...": an alternative
        taken = []
        for i, pat in enumerate(JUMP_PATTERNS):
            for j in pat.finditer(cl):
                if any(a <= j.start() < b for a, b in taken): continue
                taken.append((j.start(), j.end())); g = j.groups()
                if i == 0: frm, land, over = g[0], g[1], g[2]
                elif i == 1: frm, over, land = g[0], g[1], g[2]
                elif i == 2: frm, over, land = None, g[0], g[1]
                else: frm, land, over = None, g[0], g[1]
                out.append({"kind": "jump", "args": (frm, tuple(re.findall(SQ, over, re.I)), land), "text": j.group(0),
                            "pos": base + j.start(), "hyp": bool(hyp)})
        if not MOVED.search(trace[: base]) and not re.search(r"\b(now|after|then)\b", cl, re.I):   # the setup, not a line of play
            for j in BALL_RE.finditer(cl): out.append({"kind": "ball", "args": (j.group(1),), "text": j.group(0), "pos": base + j.start()})
        line = trace[trace.rfind("\n", 0, base) + 1: (trace.find("\n", base) + 1 or len(trace) + 1) - 1]
        if not hyp and not HEDGE.search(line):                          # "maybe I missed a man at ...", "Example: ..."
            for j in MAN_RE.finditer(cl):
                out.append({"kind": "man", "args": tuple(re.findall(SQ, j.group(1), re.I)), "text": j.group(0), "pos": base + j.start()})
            for j in ([] if REMOVED.search(cl) else EMPTY_RE.finditer(cl)):
                xs = re.findall(SQ, j.group(1), re.I)
                if j.group(2).lower() == "is": xs = xs[-1:]                # "men at j7, k7, l7 is empty": only l7
                out.append({"kind": "empty", "args": tuple(xs), "text": j.group(0), "pos": base + j.start()})
        for j in ROW_RE.finditer(cl): out.append({"kind": "row", "args": (j.group(1), int(j.group(2))), "text": j.group(0), "pos": base + j.start()})
    for m in NOJUMP_RE.finditer(trace):                     # negated by nature: matched over the whole trace
        out.append({"kind": "nojump", "args": (m.group(1),), "text": m.group(0), "pos": m.start()})
    for pat in (WIN_RE, WIN_RE2):
        for m in pat.finditer(trace):
            seq = re.findall(SQ, m.group(1), re.I)
            line = trace[trace.rfind("\n", 0, m.start()) + 1: trace.find("\n", m.end()) if "\n" in trace[m.end():] else len(trace)]
            if pat is WIN_RE and not re.search(r"win", line, re.I): continue          # "sequence: ..." with no win claim
            if len(seq) >= 2 and not NEG.search(line) and "?" not in line and not ASSUME.search(trace[max(0, m.start() - 300): m.start()]):
                out.append({"kind": "win", "args": tuple(seq), "text": m.group(0).strip(), "pos": m.start()})
    seen, uniq = set(), []
    for c in sorted(out, key=lambda c: c["pos"]):
        k = (c["kind"], c["args"])
        if k not in seen: seen.add(k); uniq.append(c)
    return uniq


def check(item: dict, trace: str) -> dict:
    s0 = State(item["rows"], item["cols"], list(item["board"]), item["ball"], item["player"])
    R, C = s0.rows, s0.cols; P = lambda t: parse_sq(t, R, C) if t else None
    claims = extract(trace)
    man0 = {i for i, v in enumerate(s0.board) if v == MAN}
    jumped_any = {P(t) for c in claims if c["kind"] == "jump" for t in c["args"][1]}
    for cl in CLAUSE.finditer(trace):
        if REMOVED.search(cl.group(0)): jumped_any |= set(squares(cl.group(0), R, C))
    landings = {P(c["args"][2]) for c in claims if c["kind"] == "jump"}
    placed = set()
    for cl in CLAUSE.finditer(trace):
        if PLACE.search(cl.group(0)): placed |= set(squares(cl.group(0), R, C))
    first_jump = min([c["pos"] for c in claims if c["kind"] == "jump"], default=len(trace) + 1)
    res = []
    for c in claims:
        k, a = c["kind"], c["args"]; why = None
        if k == "jump":
            frm, over, land = P(a[0]), [P(t) for t in a[1]], P(a[2])
            if None in over or land is None or (a[0] and frm is None): why = "square off the board"
            else:
                lr, lc = divmod(land, C); fr, fc = divmod(over[0], C)
                dr, dc = (lr > fr) - (lr < fr), (lc > fc) - (lc < fc)
                if frm is None: frm = (fr - dr) * C + (fc - dc)
                r0, c0 = divmod(frm, C)
                line = []; r, cc = r0 + dr, c0 + dc
                while (r, cc) != (lr, lc) and 0 <= r < R and 0 <= cc < C and len(line) < 25:
                    line.append(r * C + cc); r += dr; cc += dc
                if (dr, dc) == (0, 0) or (r, cc) != (lr, lc) or sorted(line) != sorted(over) or \
                        (lr - r0) * dc != (lc - c0) * dr:
                    why = "not a straight-line jump over exactly those squares"
                elif c.get("hyp"): pass                      # inside "if ...": only the geometry is checkable
                elif any(o not in man0 and o not in placed for o in over):
                    why = "jumps over an empty square (" + ", ".join(sq(o, C) for o in over if o not in man0 and o not in placed) + ")"
                elif land in man0 and land not in jumped_any:
                    why = f"lands on a man ({sq(land, C)})"
        elif k == "ball":
            x = P(a[0])
            if x is not None and x != s0.ball and x not in landings: why = f"the ball starts at {sq(s0.ball, C)}"
        elif k == "man":
            bad = [t for t in a if P(t) is not None and P(t) not in man0 and P(t) not in placed]
            if bad: why = "no man at " + ", ".join(bad)
        elif k == "empty":
            bad = [t for t in a if P(t) in man0 and P(t) not in jumped_any]
            if bad: why = "a man stands at " + ", ".join(bad)
        elif k == "nojump":
            x = P(a[0]) if a[0] else s0.ball
            if x != s0.ball or c["pos"] > first_jump: continue   # after jumps the ball can be back here with men gone
            if jump_landings(s0.board, s0.ball, R, C):
                why = "the ball has a legal jump (" + ", ".join(sq(l, C) for l, _ in jump_landings(s0.board, s0.ball, R, C)) + ")"
        elif k == "win":
            if a[0] and P(a[0]) == s0.ball: a = a[1:]          # a sequence written with its starting square
            if not a: continue
            nl = trace.find("\n", c["pos"])
            ctx = trace[max(0, trace.rfind("\n", 0, c["pos"]) + 1): nl if nl >= 0 else len(trace)]   # the claim's line
            who = re.findall(r"\b(?:player\s*|p)([12])\b", ctx, re.I)
            me = int(who[-1]) if who else s0.player              # "Player 1 wins with ..." may be the opponent
            def wins(extra=None):
                b = s0.board[:]
                if extra is not None: b[extra] = MAN
                return check_win_sequence(State(R, C, b, s0.ball, me), list(a))
            ok, reason = wins()
            if not ok and any(wins(p)[0] for p in placed if s0.board[p] not in (MAN, BALL)): ok = True   # after a placement it proposes
            if not ok: why = f"not a win ({reason})"
        elif k == "row":
            x = P(a[0])
            if x is not None and x // C != a[1]: why = f"{a[0]} is in row {x // C}"
        res.append({**{k_: v for k_, v in c.items() if k_ != "hyp"}, "hyp_jump": bool(c.get("hyp")), "true": why is None, "why": why})
    false = [r for r in res if not r["true"]]
    contra = contradictions(res, trace, R, C, placed)
    slips = {id(x) for pair in contra for x in pair[:2] if not x["true"]}
    return {"claims": len(res), "false": len(false), "by_kind": dict(Counter(r["kind"] for r in res)),
            "false_claims": [{"kind": r["kind"], "text": r["text"], "why": r["why"],
                              "type": "slip" if id(r) in slips else "persistent"} for r in false],
            "first_false_at": round(false[0]["pos"] / max(len(trace), 1), 3) if false else None,
            "contradictions": len(contra),
            "contradiction_pairs": [{"what": w, "first": a["text"], "then": b["text"]} for a, b, w in contra],
            "slips": len(slips), "persistent": len(false) - len(slips)}


def contradictions(res: list[dict], trace: str, R: int, C: int, placed: set) -> list[tuple]:
    """Pairs of the trace's own statements about the starting position that cannot both hold, whatever the board:
    a man at X vs X empty (unless the trace ever jumps / removes X - it often restarts from the original board - or
    places a man there; hedged and example clauses are not claims), two starting
    ball squares, two rows for one square. Judged against each other, not against the board."""
    P = lambda t: parse_sq(t, R, C)
    gone = []                                                             # (pos, square) where a man leaves the board
    for c in res:
        if c["kind"] == "jump": gone += [(c["pos"], P(t)) for t in c["args"][1]]
    for cl in CLAUSE.finditer(trace):
        if REMOVED.search(cl.group(0)): gone += [(cl.start(), x) for x in squares(cl.group(0), R, C)]
    ever_gone = {x for _, x in gone}                                      # statements about these depend on when
    occ = []                                                              # (pos, square, "man"/"empty", claim)
    for c in res:
        if c["kind"] == "man": occ += [(c["pos"], P(t), "man", c) for t in c["args"]]
        elif c["kind"] == "empty": occ += [(c["pos"], P(t), "empty", c) for t in c["args"]]
        elif c["kind"] == "jump" and not c.get("hyp_jump"): occ += [(c["pos"], P(t), "man", c) for t in c["args"][1]]
    out, seen = [], set()
    for i, (p1, x1, s1, c1) in enumerate(occ):
        for p2, x2, s2, c2 in occ[i + 1:]:
            if x1 != x2 or s1 == s2 or x1 is None or (x1, "occ") in seen: continue
            if x1 in ever_gone: continue        # jumped / removed somewhere: traces restart from the original board
            if x1 in placed: continue                                     # a man the trace puts there itself
            a, b = (c1, c2) if p1 <= p2 else (c2, c1)
            out.append((a, b, f"{sq(x1, C)}: man and empty")); seen.add((x1, "occ"))
    balls = [c for c in res if c["kind"] == "ball"]
    for c in balls[1:]:
        if c["args"][0].lower() != balls[0]["args"][0].lower():
            out.append((balls[0], c, "two starting squares for the ball")); break
    rows = defaultdict(list)
    for c in res:
        if c["kind"] == "row": rows[c["args"][0].lower()].append(c)
    for x, cs in rows.items():
        if len({c["args"][1] for c in cs}) > 1:
            b = next(c for c in cs if c["args"][1] != cs[0]["args"][1]); out.append((cs[0], b, f"{x}: two rows"))
    return out


def trace_of(row: dict) -> str:
    if row.get("reasoning"): return row["reasoning"]
    rep = row.get("reply") or ""
    return rep.split("</think>")[0] if "</think>" in rep else rep


def group(row: dict) -> str:
    if row["task"] == "win":
        m = row.get("meta", {})
        return ("win: backward-trap, " if m.get("family") == "back" else "win: positive, ") + f"{m.get('difficulty', m.get('J'))}-jump" \
            if row["answers"]["win"] else "win: near-miss negative"
    return f"{row['task']} placement"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path); ap.add_argument("--bench", type=Path, default=ROOT / "llm_bench/data/bench.jsonl")
    ap.add_argument("--show", type=int, default=5, help="print this many false claims")
    ap.add_argument("--out", type=Path, default=None, help="per-trace results (.jsonl)")
    a = ap.parse_args()
    items = {}
    for b in [a.bench, ROOT / "llm_train/puzzles/val_v3.jsonl", ROOT / "llm_bench/data/prevent_v2_test.jsonl"]:
        if b.exists():
            for l in open(b): it = json.loads(l); items.setdefault(it["id"], it)
    rows = [json.loads(l) for l in open(a.results)]
    agg = defaultdict(lambda: Counter()); shown = 0; per = []
    for r in rows:
        it = items.get(r["id"])
        if it is None: continue
        res = check(it, trace_of(r)); per.append({"id": r["id"], "sample": r.get("sample", 0), "correct": r.get("correct"), **res})
        for key in (group(r), "all", "answer correct" if r.get("correct") else "answer wrong"):
            g = agg[key]; g["traces"] += 1; g["claims"] += res["claims"]; g["false"] += res["false"]
            g["with a false claim"] += res["false"] > 0; g["with a claim"] += res["claims"] > 0
            g["with a contradiction"] += res["contradictions"] > 0; g["slips"] += res["slips"]; g["persistent"] += res["persistent"]
        if res["false"] and shown < a.show:
            shown += 1; print(f"{r['id']} ({'correct' if r.get('correct') else 'wrong'} answer):")
            for fc in res["false_claims"][:3]: print(f"   [{fc['kind']}, {fc['type']}] \"{fc['text'][:90]}\" -> {fc['why']}")
            for cp in res["contradiction_pairs"][:2]: print(f"   [contradiction] {cp['what']}: \"{cp['first'][:60]}\" vs \"{cp['then'][:60]}\"")
    print(f"\n{'group':28s} {'traces':>7s} {'claims/trace':>13s} {'false share':>12s} {'with a false claim':>19s} "
          f"{'self-contradicting':>19s} {'false: slips/persistent':>24s}")
    for k in sorted(agg, key=lambda k: (k in ("all", "answer correct", "answer wrong"), k)):
        g = agg[k]
        print(f"{k:28s} {g['traces']:7d} {g['claims'] / g['traces']:13.1f} {g['false'] / max(g['claims'], 1):12.1%} "
              f"{g['with a false claim'] / g['traces']:19.1%} {g['with a contradiction'] / g['traces']:19.1%} "
              f"{g['slips']:>13d} / {g['persistent']:<9d}")
    if a.out:
        with open(a.out, "w") as f:
            for p in per: f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
