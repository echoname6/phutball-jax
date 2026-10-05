#!/usr/bin/env python3
"""Versioned claim map: the frequent clause templates of the reasoning traces (llm_bench/claim_templates.py), each
labelled BY MEANING (never by how often it comes out true) with an engine check, or as a known non-claim. Frozen per
version: every round of a comparison is scored with the same version; when coverage drops (the model's phrasing
drifts), a new version is mined from the larger pool and ALL rounds are re-scored with it.

v1: mined 2026-10-05 from the v2 curriculum's evals (untrained, warm start, g0-g10, r1, r5): the top 160 templates by
traces containing them, every one present in >= 1% of traces in all 15 sources; 120 labelled below.

Checks (args come from the clause: its squares in order, list slots expanded, and its numbers):
  ball      the starting ball square (only in the setup: before any move is discussed; never "now/after/then")
  menlist   each square holds a man at the start (setup only, like ball)
  man       each square holds a man (a square the trace places a man on is fine)
  empty     each square is empty (squares the trace ever jumps/removes are skipped: traces restart from the original
            board); "last": only the last square of the clause
  row       the square is in row N;  rowlist: every square is in row N (the clause's first number)
  coord     the square's (x, y) numbers fit SOME convention (column/row order, 0- or 1-based columns, rows as labelled or shifted by one): traces switch
            conventions mid-way, so only numbers that fit none (a misread square) count as false
  colrow    "(col N, row M)": named, so only the base (0/1) is inferred
  adjacent  the two squares are king-adjacent
  dirlist   the listed squares step one square at a time in the named direction
  path      the listed squares step one square at a time along one straight line
  goal      the square is in the named player's goal rows (no player named on the line: in either goal)
  none      a known non-claim (questions, intentions, state after moves, prompt quotes, ...)
"""
from __future__ import annotations

VERSION = 1
SOURCES = "v2 evals: untrained, warmstart, g0-g10, r1, r5 (2026-10-05)"

DIRS = {"up": (-1, 0), "north": (-1, 0), "down": (1, 0), "south": (1, 0), "left": (0, -1), "west": (0, -1),
        "right": (0, 1), "east": (0, 1), "nw": (-1, -1), "ne": (-1, 1), "sw": (1, -1), "se": (1, 1),
        "up-left": (-1, -1), "up-right": (-1, 1), "down-left": (1, -1), "down-right": (1, 1),
        "north-west": (-1, -1), "north-east": (-1, 1), "south-west": (1, -1), "south-east": (1, 1)}

_ball = ["ball at <SQ>", "ball is at <SQ>", "the ball is at <SQ>", "ball: <SQ>", "ball position: <SQ>",
         "ball (@) location: <SQ>", "current ball: <SQ>", "start: <SQ>", "start: ball at <SQ>", "ball location: <SQ>",
         "ball (@) is at <SQ>", "current ball position: <SQ>", "the ball is currently at <SQ>",
         "currently, the ball is at <SQ>", "<SQ> is ball", "<SQ> (ball)"]
_menlist = ["men: <SQ>, ...", "men list: <SQ>, ...", "\"men (<N>): <SQ>, ...", "men (o) locations: <SQ>, ...",
            "men positions: <SQ>, ...", "men: <SQ>"]
_man = ["<SQ> is a man", "<SQ> (man)", "men at <SQ>, ...", "men are at <SQ>, ...", "<SQ> is man", "men at <SQ>",
        "no, <SQ> is a man", "<SQ>, ... are men", "<SQ> is o", "man at <SQ>", "the men are at <SQ>, ..."]
_empty = ["<SQ> is empty", "<SQ> (empty)", "<SQ> empty", "no, <SQ> is empty", "<SQ>: empty", "<SQ>, ... are empty",
          "<SQ> is not a man", "square <SQ> is empty", "is <SQ> (empty)"]
_empty_last = ["<SQ> -> <SQ> (empty)"] + [f"{d}: <SQ> (empty)" for d in ("sw", "se", "nw", "ne", "down", "up", "left", "right")]
_row = ["<SQ> is row <N>", "<SQ> is in row <N>", "<SQ> is at row <N>", "<SQ> (row <N>)"]
_rowlist = ["row <N>: <SQ>, ...", "row <N>: <SQ>"]
_coord = ["<SQ> (<N>,<N>)", "<SQ> is (<N>,<N>)", "<SQ> is at (<N>,<N>)", "<SQ>: (<N>,<N>)", "(<N>,<N>) is <SQ>",
          "from <SQ> (<N>,<N>)", "ball at <SQ> (<N>,<N>)", "ball: <SQ> (<N>,<N>)", "ball is at <SQ> (<N>,<N>)",
          "<SQ> -> (<N>,<N>)", "(<N>,<N>) <SQ>"]
_coord_empty = ["<SQ> (<N>,<N>) - empty"]
_dirlist = ["up (north): <SQ>, ...", "down (south): <SQ>, ...", "up: <SQ>, ...", "down: <SQ>, ...", "left: <SQ>, ...",
            "right: <SQ>, ...", "nw: <SQ>, ...", "ne: <SQ>, ...", "sw: <SQ>, ...", "se: <SQ>, ...", "up-left: <SQ>, ..."]
_path = ["path: <SQ> -> <SQ> -> <SQ>", "path: <SQ> -> <SQ> -> <SQ> -> <SQ>", "path: <SQ> -> <SQ> -> <SQ> -> <SQ> -> <SQ>",
         "path: <SQ>, ...", "squares: <SQ>, ..."]
_none = ["<SQ>", "is <SQ> empty", "<SQ>, ...", "is <SQ> a man", "men removed: <SQ>", "men removed: <SQ>, ...",
         "is there a man at <SQ>", "from <SQ>", "\"legal first jumps from the ball's square: <SQ> (over <SQ>)",
         "landing square: <SQ>", "remaining men: <SQ>, ...", "new ball position: <SQ>", "possible jumps from <SQ>",
         "jump <N>: <SQ> -> <SQ>", "lands on <SQ>", "men remaining: <SQ>, ...", "<SQ> is not in the list",
         "jump over <SQ>", "is <SQ> a goal", "<SQ> is removed", "is <SQ> adjacent to <SQ>",
         "\"legal first jumps from the ball's square: <SQ> (over <SQ>, ...)", "jump <SQ> -> <SQ>", "<SQ> -> <SQ> -> <SQ>",
         "jump to <SQ>", "neighbors of <SQ>", "landing: <SQ>", "<SQ> -> <SQ>", "jump over <SQ>, ...", "new ball pos: <SQ>",
         "jump lands on <SQ>", "(<SQ>)", "the prompt says \"legal first jumps from the ball's square: <SQ> (over <SQ>)",
         "directions from <SQ>", "legal first jumps from <SQ>", "<SQ> -> <SQ> -> <SQ> -> <SQ>", "ball lands on <SQ>",
         "is <SQ> on the board", "can we jump over <SQ>",
         "the prompt says \"legal first jumps from the ball's square: <SQ> (over <SQ>, ...)", "<SQ> removed",
         "the ball is at <SQ>)\"", "<SQ> (removed)", "men jumped: <SQ>", "men jumped: <SQ>, ...", "the ball lands on <SQ>",
         "is there any other jump from <SQ>", "square is <SQ>", "the sequence is <SQ>, ...",
         "from <SQ>, can we jump to row <N> or <N>", "the jump lands on <SQ>", "<SQ>\"", "sequence: <SQ>, ...",
         "(<SQ>, ...)", "let's visualize the relevant area around the ball (<SQ>)", "let's check neighbors of <SQ>",
         "result: ball at <SQ>", "<SQ> is on the board", "removed: <SQ>, ...", "men jumped over: <SQ>",
         "let's check the neighbors of <SQ>", "are <SQ>, ... contiguous", "square: <SQ>", "<SQ> is occupied",
         # jump claims: checked by the jump extractor in llm_bench.claims, not by the map
         "jump over <SQ> lands on <SQ>", "<SQ> (over <SQ>)", "jump over <SQ> to <SQ>", "<SQ> -> <SQ> (over <SQ>)",
         "<SQ> (over <SQ>, ...)", "jump over <SQ>, ... lands on <SQ>", "jump <N>: <SQ> -> <SQ> (over <SQ>)",
         "jump <SQ> -> <SQ> is valid", "no men adjacent to <SQ>"]

MAP: dict[str, dict] = {}
for _ts, _spec in ((_ball, {"check": "ball"}), (_menlist, {"check": "menlist"}), (_man, {"check": "man"}),
                   (_empty, {"check": "empty"}), (_empty_last, {"check": "empty", "last": True}),
                   (_row, {"check": "row"}), (_rowlist, {"check": "rowlist"}), (_coord, {"check": "coord"}),
                   (_coord_empty, {"check": "coord", "also_empty": True}), (_dirlist, {"check": "dirlist"}),
                   (_path, {"check": "path"}), (_none, {"check": "none"}),
                   (["<SQ> (col <N>, row <N>)"], {"check": "colrow"}), (["<SQ> is adjacent to <SQ>"], {"check": "adjacent"}),
                   (["landing on <SQ> is a win"], {"check": "goal"})):
    for _t in _ts:
        assert _t not in MAP, _t
        MAP[_t] = _spec
