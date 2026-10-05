#!/usr/bin/env python3
"""Mine claim templates from reasoning traces: every clause that mentions a square becomes a template (squares ->
<SQ>, square lists -> "<SQ>, ...", numbers -> <N>, leading fillers like "so"/"wait" dropped), ranked by how many traces
contain it, per source file. The frequent, stable templates are labelled by meaning with an engine check
(llm_bench/claim_map_v1.json) and the checker matches clauses against that frozen, versioned set.

  python -m llm_bench.claim_templates mine ~/Projects/phutball/eval/*.jsonl --top 150 --out templates.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SQ_RE = re.compile(r"\b[a-o](?:1\d|20|\d)\b", re.I)
FILLER = re.compile(r"^(?:(?:so|wait|then|and|now|also|but|okay|ok|yes|thus|hence|therefore|hmm|well|again|here|"
                    r"note|actually|indeed|first|next|finally|check|verify|let's see|recall)\b[\s,:]*)+", re.I)
SPLIT = re.compile(r"[.;?\n]")


def normalize(clause: str) -> str:
    t = clause.strip()
    t = re.sub(r"^[\s\-*#>|`\d.)]+", "", t)                               # list markers, numbering
    t = re.sub(r"[*_`]+", "", t)                                           # markdown emphasis
    t = FILLER.sub("", t.strip()).lower()
    t = SQ_RE.sub("<SQ>", t)
    t = re.sub(r"\(\s*\d+\s*,\s*\d+\s*\)", "(<N>,<N>)", t)
    t = re.sub(r"\b\d+\b", "<N>", t)
    t = re.sub(r"<SQ>(?:\s*(?:,|and|&)\s*<SQ>)+", "<SQ>, ...", t)          # square lists collapse
    t = re.sub(r"\s+", " ", t).strip(" ,:")
    return t


def clauses(trace: str):
    """(start offset, clause text) for clauses that mention a square."""
    pos = 0
    for part in SPLIT.split(trace):
        if SQ_RE.search(part) and len(part) < 200: yield pos, part
        pos += len(part) + 1


def mine(paths: list[Path], top: int) -> list[dict]:
    from llm_bench.claims import trace_of
    per = defaultdict(lambda: defaultdict(set)); n_traces = {}; examples = {}
    for p in paths:
        rows = [json.loads(l) for l in open(p)]; n_traces[p.stem] = len(rows)
        for k, r in enumerate(rows):
            for _, cl in clauses(trace_of(r)):
                t = normalize(cl)
                if "<SQ>" not in t: continue
                per[t][p.stem].add(k); examples.setdefault(t, cl.strip()[:120])
    ranked = []
    for t, by in per.items():
        share = {s: len(by.get(s, ())) / n_traces[s] for s in n_traces}
        ranked.append({"template": t, "traces": sum(len(v) for v in by.values()),
                       "sources_at_1pct": sum(v >= 0.01 for v in share.values()),
                       "share_by_source": {s: round(v, 3) for s, v in share.items()}, "example": examples[t]})
    ranked.sort(key=lambda d: -d["traces"])
    return ranked[:top]


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine"); m.add_argument("files", nargs="+", type=Path); m.add_argument("--top", type=int, default=150)
    m.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    ranked = mine(a.files, a.top)
    for d in ranked: print(f"{d['traces']:6d} {d['sources_at_1pct']:3d}  {d['template'][:80]:80s} | {d['example'][:60]}")
    if a.out: a.out.write_text(json.dumps(ranked, indent=1))


if __name__ == "__main__":
    main()
