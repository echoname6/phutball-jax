#!/usr/bin/env python3
"""Compare two GRPO runs' rollouts call by call (rollouts.jsonl from grpo_pilot): how many texts are identical.

Two runs with the same seed and starting weights produce identical first calls; after the first optimizer step the
policy changes and the sampled texts should diverge. Identical texts on later calls mean generation never saw the
updated weights (the LoRA was not synced into the colocated vLLM engine).

  python -m llm_train.compare_rollouts RUN_A/rollouts.jsonl RUN_B/rollouts.jsonl
"""
import json
import sys
from collections import defaultdict


def by_call(path):
    d = defaultdict(list)
    for line in open(path):
        r = json.loads(line); d[r["call"]].append(r.get("text", ""))
    return d


def main():
    a, b = by_call(sys.argv[1]), by_call(sys.argv[2])
    for c in sorted(set(a) & set(b)):
        same = sum(x == y for x, y in zip(a[c], b[c]))
        print(f"call {c:2d}: {same:2d}/{len(b[c])} rollouts identical")


if __name__ == "__main__":
    main()
