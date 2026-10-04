"""Reward for GRPO on the phutball puzzles: the engine verifies the final answer; the thinking is never scored.

  reward = 1[answer verified by the engine]                     (llm_bench.run.score, the benchmark's own scorer)
         + EFF * (mean_len_correct - len) / mean_len_correct    only for correct samples, within the prompt's group
                                                                (shorter than its correct siblings: up to +EFF)
         - TRUNC                                                if the completion hit the token cap
A truncated sample therefore scores below a wrong answer, and length pressure is relative to each position's
difficulty (a hard position whose correct samples are all long is not pushed to be short). No language, format or
readability term: the thinking may drift into any token code that keeps the answers verified.

Every scored sample is appended to a JSONL log (full text) for the legibility metrics (legibility() below and
llm_train/analyze_rollouts.py).
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict

from llm_bench.run import score

EFF = 0.1
TRUNC = 0.5
SQUARE = re.compile(r"^[a-o](1?\d|20)$")
WORD = re.compile(r"^[A-Za-z][a-z]*'?[a-z]*$")


def text_of(c) -> str:
    if isinstance(c, list): return "".join(m.get("content") or "" for m in c)
    return c


def split(text: str):
    """(thinking, answer part). The answer counts only after </think>."""
    if "</think>" in text:
        th, ans = text.rsplit("</think>", 1)
        return th.replace("<think>", ""), ans
    return text.replace("<think>", ""), ""


def legibility(think: str) -> dict:
    """Cheap drift metrics: share of non-ASCII characters; share of whitespace tokens that are plain English-like words
    or square names (stripped of punctuation)."""
    toks = think.split()
    if not toks: return {"nonascii": 0.0, "wordlike": 0.0}
    clean = [t.strip(".,:;()!?\"'-*`") for t in toks]
    wordlike = sum(1 for t in clean if WORD.match(t) or SQUARE.match(t.lower())) / len(toks)
    return {"nonascii": sum(ord(ch) > 127 for ch in think) / max(len(think), 1), "wordlike": wordlike}


class PhutballReward:
    """TRL GRPO reward function. Dataset columns used: item (JSON string), task, subtask."""
    __name__ = "phutball"

    def __init__(self, cap: int, log_path: str | None = None, eff: float = EFF, trunc: float = TRUNC):
        self.cap, self.log_path, self.eff, self.trunc = cap, log_path, eff, trunc
        self.calls = 0; self.history = []

    def __call__(self, prompts, completions, item, subtask=None, completion_ids=None, **kw):
        self.calls += 1
        texts = [text_of(c) for c in completions]
        lens = [len(x) for x in completion_ids] if completion_ids is not None else [len(t) // 3 for t in texts]
        rows = []
        for i, (t, it, L) in enumerate(zip(texts, item, lens)):
            it = json.loads(it) if isinstance(it, str) else it
            think, ans = split(t)
            trunc = L >= self.cap - 1 and not re.search(r"ANSWER:", ans)
            sc = score(it, ans) if ans else {"correct": False, "outcome": "truncated" if trunc else "no answer after </think>"}
            rows.append({"i": i, "key": json.dumps(prompts[i]) if not isinstance(prompts[i], str) else prompts[i],
                         "id": it.get("id"), "task": it["task"], "subtask": subtask[i] if subtask else it["task"],
                         "len": L, "trunc": trunc, "correct": bool(sc["correct"]), "outcome": sc["outcome"],
                         **legibility(think)})
        groups = defaultdict(list)
        for r in rows: groups[r["key"]].append(r)
        for g in groups.values():
            ok = [r for r in g if r["correct"]]
            m = sum(r["len"] for r in ok) / len(ok) if len(ok) >= 2 else None
            for r in g:
                bonus = self.eff * max(-1.0, min(1.0, (m - r["len"]) / m)) if (m and r["correct"]) else 0.0
                r["reward"] = float(r["correct"]) + bonus - self.trunc * r["trunc"]
        self.summary(rows)
        if self.log_path:
            with open(self.log_path, "a") as f:
                for r, t in zip(rows, texts):
                    f.write(json.dumps({"call": self.calls, "time": time.time(), **{k: v for k, v in r.items() if k != "key"},
                                        "text": t}) + "\n")
        return [r["reward"] for r in rows]

    def summary(self, rows):
        by = defaultdict(list)
        for r in rows: by[r["subtask"]].append(r)
        n = len(rows)
        line = (f"[reward call {self.calls}] n={n} correct {sum(r['correct'] for r in rows) / n:.0%} "
                f"trunc {sum(r['trunc'] for r in rows) / n:.0%} mean len {sum(r['len'] for r in rows) / n:.0f} "
                f"wordlike {sum(r['wordlike'] for r in rows) / n:.2f} | " +
                " ".join(f"{k} {sum(r['correct'] for r in v)}/{len(v)}" for k, v in sorted(by.items())))
        print(line, flush=True); self.history.append(line)
