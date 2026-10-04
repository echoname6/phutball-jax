"""Reward for GRPO on the phutball puzzles: the engine verifies the final answer; the thinking is never scored.

  reward = 1[answer verified by the engine]                     (llm_bench.run.score, the benchmark's own scorer)
         + EFF * (mean_len_correct - len) / mean_len_correct    only for correct samples, within the prompt's group
                                                                (shorter than its correct siblings: up to +EFF)
  truncated (hit the token cap): -TRUNC + FORCE_CREDIT * 1[budget-forced answer verified]
         budget forcing: the cut-off thinking + "Time is up ... </think> ANSWER:" is completed by the same vLLM engine
         (12 tokens, greedy), so a cut-off search that was heading to the right answer still beats one that was not.
         Order: finished correct (1) > cut off, forced correct (0.25) > finished wrong (0) > cut off, forced wrong (-0.5).
  --force-budgets 0.25,0.5,1 ("anytime" credit): the cut-off thinking is forced at several prefixes (fractions of the
         cap) and the credit is FORCE_CREDIT * (share of those prefixes whose forced answer verifies), so reaching the
         right answer earlier (and staying there) earns more even while no sample in the group finishes; this keeps
         a within-group pressure toward efficient search when the truncation penalty is the same for every sample.
Length pressure is relative to each position's difficulty (a hard position whose correct samples are all long is
not pushed to be short). No language, format or
readability term: the thinking may drift into any token code that keeps the answers verified.

Every scored sample is appended to a JSONL log (full text) for the legibility metrics (legibility() below and
llm_train/analyze_rollouts.py).
"""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import defaultdict

from llm_bench.run import score

EFF = 0.1
TRUNC = 0.5
FORCE_CREDIT = 0.75
FORCE_TAIL = "\n\nTime is up. I must commit to my best answer now.\n</think>\n\nANSWER:"
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

    def __init__(self, cap: int, log_path: str | None = None, eff: float = EFF, trunc: float = TRUNC,
                 force_credit: float = FORCE_CREDIT, budgets: tuple = (1.0,)):
        self.cap, self.log_path, self.eff, self.trunc, self.force_credit = cap, log_path, eff, trunc, force_credit
        self.budgets = tuple(sorted(budgets))
        self.decode = None              # callable(token ids) -> text; needed for budgets < 1
        self.calls = 0; self.history = []; self.stats = []
        self.forcer = None              # callable(list[str] raw prompts) -> list[str] continuations; set by the trainer script
        self.template = None            # callable(prompt messages) -> chat-templated prompt text (generation prompt included)

    def force(self, prompts, texts) -> list[str]:
        raws = [self.template(p) + t + FORCE_TAIL for p, t in zip(prompts, texts)]
        return ["ANSWER:" + c.split("\n")[0] for c in self.forcer(raws)]

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
        cut = [r["i"] for r in rows if r["trunc"]]
        if cut and self.forcer and self.force_credit:
            try:
                jobs = []                                               # (sample index, budget fraction, thinking prefix)
                for i in cut:
                    for b in self.budgets:
                        if b >= 1.0 or self.decode is None or completion_ids is None:
                            jobs.append((i, 1.0, split(texts[i])[0]))
                        else:
                            jobs.append((i, b, split(self.decode(list(completion_ids[i])[: int(b * self.cap)]))[0]))
                forced = self.force([prompts[i] for i, _, _ in jobs], [t for _, _, t in jobs])
                hits = defaultdict(list)
                for (i, b, _), ans in zip(jobs, forced):
                    ok = bool(score(json.loads(item[i]) if isinstance(item[i], str) else item[i], ans)["correct"])
                    hits[i].append(ok)
                    if b >= 1.0: rows[i]["forced"] = ans; rows[i]["forced_correct"] = ok
                for i, h in hits.items():
                    rows[i]["forced_score"] = sum(h) / len(h); rows[i]["forced_hits"] = h
            except Exception as e:                                   # noqa: BLE001  (never kill a training step)
                print("budget forcing failed:", repr(e)[:300], flush=True)
        groups = defaultdict(list)
        for r in rows: groups[r["key"]].append(r)
        for g in groups.values():
            ok = [r for r in g if r["correct"]]
            m = sum(r["len"] for r in ok) / len(ok) if len(ok) >= 2 else None
            for r in g:
                bonus = self.eff * max(-1.0, min(1.0, (m - r["len"]) / m)) if (m and r["correct"]) else 0.0
                r["reward"] = (float(r["correct"]) + bonus - self.trunc * r["trunc"]
                               + self.force_credit * r.get("forced_score", float(r.get("forced_correct", False))))
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
                f"trunc {sum(r['trunc'] for r in rows) / n:.0%} forced-ok {sum(r.get('forced_correct', False) for r in rows)} "
                f"reward std {statistics.pstdev([r['reward'] for r in rows]):.2f} mean len {sum(r['len'] for r in rows) / n:.0f} "
                f"wordlike {sum(r['wordlike'] for r in rows) / n:.2f} | " +
                " ".join(f"{k} {sum(r['correct'] for r in v)}/{len(v)}" for k, v in sorted(by.items())))
        print(line, flush=True); self.history.append(line)
        self.stats.append({"correct": sum(r["correct"] for r in rows) / n, "trunc": sum(r["trunc"] for r in rows) / n,
                           "forced": sum(r.get("forced_correct", False) for r in rows) / n,
                           "len": sum(r["len"] for r in rows) / n})
