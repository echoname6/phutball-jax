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
import math
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
            nowin = it["task"] == "win" and bool(re.search(r"ANSWER:\s*NO\s*WIN", ans, re.I))
            rows.append({"nowin": nowin, "i": i, "key": json.dumps(prompts[i]) if not isinstance(prompts[i], str) else prompts[i],
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
                # no length bonus for a correct NO WIN: proving there is no win needs the full search, and rewarding
                # short NO WINs taught "stop early and say NO WIN" (stage-1 probe: wins 6/24 vs no-wins 29/40)
                bonus = self.eff * max(-1.0, min(1.0, (m - r["len"]) / m)) if (m and r["correct"] and not r["nowin"]) else 0.0
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
        per = defaultdict(lambda: [0, 0])
        for r in rows: per[r["subtask"]][0] += r["correct"]; per[r["subtask"]][1] += 1
        self.stats.append({"per": {k: tuple(v) for k, v in per.items()}, "correct": sum(r["correct"] for r in rows) / n, "trunc": sum(r["trunc"] for r in rows) / n,
                           "forced": sum(r.get("forced_correct", False) for r in rows) / n,
                           "len": sum(r["len"] for r in rows) / n})


# --- cost-sensitive reward (v2) --------------------------------------------------------------------------------------
FORCE_TAIL_NF = ("\n\nTime is up. Commit to your best answer now; if you have not found one, answer NOT FOUND.\n"
                 "</think>\n\nANSWER:")
NOT_FOUND = re.compile(r"ANSWER:\s*NOT\s*FOUND", re.I)


class PhutballCostReward(PhutballReward):
    """Decision-theoretic reward (expert iteration could only reward correct answers, never penalise the two
    mistakes, so the training mix decided which one the model drifted into: v4 r7-r13 said NO WIN on real wins,
    r15-r16 claimed wins that do not work).

      finished, correct                       +1 - lam * tokens      (lam ramps up over --lam-ramp reward calls)
      finished, false win claim / wrong move  -a                     (a jump chain that does not win, a bad placement)
      finished, NO WIN on a real win          -b
      finished, no parsable answer            -c                     (incl. NOT FOUND it chose itself: not offered)
      cut off at the cap -> forced answer ("Time is up ... if you have not found one, answer NOT FOUND"):
            correct  +force_ok   NOT FOUND  0   false win -a   false NO WIN -b   unparsable -c,  all minus cut_cost
    cut_cost (gated GRPO phases 1-2): without it, running to the cap was the safe bet on hard puzzles - the forced answer
    comes after the full search, so it is wrong less often than an earlier commitment (commit EV ~ 2p - 1, cut-off
    EV ~ -(1 - q)), and validation wins cut off at the cap rose 33% -> 57-62% (finished 0.459 -> 0.34-0.36) while forced
    accuracy rose 65% -> 75-81%. With cut_cost t a cut-off is worth at most -t, so committing (EV 2p - 1 for a = 1) beats
    it once p > (1 - t) / 2 even against a sure forced answer, and once p > (q - t) / 2 against the forced accuracy q
    the model actually has (t = 0.35, q ~ 0.75: p > 0.2).
    With calibrated beliefs a win is claimed when P(win) > (1 + a) / (2 + a + b); lam * cap << min(a, b), so thinking
    longer is never worse than a wrong early answer, and lam only orders correct answers. NOT FOUND (0) beats any
    wrong forced guess, and running to the cap and saying NOT FOUND beats a self-chosen early abstention (-c): when
    unsure, keep searching."""

    def __init__(self, cap, log_path=None, a=1.0, b=1.5, c=0.25, lam=3e-5, lam_ramp=50, force_ok=0.5, budgets=(1.0,),
                 cut_cost=0.0, prior_correct=False):
        super().__init__(cap, log_path=log_path, eff=0.0, trunc=0.0, force_credit=1.0, budgets=budgets)
        self.a, self.b, self.c, self.lam, self.lam_ramp, self.force_ok = a, b, c, lam, lam_ramp, force_ok
        self.cut_cost = cut_cost; self.prior_correct = prior_correct

    def force(self, prompts, texts):
        raws = [self.template(p) + t + FORCE_TAIL_NF for p, t in zip(prompts, texts)]
        return ["ANSWER:" + c.split("\n")[0] for c in self.forcer(raws)]

    tag = "cost reward"

    def row_reward(self, r, lam):
        if not r["trunc"]:
            return {"correct": 1.0 - lam * r["len"], "false_claim": -self.a, "false_nowin": -self.b}.get(r["cat"], -self.c)
        return {"correct": self.force_ok, "not_found": 0.0, "false_claim": -self.a,
                "false_nowin": -self.b}.get(r["forced_cat"], -self.c) - self.cut_cost

    def judge(self, it, ans):
        """(category, correct) for an answer string."""
        if NOT_FOUND.search(ans or ""): return "not_found", False
        sc = score(it, ans) if ans else {"correct": False, "outcome": "unparsed"}
        o = sc["outcome"]
        if sc["correct"]: return "correct", True
        if o == "missed win (said NO WIN)": return "false_nowin", False
        if o in ("unparsed", "wrong answer type", "unverifiable (search cap)") or o.startswith("no answer"): return "unparsed", False
        return "false_claim", False                     # a chain that does not win / a placement that does not work

    def __call__(self, prompts, completions, item, subtask=None, completion_ids=None, **kw):
        self.calls += 1
        texts = [text_of(c) for c in completions]
        lens = [len(x) for x in completion_ids] if completion_ids is not None else [len(t) // 3 for t in texts]
        lam = self.lam * min(1.0, self.calls / max(1, self.lam_ramp))
        rows, cut = [], []
        for i, (t, it, L) in enumerate(zip(texts, item, lens)):
            it = json.loads(it) if isinstance(it, str) else it
            think, ans = split(t)
            trunc = L >= self.cap - 1 and not re.search(r"ANSWER:", ans)
            cat, ok = ("cut", False) if trunc else self.judge(it, ans)
            rows.append({"i": i, "id": it.get("id"), "task": it["task"], "subtask": subtask[i] if subtask else it["task"],
                         "len": L, "trunc": trunc, "correct": ok, "cat": cat, **legibility(think)})
            if trunc: cut.append(i)
        forced = {}
        if cut and self.forcer:
            try:
                fa = self.force([prompts[i] for i in cut], [split(texts[i])[0] for i in cut])
                for i, ans in zip(cut, fa):
                    it = json.loads(item[i]) if isinstance(item[i], str) else item[i]
                    forced[i] = (ans, *self.judge(it, ans))
            except Exception as e:                                   # noqa: BLE001  (never kill a training step)
                print("budget forcing failed:", repr(e)[:300], flush=True)
        for r in rows:
            if r["trunc"]:
                ans, cat, ok = forced.get(r["i"], ("", "unparsed", False))
                r.update(forced=ans, forced_cat=cat, forced_correct=ok)
            r["reward"] = self.row_reward(r, lam)
        self.summary(rows)
        cats = defaultdict(int)
        for r in rows: cats[r.get("forced_cat") if r["trunc"] else r["cat"]] += 1
        print(f"  [{self.tag}] lam {lam:.1e} | " + " ".join(f"{k} {v}" for k, v in sorted(cats.items())), flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                for r, t in zip(rows, texts):
                    f.write(json.dumps({"call": self.calls, "time": time.time(), **r, "text": t}) + "\n")
        w = kw.get("w_prior") if self.prior_correct else None   # importance weight: the advantage scales by it
        if w is not None: return [r["reward"] * float(x) for r, x in zip(rows, w)]
        return [r["reward"] for r in rows]


# --- curve reward (run 2) ----------------------------------------------------------------------------------------------
class PhutballCurveReward(PhutballCostReward):
    """The curve gate's objective as a reward: the expected cost-reward under an UNKNOWN deadline B drawn log-uniformly
    (budgets bmin, 2 bmin, ..., bmax, as the gate's 2k / 4k / 8k / 16k), computed from one rollout instead of sampled.
    A rollout that finishes at t tokens with answer utility u (+1 correct, -a false claim, -b false NO WIN) has its
    answer by every deadline >= t:

      finished:  u * w(t),   w(t) = min(1, log2(2 bmax / t) / log2(2 bmax / bmin))   (1 at bmin, 0.5 at bmax / 2,
                                                                                     0.25 at bmax: the gate's grid)
                 no parsable answer (incl. a self-chosen NOT FOUND): -c, flat (quitting is never cheap)
      cut off at bmax (the generation cap): forced answer wrong -> u * w(bmax); correct or NOT FOUND -> 0; minus cut_cost

    Against the 6k cost reward: a hard puzzle solved at 12k earns credit (GRPO gets signal on puzzles every sample of
    which was cut off at 6k), the same answer earlier earns more by the same factor at every scale (no per-token lam),
    and a wrong early answer costs the most (it is wrong at every deadline). cut_cost (default 0.1): with 0, running to
    the cap (0) beats answering at the cap (0.25 (2p - 1)) whenever P(right) < 0.5; with 0.1, answering pays from p ~ 0.3.
    Tightening later: lower bmin / bmax (the deadlines), not a per-bucket schedule."""
    tag = "curve reward"

    def __init__(self, cap, log_path=None, a=1.0, b=1.5, c=0.25, bmin=2048, cut_cost=0.1, prior_correct=False, **_):
        super().__init__(cap, log_path=log_path, a=a, b=b, c=c, lam=0.0, lam_ramp=1, force_ok=0.0, cut_cost=cut_cost,
                         prior_correct=prior_correct)
        self.bmin, self.bmax = bmin, cap

    def weight(self, tokens: int) -> float:
        return max(0.0, min(1.0, math.log2(2 * self.bmax / max(tokens, 1)) / math.log2(2 * self.bmax / self.bmin)))

    def row_reward(self, r, lam):
        u = {"correct": 1.0, "false_claim": -self.a, "false_nowin": -self.b}
        if not r["trunc"]:
            return u[r["cat"]] * self.weight(r["len"]) if r["cat"] in u else -self.c
        fc = r.get("forced_cat")
        return (u[fc] * self.weight(self.bmax) if fc in ("false_claim", "false_nowin") else 0.0) - self.cut_cost
