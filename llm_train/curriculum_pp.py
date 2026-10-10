#!/usr/bin/env python3
"""Per-puzzle curriculum for expert iteration (v3). Every puzzle the scheduler uses has its own measured success rate,
from K = 8 samples by the current best model; success = a verified answer the model finished on its own (not a
budget-forced cut, which on a no-win puzzle is a nearly free "NO WIN").

Why (v2, llm_train/curriculum.py, rounds 1-9): with 300 of 15,512 puzzles per round, a puzzle came up about once in 50
rounds, so per-puzzle parole never fired (paroled 0, released 0 in every round) and allocation ran on bucket averages,
which mix always-solved and never-solved puzzles; the no-win bucket read 0.92-0.96 (cheap stable cuts) and was starved
to 3 puzzles a round while finishing only 40-67% of near-miss negatives.

States: unscreened -> active (0 < successes < K) | mastered (K/K) | paroled (0/K).
  screen     round 1 samples SCREEN puzzles, stratified evenly over the buckets: their 8 samples are both the screen and
             round 1's training data (screening the whole pool, 8 x 15,512 samples, would take ~40 A100 hours).
  allocate   each later round: SPOT of N goes to paroled puzzles (spot checks); fresh unscreened puzzles refill the
             working set (up to FRESH_MAX of N while fewer than WORKING are active), spread over the buckets in
             proportion to how many of each bucket's screened puzzles turned out learnable; the rest are active puzzles
             drawn without replacement with weight priority(p) = (1 - (1-p)^K) (1 - p) from their latest rate, none
             sampled in the previous COOLDOWN rounds, at most MAX_SHARE of the round from one bucket.
  update     latest result decides: K/K -> mastered (never sampled again), 0/K -> paroled, else active with the new p.
             A spot-checked paroled puzzle with any finished success returns to active (K/K: mastered). Spot checks are
             the only way back: a bucket-rate uplift rule compared random screening puzzles with priority-selected
             ones (biased toward solvable), and in simulation released 110 puzzles at once on that bias alone.
  train set  sft_stop fine-tunes a fresh LoRA from the base model every round, so a puzzle that stops appearing in the
             data can be forgotten. Each puzzle's latest verified examples go into a bank; the round's training set is
             its new examples (mastered puzzles' excluded: no further training on them) plus replay from the bank (other
             puzzles, mastered ones included) up to REPLAY x the new examples, at most TRAIN_MAX in total.
  gate       paired, regression-only, on the validation set (llm_train/make_val.py, never trained on; the benchmark
             stays the test set). The objective is more correct answers the model finishes on its own, so: promote
             unless FINISHED accuracy (all but prevent) is clearly worse, z = (b - c) / sqrt(b + c) < -GATE_Z, with
             b / c the items only the new / only the best model gets right; plus one safety net: reject a
             catastrophic forced-accuracy drop within wins / no-win / placements (z < -FORCED_Z), e.g. a model that
             "finishes" by answering garbage. v3.1 (2026-10-06, after rounds 1-6): v3.0 also rejected z < -1.5 on
             forced accuracy and a rise in finished "NO WIN on a real win" answers; both penalised learning to stop
             (a self-stopped answer is committed with less search than one forced at the cap, and finishing more
             means more of every finished answer), and rejected rounds 1-4 (wins finished 17% -> 37-41%) for
             r5 (23%). v3.2: also not clearly worse (same z) than the PEAK, the promoted model with the highest
             finished accuracy so far, so small accepted losses cannot add up round after round (under v3.1 r6, 0.346,
             replaced r4, 0.367, at z -1.07; a chain of such steps could drift arbitrarily far down). Use `regate` to
             replay all rounds under the current rule.

  python -m llm_train.curriculum_pp screen   --state S --pool P --n 1600 --out R1/ids.json
  python -m llm_train.curriculum_pp allocate --state S --pool P --n 150 --round R --out R/ids.json
  python -m llm_train.curriculum_pp update   --state S --round R --outcomes R/outcomes.jsonl --sft R/sft.jsonl \
                                             --bank BANK.jsonl --train-out R/train.jsonl
v4 (curriculum_v4_colab.ipynb): the same scheduler and gate, but each round CONTINUES the best adapter
(sft_stop --init-adapter) instead of a fresh LoRA on the base model, so rounds build on each other; replay then only
guards against forgetting: --replay 0.5 --balance-replay; learning rate 1e-4. Forks from v3's round 1 screen.

  python -m llm_train.curriculum_pp gate     --state S --round R --new EVAL/vR --adapter R/adapter/final \
                                             --init-best EVAL/v0 --init-adapter INIT     (EVAL/name.jsonl: llm_bench rows)
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from llm_train.curriculum import bucket, load_pool

K = 8
FLOOR = 0.02
SPOT = 0.10
FRESH_MAX = 0.5
WORKING = 1000
COOLDOWN = 1
MAX_SHARE = 0.3
MIN_SAMPLES = 32          # a bucket's finished rate is logged only from at least this many regular samples
REPLAY = 1.0
TRAIN_MAX = 1000
MIN_NOWIN_THINK = 2048    # a NO WIN example must show at least this much search (natural finishes included)
GATE_Z = 1.5
FORCED_Z = 3.0


def priority(p: float, k: int = K) -> float:
    return max(FLOOR, (1 - (1 - p) ** k) * (1 - p))


def weighted_sample(items: list, weights: list, m: int, rng: random.Random) -> list:
    """m items without replacement, P(item) proportional to its weight (Efraimidis-Spirakis keys)."""
    keys = [(math.log(rng.random()) / w, it) for it, w in zip(items, weights) if w > 0]
    return [it for _, it in sorted(keys, key=lambda x: -x[0])[:m]]


def spread(n: int, weights: dict, room: dict) -> dict:
    """Split n over keys in proportion to weights (largest remainder), never more than room[k]; leftovers go on."""
    out = {k: 0 for k in weights}; left = n
    while left > 0:
        open_ = {k: w for k, w in weights.items() if out[k] < room[k] and w > 0}
        if not open_: break
        tot = sum(open_.values()); share = {k: left * w / tot for k, w in open_.items()}
        add = {k: min(room[k] - out[k], int(share[k])) for k in open_}
        if not any(add.values()):                             # hand out single units by largest remainder
            for k in sorted(open_, key=lambda k: -(share[k] - int(share[k]))):
                if left == 0: break
                if out[k] < room[k]: out[k] += 1; left -= 1
            continue
        for k, a in add.items(): out[k] += a; left -= a
    return out


class State:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else \
            {"puzzles": {}, "rates": {}, "applied": [], "best": None, "best_eval": None, "log": []}

    def save(self):
        tmp = self.path.with_suffix(".tmp"); tmp.write_text(json.dumps(self.d)); tmp.replace(self.path)

    def init_pool(self, pool, exclude=()):
        exclude = set(exclude)
        for r in pool:
            if r["id"] in exclude: continue                   # validation puzzles: never sampled
            self.d["puzzles"].setdefault(r["id"], {"bucket": bucket(r), "status": "unscreened", "attempts": []})

    def by_status(self) -> dict:
        out = defaultdict(lambda: defaultdict(list))
        for pid, s in self.d["puzzles"].items(): out[s["status"]][s["bucket"]].append(pid)
        return out

    def screen(self, n: int, seed: int) -> list[str]:
        """Round 1: n unscreened puzzles, as evenly over the buckets as their sizes allow."""
        rng = random.Random(seed); un = self.by_status()["unscreened"]
        alloc = spread(n, {b: 1.0 for b in un}, {b: len(v) for b, v in un.items()})
        ids = []
        for b, m in alloc.items():
            pool = sorted(un[b]); rng.shuffle(pool); ids += pool[:m]
        self.d["log"].append({"round": 1, "event": "screen", "alloc": alloc})
        return ids

    def allocate(self, n: int, rnd: int, seed: int) -> list[str]:
        rng = random.Random(seed); P = self.d["puzzles"]; S = self.by_status()
        paroled = sorted(p for v in S["paroled"].values() for p in v)
        n_spot = min(len(paroled), int(round(SPOT * n)))
        n_active = sum(len(v) for v in S["active"].values())
        eligible = [p for v in S["active"].values() for p in v
                    if not P[p]["attempts"] or P[p]["attempts"][-1][0] < rnd - COOLDOWN]
        n_un = sum(len(v) for v in S["unscreened"].values())
        n_fresh = min(n_un, max(0, WORKING - n_active), int(round(FRESH_MAX * n)))
        # regular: active puzzles by priority, capped per bucket
        cap = max(1, int(MAX_SHARE * n)); per_b = Counter(); regular = []
        for pid in weighted_sample(sorted(eligible), [priority(P[p].get("p")) for p in sorted(eligible)],
                                   len(eligible), rng):
            if len(regular) >= n - n_spot - n_fresh: break
            if per_b[P[pid]["bucket"]] < cap: regular.append(pid); per_b[P[pid]["bucket"]] += 1
        n_fresh = min(n_un, n - n_spot - len(regular))                    # caps left room: give it to fresh puzzles
        # fresh: buckets weighted by the learnable share of their screened puzzles (unknown -> 0.5)
        learn = {}
        for b in S["unscreened"]:
            scr = [p for p, s in P.items() if s["bucket"] == b and s["status"] != "unscreened"]
            act = sum(bool(P[p].get("ever_active")) for p in scr)      # measured 0 < successes < K at least once
            learn[b] = max(0.05, act / len(scr)) if scr else 0.5
        alloc = spread(n_fresh, learn, {b: len(v) for b, v in S["unscreened"].items()})
        fresh = []
        for b, m in alloc.items():
            pool = sorted(S["unscreened"][b]); rng.shuffle(pool); fresh += pool[:m]
        spot = rng.sample(paroled, n_spot)
        self.d["log"].append({"round": rnd, "event": "allocate", "regular": len(regular), "fresh": dict(alloc),
                              "spot_checks": n_spot, "regular_by_bucket": dict(per_b),
                              "fresh_weights": {b: round(v, 2) for b, v in learn.items()}})
        return regular + fresh + spot

    def update(self, rnd: int, outcomes: list[dict]) -> dict:
        """Applied once per round. Returns {id: status after this round} for the puzzles sampled."""
        P = self.d["puzzles"]
        if rnd in self.d["applied"]: return {o["id"]: P[o["id"]]["status"] for o in outcomes}
        per = defaultdict(lambda: [0, 0])
        for o in outcomes:                                    # bucket finished rates: regular samples only
            if P[o["id"]]["status"] != "paroled":
                per[P[o["id"]]["bucket"]][0] += o["finished_correct"]; per[P[o["id"]]["bucket"]][1] += o["samples"]
        rates = {b: v / n for b, (v, n) in per.items() if n >= MIN_SAMPLES}
        self.d["rates"][str(rnd)] = rates
        moves = Counter()
        for o in outcomes:
            s = P[o["id"]]; fc, n = o["finished_correct"], o["samples"]
            s["attempts"].append([rnd, fc, n, o["verified"]])
            before = s["status"]
            if before == "paroled":
                if fc > 0:                                    # spot check solved it: back in (or straight to mastered)
                    s.update(status="mastered" if fc == n else "active", p=fc / n, released_round=rnd)
                    if fc < n: s["ever_active"] = True
                    moves["released (spot check)"] += 1
                continue
            s["p"] = fc / n
            if fc == n: s["status"] = "mastered"
            elif fc == 0:
                s.update(status="paroled", paroled_round=rnd)
            else: s["status"] = "active"; s["ever_active"] = True
            moves[f"{before} -> {s['status']}"] += 1
        self.d["applied"].append(rnd)
        st = Counter(s["status"] for s in P.values())
        self.d["log"].append({"round": rnd, "event": "update", "finished_rates": {b: round(v, 3) for b, v in rates.items()},
                              "moves": dict(moves), "status": dict(st)})
        return {o["id"]: P[o["id"]]["status"] for o in outcomes}

    def rate(self, b: str):
        for rnd in sorted(self.d["rates"], key=int, reverse=True):
            if b in self.d["rates"][rnd]: return self.d["rates"][rnd][b]
        return None


def build_train(rnd: int, sft_rows: list[dict], status: dict, bank_path: Path, out: Path, seed: int,
                replay: float = REPLAY, train_max: int = TRAIN_MAX, balance: bool = False,
                min_nowin_think: int = MIN_NOWIN_THINK) -> dict:
    """Bank this round's examples (latest per puzzle), then write the training set: new examples of non-mastered
    puzzles + replay of other puzzles' banked examples (balance: spread evenly over the subtasks, as far as each has
    banked examples, instead of in proportion to the bank, so every skill gets refreshed every round)."""
    bank = {}
    if bank_path.exists():
        for l in open(bank_path):
            r = json.loads(l); bank.setdefault(r["id"], []).append(r)
    new_by_id = defaultdict(list)
    for r in sft_rows: new_by_id[r["id"]].append({**r, "round": rnd})
    for pid, rows in new_by_id.items(): bank[pid] = rows                # latest examples replace older ones
    tmp = bank_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for rows in bank.values():
            for r in rows: f.write(json.dumps(r) + "\n")
    tmp.replace(bank_path)
    rng = random.Random(seed)
    # v4 rounds 6-13: 40-55% of the NO WIN examples (natural finishes; the 2,048-token floor in expert_iter covered
    # only cut-off answers) had < 1,000 thinking tokens against a ~3,100 median for wins, so each round taught "no
    # early win -> say NO WIN": answers fell to 12-16 s and 400+ real wins were called NO WIN. The bank keeps them.
    short = lambda r: r.get("subtask") == "win_neg" and r.get("think_tokens", min_nowin_think) < min_nowin_think
    dropped_short = sum(short(r) for rows in new_by_id.values() for r in rows)
    new = [r for pid, rows in new_by_id.items() if status.get(pid) != "mastered" for r in rows if not short(r)]
    if len(new) > train_max: rng.shuffle(new); new = new[:train_max]
    others = [r for pid, rows in bank.items() if pid not in new_by_id or status.get(pid) == "mastered" for r in rows
              if not short(r)]
    rng.shuffle(others)
    n_rep = max(0, min(int(replay * len(new)), train_max - len(new)))
    if balance:
        by = defaultdict(list)
        for r in others: by[r["subtask"]].append(r)
        alloc = spread(n_rep, {k: 1.0 for k in by}, {k: len(v) for k, v in by.items()})
        rep = [r for k, m in alloc.items() for r in by[k][:m]]
    else:
        rep = others[:n_rep]
    rows = new + rep; rng.shuffle(rows)
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps({k: v for k, v in r.items() if k != "round"}) + "\n")
    info = {"new": len(new), "replay": len(rep), "bank_puzzles": len(bank), "dropped_short_nowin": dropped_short,
            "dropped_mastered": sum(len(v) for pid, v in new_by_id.items() if status.get(pid) == "mastered"),
            "by_subtask": dict(Counter(r["subtask"] for r in rows))}
    return info


def load_rows(prefix: str) -> dict:
    """id -> llm_bench result row (first sample) from <prefix>.jsonl (finished and, after --salvage, forced fields)."""
    out = {}
    for l in open(f"{prefix}.jsonl"):
        r = json.loads(l); out.setdefault(r["id"], r)
    return out


def family(r: dict) -> str | None:
    if r["task"] == "win": return "wins" if r["answers"]["win"] else "no-win"
    return "placements" if r["task"] in ("block", "forced") else None          # prevent: reported only


def paired_z(best: dict, new: dict, ids, ok) -> tuple[int, int, float]:
    b = sum(ok(new[i]) and not ok(best[i]) for i in ids); c = sum(ok(best[i]) and not ok(new[i]) for i in ids)
    return b, c, (b - c) / math.sqrt(b + c) if b + c else 0.0


def finished_acc(rows: dict) -> float:
    ids = [i for i in rows if family(rows[i])]
    return sum(rows[i]["correct"] for i in ids) / len(ids)


def gate(best: dict | None, new: dict, peak: dict | None = None) -> tuple[bool, str]:
    """best/new/peak: load_rows() of the same validation items. Promote unless finished accuracy is clearly worse than
    the best's or the peak's, or forced accuracy collapses (see the module docstring); the NO-WIN count is reported,
    not used."""
    if best is None: return True, "no previous model"
    ids = [i for i in new if i in best and family(new[i])]
    acc = lambda R: sum(R[i]["correct"] for i in ids) / len(ids)
    b, c, z = paired_z(best, new, ids, lambda r: r["correct"])
    pos = [i for i in ids if family(new[i]) == "wins"]
    nb, nc, nz = paired_z(best, new, pos, lambda r: r.get("forced_outcome", r["outcome"]) == "missed win (said NO WIN)")
    head = f"finished {acc(best):.3f} -> {acc(new):.3f} (+{b}/-{c}, z {z:+.2f}); forced NO WIN on real wins +{nb}/-{nc}"
    if z < -GATE_Z: return False, f"finished accuracy clearly worse: {head}"
    if peak is not None and peak is not best:
        pids = [i for i in ids if i in peak]
        pb, pc, pz = paired_z(peak, new, pids, lambda r: r["correct"])
        head += f"; vs peak {finished_acc(peak):.3f} (+{pb}/-{pc}, z {pz:+.2f})"
        if pz < -GATE_Z: return False, f"finished accuracy clearly worse than the peak: {head}"
    for fam in ("wins", "no-win", "placements"):
        fi = [i for i in ids if family(new[i]) == fam]
        fb, fc, fz = paired_z(best, new, fi, lambda r: r.get("forced_correct", r["correct"]))
        if fz < -FORCED_Z: return False, f"forced accuracy on {fam} collapsed (+{fb}/-{fc}, z {fz:+.2f}); {head}"
    return True, head

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a0 = sub.add_parser("screen"); a0.add_argument("--state", required=True); a0.add_argument("--pool", required=True)
    a0.add_argument("--exclude", default=None, help="JSON list of ids never to sample (llm_train.make_val)")
    a0.add_argument("--n", type=int, default=1600); a0.add_argument("--out", required=True)
    a1 = sub.add_parser("allocate"); a1.add_argument("--state", required=True); a1.add_argument("--pool", required=True)
    a1.add_argument("--exclude", default=None, help="JSON list of ids never to sample (llm_train.make_val)")
    a1.add_argument("--n", type=int, default=150); a1.add_argument("--round", type=int, required=True); a1.add_argument("--out", required=True)
    a2 = sub.add_parser("update"); a2.add_argument("--state", required=True); a2.add_argument("--round", type=int, required=True)
    a2.add_argument("--outcomes", required=True); a2.add_argument("--sft", required=True); a2.add_argument("--bank", required=True)
    a2.add_argument("--train-out", required=True); a2.add_argument("--replay", type=float, default=REPLAY)
    a2.add_argument("--train-max", type=int, default=TRAIN_MAX)
    a2.add_argument("--balance-replay", action="store_true", help="replay spread evenly over the subtasks")
    a2.add_argument("--min-nowin-think", type=int, default=MIN_NOWIN_THINK,
                    help="drop NO WIN examples (new and replayed) with fewer thinking tokens")
    a3 = sub.add_parser("gate"); a3.add_argument("--state", required=True); a3.add_argument("--round", type=int, required=True)
    a3.add_argument("--new", required=True); a3.add_argument("--adapter", required=True)
    a3.add_argument("--init-best", default=None); a3.add_argument("--init-adapter", default=None)
    a5 = sub.add_parser("regate", help="replay every round's gate under the current rule and set the best")
    a5.add_argument("--state", required=True); a5.add_argument("--eval-dir", required=True)
    a5.add_argument("--out-dir", required=True, help="the run folder (round adapters at <out-dir>/rR/adapter/final)")
    a5.add_argument("--init-best", required=True); a5.add_argument("--init-adapter", required=True)
    a4 = sub.add_parser("show"); a4.add_argument("--state", required=True)
    a = ap.parse_args(); st = State(a.state)
    if a.cmd == "screen":
        st.init_pool(load_pool(a.pool), json.loads(Path(a.exclude).read_text()) if a.exclude else ()); ids = st.screen(a.n, seed=1001)
        Path(a.out).write_text(json.dumps(ids)); st.save(); print(json.dumps(st.d["log"][-1], indent=1))
    elif a.cmd == "allocate":
        st.init_pool(load_pool(a.pool), json.loads(Path(a.exclude).read_text()) if a.exclude else ()); ids = st.allocate(a.n, a.round, seed=1000 + a.round)
        Path(a.out).write_text(json.dumps(ids)); st.save(); print(json.dumps(st.d["log"][-1], indent=1))
    elif a.cmd == "update":
        status = st.update(a.round, [json.loads(l) for l in open(a.outcomes)])
        info = build_train(a.round, [json.loads(l) for l in open(a.sft)], status, Path(a.bank), Path(a.train_out),
                           seed=a.round, replay=a.replay, train_max=a.train_max, balance=a.balance_replay,
                           min_nowin_think=a.min_nowin_think)
        st.d["log"].append({"round": a.round, "event": "train_set", **info}); st.save()
        print(json.dumps(st.d["log"][-2], indent=1)); print(json.dumps(st.d["log"][-1], indent=1))
    elif a.cmd == "gate":
        if st.d["best"] is None and a.init_best:
            st.d["best"], st.d["best_eval"] = a.init_adapter, a.init_best
        best_rows = load_rows(st.d["best_eval"]) if st.d["best_eval"] else None
        peak_eval = st.d.get("peak_eval") or st.d["best_eval"]
        peak_rows = (best_rows if peak_eval == st.d["best_eval"] else load_rows(peak_eval)) if peak_eval else None
        new_rows = load_rows(a.new); ok, why = gate(best_rows, new_rows, peak_rows)
        if ok:
            st.d["best"], st.d["best_eval"] = a.adapter, a.new
            if peak_rows is None or finished_acc(new_rows) > finished_acc(peak_rows): st.d["peak_eval"] = a.new
        st.d["log"].append({"round": a.round, "event": "gate", "promoted": ok, "reason": why, "best": st.d["best"],
                            "peak_eval": st.d.get("peak_eval")})
        st.save(); print(("PROMOTED: " if ok else "kept previous best: ") + why); print("best:", st.d["best"])
    elif a.cmd == "regate":
        best_eval, best = a.init_best, a.init_adapter; peak_eval = best_eval; R = 1; decisions = []
        while Path(f"{a.eval_dir}/v{R}.jsonl").exists():
            new_eval = f"{a.eval_dir}/v{R}"; best_rows, new_rows = load_rows(best_eval), load_rows(new_eval)
            peak_rows = best_rows if peak_eval == best_eval else load_rows(peak_eval)
            ok, why = gate(best_rows, new_rows, peak_rows)
            decisions.append({"round": R, "promoted": ok, "reason": why})
            print(f"round {R}: " + ("PROMOTED: " if ok else "kept previous best: ") + why)
            if ok:
                best_eval, best = new_eval, f"{a.out_dir}/r{R}/adapter/final"
                if finished_acc(new_rows) > finished_acc(peak_rows): peak_eval = new_eval
            R += 1
        st.d["best"], st.d["best_eval"], st.d["peak_eval"] = best, best_eval, peak_eval
        st.d["log"].append({"round": R - 1, "event": "regate", "rule": "v3.2", "decisions": decisions, "best": best,
                            "peak_eval": peak_eval})
        st.save(); print("best:", best)
    else:
        print(json.dumps({k: v for k, v in st.d.items() if k != "puzzles"}, indent=1)[:6000])
        print(dict(Counter(s["status"] for s in st.d["puzzles"].values())))


if __name__ == "__main__":
    main()
