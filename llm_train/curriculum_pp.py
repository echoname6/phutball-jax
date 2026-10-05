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
  gate       paired and regression-only, on the validation set (llm_train/make_val.py: 100 puzzles per bucket, never
             trained on; the benchmark stays the test set). Per-round gains are a few points, too small to confirm on
             ~1,000 items, so demanding a significant improvement stalls the run (v2 rejected its two lowest-truncation
             rounds on 25-item noise); sampling from a slightly worse model costs little with the replay bank. The new
             model is compared item by item with the best's stored results: b = items only the new model gets right,
             c = items only the best gets right, z = (b - c) / sqrt(b + c). Promote unless
               finished accuracy (all but prevent)                    z < -GATE_Z   (clearly worse)
               forced accuracy within wins / no-win / placements      z < -GATE_Z   (knowledge traded for stopping)
               "NO WIN" said on a real win (finished)                 z > +GATE_Z   (v2 round 8's failure)

  python -m llm_train.curriculum_pp screen   --state S --pool P --n 1600 --out R1/ids.json
  python -m llm_train.curriculum_pp allocate --state S --pool P --n 150 --round R --out R/ids.json
  python -m llm_train.curriculum_pp update   --state S --round R --outcomes R/outcomes.jsonl --sft R/sft.jsonl \
                                             --bank BANK.jsonl --train-out R/train.jsonl
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
GATE_Z = 1.5


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
                replay: float = REPLAY, train_max: int = TRAIN_MAX) -> dict:
    """Bank this round's examples (latest per puzzle), then write the training set: new examples of non-mastered
    puzzles + replay of other puzzles' banked examples."""
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
    new = [r for pid, rows in new_by_id.items() if status.get(pid) != "mastered" for r in rows]
    if len(new) > train_max: rng.shuffle(new); new = new[:train_max]
    others = [r for pid, rows in bank.items() if pid not in new_by_id or status.get(pid) == "mastered" for r in rows]
    rng.shuffle(others)
    rep = others[: max(0, min(int(replay * len(new)), train_max - len(new)))]
    rows = new + rep; rng.shuffle(rows)
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps({k: v for k, v in r.items() if k != "round"}) + "\n")
    info = {"new": len(new), "replay": len(rep), "bank_puzzles": len(bank),
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


def gate(best: dict | None, new: dict) -> tuple[bool, str]:
    """best/new: load_rows() of the same validation items. Promote unless clearly worse (see the module docstring)."""
    if best is None: return True, "no previous model"
    ids = [i for i in new if i in best and family(new[i])]
    acc = lambda R: sum(R[i]["correct"] for i in ids) / len(ids)
    b, c, z = paired_z(best, new, ids, lambda r: r["correct"])
    head = f"finished {acc(best):.3f} -> {acc(new):.3f} (+{b}/-{c}, z {z:+.2f})"
    if z < -GATE_Z: return False, f"finished accuracy clearly worse: {head}"
    for fam in ("wins", "no-win", "placements"):
        fi = [i for i in ids if family(new[i]) == fam]
        fb, fc, fz = paired_z(best, new, fi, lambda r: r.get("forced_correct", r["correct"]))
        if fz < -GATE_Z: return False, f"forced accuracy on {fam} clearly worse (+{fb}/-{fc}, z {fz:+.2f}); {head}"
    pos = [i for i in ids if family(new[i]) == "wins"]
    nb, nc, nz = paired_z(best, new, pos, lambda r: r["outcome"] == "missed win (said NO WIN)")
    if nz > GATE_Z: return False, f"says NO WIN on more real wins (+{nb}/-{nc}, z {nz:+.2f}); {head}"
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
    a3 = sub.add_parser("gate"); a3.add_argument("--state", required=True); a3.add_argument("--round", type=int, required=True)
    a3.add_argument("--new", required=True); a3.add_argument("--adapter", required=True)
    a3.add_argument("--init-best", default=None); a3.add_argument("--init-adapter", default=None)
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
                           seed=a.round, replay=a.replay, train_max=a.train_max)
        st.d["log"].append({"round": a.round, "event": "train_set", **info}); st.save()
        print(json.dumps(st.d["log"][-2], indent=1)); print(json.dumps(st.d["log"][-1], indent=1))
    elif a.cmd == "gate":
        if st.d["best"] is None and a.init_best:
            st.d["best"], st.d["best_eval"] = a.init_adapter, a.init_best
        ok, why = gate(load_rows(st.d["best_eval"]) if st.d["best_eval"] else None, load_rows(a.new))
        if ok: st.d["best"], st.d["best_eval"] = a.adapter, a.new
        st.d["log"].append({"round": a.round, "event": "gate", "promoted": ok, "reason": why, "best": st.d["best"]})
        st.save(); print(("PROMOTED: " if ok else "kept previous best: ") + why); print("best:", st.d["best"])
    else:
        print(json.dumps({k: v for k, v in st.d.items() if k != "puzzles"}, indent=1)[:6000])
        print(dict(Counter(s["status"] for s in st.d["puzzles"].values())))


if __name__ == "__main__":
    main()
