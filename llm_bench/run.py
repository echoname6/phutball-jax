#!/usr/bin/env python3
"""Run the phutball benchmark (llm_bench/build.py) against a language model and score it with the engine.

Any OpenAI-compatible endpoint: a vLLM server (e.g. on Colab: `vllm serve Qwen/Qwen3.5-9B`), a local server, or an
API provider. Chat mode for instruct/thinking models; --completions for base models (raw prompt, no chat template).

  # sanity check of the scoring, no model:
  python -m llm_bench.run --baseline nowin
  python -m llm_bench.run --baseline random
  # a model:
  OPENAI_API_KEY=... python -m llm_bench.run --base-url https://host/v1 --model Qwen/Qwen3.5-9B --concurrency 16
  python -m llm_bench.run --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-9B-Base --completions

Scoring: win positives: correct if the reply's jump sequence wins when replayed (any winning sequence counts);
win negatives: correct if NO WIN; forced / block: correct if the placement is in the (complete) answer set; prevent: a
placement in the complete set, or a jump sequence after which the engine finds no attacker win or unstoppable placement.
Results: llm_bench/results/<name>.jsonl (every reply) and a summary printed and saved as <name>.summary.json.
Every summary line reports the share of replies TRUNCATED at --max-tokens: a truncated reply has no answer and scores
as wrong, so read accuracy together with it. --salvage budget-forces an answer from truncated replies ("time is up" +
"ANSWER:"), giving "best answer within the token budget" (<name>-forced.summary.json).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from expert.engine import BALL, MAN, State, jump_landings  # noqa: E402
from llm_bench.text import check_win_sequence, parse_answer, replay_jumps, sq  # noqa: E402


def call(a, prompt_text: str, raw: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(a.api_key_env)
    if key: headers["Authorization"] = f"Bearer {key}"
    if a.completions:
        body = {"model": a.model, "prompt": raw if raw is not None else prompt_text + "\n\nReply:\n", "max_tokens": a.max_tokens, "temperature": a.temperature}
        url = a.base_url.rstrip("/") + "/completions"
    else:
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt_text}], "max_tokens": a.max_tokens,
                "temperature": a.temperature}
        if a.thinking is not None:                           # Qwen-style switch (vLLM / SGLang chat_template_kwargs)
            body["chat_template_kwargs"] = {"enable_thinking": a.thinking}
        url = a.base_url.rstrip("/") + "/chat/completions"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                out = json.loads(r.read())
            ch = out["choices"][0]
            text = ch.get("text") if a.completions else (ch["message"].get("content") or "")
            reasoning = None if a.completions else ch["message"].get("reasoning_content")
            return {"reply": text, "reasoning": reasoning, "usage": out.get("usage", {}), "finish_reason": ch.get("finish_reason")}
        except Exception as e:                               # noqa: BLE001  (retry transient errors)
            err = repr(e); time.sleep(2 ** attempt)
    return {"reply": "", "error": err}


def baseline_reply(kind: str, it: dict, rng: random.Random) -> str:
    s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"])
    if it["task"] == "win":
        if kind in ("nowin", "adjacent"): return "ANSWER: NO WIN"
        js = jump_landings(s.board, s.ball, s.rows, s.cols)
        return "ANSWER: " + (f"JUMP {sq(rng.choice(js)[0], s.cols)}" if js and rng.random() < 0.5 else "NO WIN")
    if kind == "adjacent":                                   # one fixed rule: the square next to the ball, toward my goal
        r, c = divmod(s.ball, s.cols); r += -1 if s.player == 1 else 1
        return f"ANSWER: PLACE {sq(r * s.cols + c, s.cols)}"
    empty = [i for i in range(s.cols, (s.rows - 1) * s.cols) if s.board[i] not in (BALL, MAN)]
    return f"ANSWER: PLACE {sq(rng.choice(empty), s.cols)}"


def score(it: dict, reply: str) -> dict:
    s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"])
    kind, val = parse_answer(reply)
    if kind is None: return {"correct": False, "outcome": "unparsed"}
    if it["task"] == "win":
        if it["answers"]["win"]:
            if kind == "nowin": return {"correct": False, "outcome": "missed win (said NO WIN)"}
            if kind != "jump" or not val: return {"correct": False, "outcome": "wrong answer type"}
            ok, why = check_win_sequence(s, val)
            return {"correct": ok, "outcome": "found a win" if ok else why}
        if kind == "nowin": return {"correct": True, "outcome": "correct NO WIN"}
        if kind == "jump":
            ok, why = check_win_sequence(s, val)
            return {"correct": False, "outcome": "claimed a win (BUG: engine says none)" if ok else f"claimed a win: {why}"}
        return {"correct": False, "outcome": "wrong answer type"}
    if it["task"] == "prevent" and kind == "jump" and val:
        from expert.puzzle_place import CapHit, attacker_threatens
        board, ball, w, err = replay_jumps(s, val)
        if err: return {"correct": False, "outcome": err}
        if w: return {"correct": w == s.player, "outcome": "jumps into a goal: " + ("wins" if w == s.player else "loses")}
        try:
            saved = not attacker_threatens(State(s.rows, s.cols, board, ball, 3 - s.player))
        except CapHit:
            return {"correct": False, "outcome": "unverifiable (search cap)"}
        return {"correct": saved, "outcome": "saving jump sequence" if saved else "jump sequence does not stop the threat"}
    if kind != "place": return {"correct": False, "outcome": "wrong answer type"}
    ok = val in it["answers"]["placements"]
    good = {"forced": "unstoppable", "block": "blocks", "prevent": "saving placement"}[it["task"]]
    return {"correct": ok, "outcome": good if ok else "not " + good}


FORCE = {True: "\n\nTime is up. I must commit to my best answer now.\n</think>\n\n", False: "\n\nTime is up. My best answer now:\n"}


def salvage(a, items_by_id: dict):
    """Budget forcing for replies that hit the token cap: re-send prompt + the truncated reply + "time is up" +
    "ANSWER:" as a raw completion and let the model finish the answer line (30 tokens). Scores the replies already
    paid for, as "best answer within the token budget". Reads and rewrites llm_bench/results/<name>.jsonl."""
    if not a.completions:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
    path = ROOT / "llm_bench/results" / f"{a.salvage}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    todo = [r for r in rows if r["outcome"] == "unparsed" and r.get("reply")]
    think = bool(a.thinking)

    def one(r):
        if a.completions: head = items_by_id[r["id"]]["prompt"] + "\n\nReply:\n"
        else: head = tok.apply_chat_template([{"role": "user", "content": items_by_id[r["id"]]["prompt"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=think)
        body = (r.get("reasoning") or "") + r["reply"]
        tail = FORCE[think and "</think>" not in body] + "ANSWER:"
        b = argparse.Namespace(**{**vars(a), "completions": True, "max_tokens": 30})
        out = call(b, None, raw=head + body + tail)
        return r, "ANSWER:" + out["reply"].split("\n")[0]

    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for r, forced in ex.map(one, todo):
            sc = score(items_by_id[r["id"]], forced)
            r["forced_answer"] = forced; r["forced_correct"] = sc["correct"]; r["forced_outcome"] = sc["outcome"]
    for r in rows:
        if "forced_correct" not in r: r["forced_correct"], r["forced_outcome"] = r["correct"], r["outcome"]
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    summarize(rows, a.salvage + "-forced", "forced_correct", "forced_outcome", time.time())


def truncated(r) -> bool:
    """The reply hit the token cap (finish_reason "length"; older result files: unparsed at the cap)."""
    if r.get("finish_reason"): return r["finish_reason"] == "length"
    return r["outcome"] == "unparsed" and r.get("usage", {}).get("completion_tokens", 0) in (2048, 4096, 24000)


def summarize(rows, name, ck="correct", ok="outcome", t0=None):
    groups = defaultdict(list)
    for row in rows:
        if row["task"] == "win":
            key = f"win: {'positive, ' + str(row['meta']['difficulty']) + '-jump' if row['answers']['win'] else 'near-miss negative'}"
        else:
            key = f"{row['task']} placement"
        groups[key].append(row)
        if row["task"] == "win": groups["win (all)"].append(row)
    summary = {}
    print(f"\n{name}: {len(rows)} items" + (f" in {time.time() - t0:.0f}s" if t0 else ""))
    for key in sorted(groups):
        g = groups[key]; acc = sum(r[ck] for r in g) / len(g); trunc = sum(map(truncated, g)) / len(g)
        outc = defaultdict(int)
        for r in g: outc[r[ok]] += 1
        summary[key] = {"n": len(g), "accuracy": round(acc, 3), "truncated": round(trunc, 3), "outcomes": dict(outc)}
        top = dict(sorted(outc.items(), key=lambda kv: -kv[1])[:4])
        print(f"  {key:28s} n={len(g):4d}  accuracy {acc:6.1%}  truncated {trunc:6.1%}   {top}")
    toks = [r.get("usage", {}).get("completion_tokens", 0) for r in rows]
    summary["mean_completion_tokens"] = round(sum(toks) / max(len(toks), 1), 1)
    (ROOT / "llm_bench/results" / f"{name}.summary.json").write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=ROOT / "llm_bench/data/bench.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000/v1"); ap.add_argument("--model", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY"); ap.add_argument("--completions", action="store_true")
    ap.add_argument("--thinking", type=lambda v: v.lower() in ("1", "true", "yes", "on"), default=None,
                    help="Qwen thinking switch (true/false); omitted = the server's default")
    ap.add_argument("--max-tokens", type=int, default=4096); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tasks", default="win,forced,block,prevent")
    ap.add_argument("--baseline", choices=["nowin", "random", "adjacent"], default=None, help="no model: a scoring sanity check")
    ap.add_argument("--name", default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--salvage", default=None, help="results name: budget-force an answer from replies that hit the cap "
                                                     "(needs --model and --thinking as in the original run)")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.bench)]
    if a.salvage: return salvage(a, {it["id"]: it for it in items})
    items = [it for it in items if it["task"] in a.tasks.split(",")]
    if a.limit: items = items[:a.limit]
    name = a.name or (f"baseline-{a.baseline}" if a.baseline else re.sub(r"[^A-Za-z0-9._-]", "_", a.model or "model")
                      + ("" if a.thinking is None else f"-think{int(a.thinking)}") + ("-completions" if a.completions else ""))
    out_dir = ROOT / "llm_bench/results"; out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed); t0 = time.time()

    def run_one(it):
        r = {"reply": baseline_reply(a.baseline, it, rng)} if a.baseline else call(a, it["prompt"])
        return it, r, score(it, r["reply"])

    rows = []
    with cf.ThreadPoolExecutor(max_workers=1 if a.baseline else a.concurrency) as ex:
        for k, (it, r, sc) in enumerate(ex.map(run_one, items)):
            rows.append({"id": it["id"], "task": it["task"], "meta": it["meta"], "answers": it["answers"], **r, **sc})
            if (k + 1) % 20 == 0: print(f"  {k + 1}/{len(items)} | {time.time() - t0:.0f}s", flush=True)
    with open(out_dir / f"{name}.jsonl", "w") as f:
        for row in rows: f.write(json.dumps(row) + "\n")

    summarize(rows, name, t0=t0)


if __name__ == "__main__":
    main()
