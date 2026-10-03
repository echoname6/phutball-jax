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
win negatives: correct if NO WIN; forced: correct if the placement is one of the (complete) unstoppable placements.
Results: llm_bench/results/<name>.jsonl (every reply) and a summary printed and saved as <name>.summary.json.
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
from llm_bench.text import check_win_sequence, parse_answer, sq  # noqa: E402


def call(a, prompt_text: str) -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(a.api_key_env)
    if key: headers["Authorization"] = f"Bearer {key}"
    if a.completions:
        body = {"model": a.model, "prompt": prompt_text + "\n\nReply:\n", "max_tokens": a.max_tokens, "temperature": a.temperature}
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
            return {"reply": text, "reasoning": reasoning, "usage": out.get("usage", {})}
        except Exception as e:                               # noqa: BLE001  (retry transient errors)
            err = repr(e); time.sleep(2 ** attempt)
    return {"reply": "", "error": err}


def baseline_reply(kind: str, it: dict, rng: random.Random) -> str:
    s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"])
    if it["task"] == "win":
        if kind == "nowin": return "ANSWER: NO WIN"
        js = jump_landings(s.board, s.ball, s.rows, s.cols)
        return "ANSWER: " + (f"JUMP {sq(rng.choice(js)[0], s.cols)}" if js and rng.random() < 0.5 else "NO WIN")
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
    if kind != "place": return {"correct": False, "outcome": "wrong answer type"}
    ok = val in it["answers"]["placements"]
    return {"correct": ok, "outcome": "unstoppable" if ok else "not unstoppable"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=ROOT / "llm_bench/data/bench.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000/v1"); ap.add_argument("--model", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY"); ap.add_argument("--completions", action="store_true")
    ap.add_argument("--thinking", type=lambda v: v.lower() in ("1", "true", "yes", "on"), default=None,
                    help="Qwen thinking switch (true/false); omitted = the server's default")
    ap.add_argument("--max-tokens", type=int, default=4096); ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tasks", default="win,forced")
    ap.add_argument("--baseline", choices=["nowin", "random"], default=None, help="no model: a scoring sanity check")
    ap.add_argument("--name", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.bench)]
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

    groups = defaultdict(list)
    for row in rows:
        if row["task"] == "win":
            key = f"win: {'positive, ' + str(row['meta']['difficulty']) + '-jump' if row['answers']['win'] else 'near-miss negative'}"
        else:
            key = "forced placement"
        groups[key].append(row); groups["win (all)" if row["task"] == "win" else "forced (all)"].append(row)
    summary = {}
    print(f"\n{name}: {len(rows)} items in {time.time() - t0:.0f}s")
    for key in sorted(groups):
        g = groups[key]; acc = sum(r["correct"] for r in g) / len(g)
        outc = defaultdict(int)
        for r in g: outc[r["outcome"]] += 1
        summary[key] = {"n": len(g), "accuracy": round(acc, 3), "outcomes": dict(outc)}
        print(f"  {key:28s} n={len(g):4d}  accuracy {acc:6.1%}   {dict(outc)}")
    toks = [r.get("usage", {}).get("completion_tokens", 0) for r in rows]
    summary["mean_completion_tokens"] = round(sum(toks) / max(len(toks), 1), 1)
    (out_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
