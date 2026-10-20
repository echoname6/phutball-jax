#!/usr/bin/env python3
"""Run the transfer suite (llm_transfer/build.py) against an OpenAI-compatible server (vLLM: base model or a LoRA name),
then budget-force an answer from every reply that hit the token cap (as llm_bench --salvage: prompt + truncated thinking
+ "Time is up ..." + "ANSWER:", greedy), and score both.

Per item: finished (the model's own answer) and forced (best answer within the budget); the category of each answer:
correct / false_absent (said NONE / NO / NO WIN when an answer exists) / false_present (gave an answer when the right
one is "none") / wrong / unparsed. Writes llm_transfer/results/<name>.jsonl and <name>.summary.json (per family and dial:
finished / forced accuracy, categories, median completion tokens).

  python -m llm_transfer.run --suite llm_transfer/data/suite_v1.jsonl --model r14 --tokenizer Qwen/Qwen3.5-9B --name r14
"""
from __future__ import annotations

import argparse
import hashlib
import concurrent.futures as cf
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from expert.engine import State  # noqa: E402
from llm_bench.run import FORCE, call  # noqa: E402
from llm_bench.text import check_win_sequence, sq  # noqa: E402
from llm_transfer.build import parse_place  # noqa: E402

ANS = re.compile(r"ANSWER:\s*(.*)", re.I)


def answer_line(text: str) -> str | None:
    m = None
    for m in ANS.finditer(text or ""): pass
    return m.group(1).strip().strip("`*").strip() if m else None


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9,]", "", x.lower())


def split_list(body: str) -> list[str]:
    return [t for t in re.split(r"\s*(?:,|\band\b|->|→|\s)\s*", body.strip().strip(".")) if t]


def score(it: dict, body: str | None) -> tuple[bool, str]:
    """(correct, category) for the text after ANSWER: (None: no answer line)."""
    if body is None or not body.strip(): return False, "unparsed"
    b = body.strip(); low = b.lower()
    said_none = low.startswith(("none", "no win", "nothing", "no route", "no path"))
    k = it["kind"]
    if k == "yesno":
        v = "YES" if low.startswith("yes") else "NO" if low.startswith("no") else None
        if v is None: return False, "unparsed"
        if v == it["answer"]: return True, "correct"
        return False, "false_absent" if v == "NO" else "false_present"
    if said_none and k != "set":
        return (True, "correct") if it["absent"] else (False, "false_absent")
    if k == "bridge":
        it2 = it["item"]; s = State(it2["rows"], it2["cols"], it2["board"], it2["ball"], it2["player"])
        toks = split_list(re.sub(r"^leaps?\b", "", low))
        idx = [parse_place(t, s) for t in toks]
        if not toks or any(i is None for i in idx): return False, "unparsed"
        ok, _ = check_win_sequence(s, [sq(i, s.cols) for i in idx])
        if ok: return True, "correct"
        return False, "false_present" if it["absent"] else "wrong"
    if k == "value":
        if norm(b.split()[0] if it["family"] != "babilong" else b) == norm(it["answer"]) or norm(b) == norm(it["answer"]):
            return True, "correct"
        return False, "false_present" if it["absent"] else "wrong"
    if k == "path":
        toks = split_list(low); edges = {tuple(e) for e in it["edges"]}
        ok = (len(toks) >= 2 and toks[0] == it["start"] and toks[-1] == it["target"] and
              all((toks[i], toks[i + 1]) in edges for i in range(len(toks) - 1)))
        if ok: return True, "correct"
        return False, "false_present" if it["absent"] else "wrong"
    if k == "set":
        got = set() if said_none else {norm(t) for t in split_list(b) if norm(t) not in ("", "var", "the")}
        want = {norm(t) for t in it["answer"]}
        if got == want: return True, "correct"
        if not got and want: return False, "false_absent"
        if got and not want: return False, "false_present"
        return False, "wrong"
    raise ValueError(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, default=ROOT / "llm_transfer/data/suite_v1.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000/v1"); ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-tokens", type=int, default=6144); ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--families", default=""); ap.add_argument("--limit", type=int, default=0, help="items per family (0: all)")
    ap.add_argument("--timeout", type=float, default=1800); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.suite)]
    if a.families: items = [it for it in items if it["family"] in a.families.split(",")]
    if a.limit:
        per = defaultdict(int); keep = []
        for it in items:
            if per[it["family"]] < a.limit: keep.append(it); per[it["family"]] += 1
        items = keep
    gen = argparse.Namespace(claude_code=False, api_key_env="OPENAI_API_KEY", completions=False, model=a.model,
                             max_tokens=a.max_tokens, temperature=0.6, top_p=0.95, top_k=20, seed=a.seed, thinking=True,
                             base_url=a.base_url, timeout=a.timeout)
    t0 = time.time()

    def one(it):
        out = call(gen, it["prompt"])
        reasoning, reply = out.get("reasoning") or "", out.get("reply") or ""
        ans_part = reply.rsplit("</think>", 1)[1] if "</think>" in reply else (reply if reasoning else "")
        body = answer_line(ans_part)
        ok, cat = score(it, body)
        return it, {"id": it["id"], "prompt_sha": hashlib.sha1(it["prompt"].encode()).hexdigest()[:12],
                    "family": it["family"], "dial": it["dial"], "absent": it["absent"], "answer": body,
                    "correct": ok, "category": cat, "finish_reason": out.get("finish_reason"),
                    "tokens": out.get("usage", {}).get("completion_tokens"), "reasoning": reasoning, "reply": reply,
                    "error": out.get("error")}

    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for i, (it, r) in enumerate(ex.map(one, items), 1):
            rows.append(r)
            if i % 50 == 0: print(f"  {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    by_id = {it["id"]: it for it in items}

    # budget forcing for every reply without an answer line (cut off at the cap, or no answer after the thinking)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    force = argparse.Namespace(**{**vars(gen), "completions": True, "max_tokens": 60, "temperature": 0.0})

    def forced(r):
        head = tok.apply_chat_template([{"role": "user", "content": by_id[r["id"]]["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        body = r["reasoning"] + r["reply"]
        out = call(force, None, raw=head + body + FORCE["</think>" not in body] + "ANSWER:")
        return r, (out.get("reply") or "").split("\n")[0]

    todo = [r for r in rows if r["category"] == "unparsed"]
    with cf.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for r, fa in ex.map(forced, todo):
            r["forced_answer"] = fa.strip(); r["forced_correct"], r["forced_category"] = score(by_id[r["id"]], fa)
    for r in rows:
        if "forced_correct" not in r: r["forced_correct"], r["forced_category"] = r["correct"], r["category"]

    out_dir = ROOT / "llm_transfer/results"; out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{a.name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    summary = summarize(rows)
    (out_dir / f"{a.name}.summary.json").write_text(json.dumps(summary, indent=1))
    print_summary(a.name, summary); print(f"{len(rows)} items in {time.time() - t0:.0f}s -> {out_dir / a.name}.jsonl")


def dial_key(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items() if k not in ("difficulty", "needles", "language"))


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], "all")].append(r); groups[(r["family"], dial_key(r["dial"]))].append(r)
    out = {}
    for (fam, key), g in sorted(groups.items()):
        cats = defaultdict(int); fcats = defaultdict(int)
        for r in g: cats[r["category"]] += 1; fcats[r["forced_category"]] += 1
        toks = [r["tokens"] for r in g if r.get("tokens")]
        out[f"{fam} | {key}"] = {"n": len(g), "finished": sum(r["correct"] for r in g) / len(g),
                                 "forced": sum(r["forced_correct"] for r in g) / len(g), "categories": dict(cats),
                                 "forced_categories": dict(fcats), "cut_off": cats.get("unparsed", 0) / len(g),
                                 "median_tokens": statistics.median(toks) if toks else None}
    return out


def print_summary(name, s):
    print(f"\n{name}: finished / forced accuracy, cut off, false absent / false present (forced)")
    for k, v in s.items():
        fc = v["forced_categories"]
        print(f"  {k:46s} n={v['n']:4d}  {v['finished']:6.1%} / {v['forced']:6.1%}  cut {v['cut_off']:5.1%}  "
              f"FA {fc.get('false_absent', 0):3d} FP {fc.get('false_present', 0):3d}  tok {v['median_tokens']}")


if __name__ == "__main__":
    main()
