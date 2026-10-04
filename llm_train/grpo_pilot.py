#!/usr/bin/env python3
"""GRPO pilot: Qwen3.5-9B (instruct, thinking ON) + LoRA on the phutball puzzles, rewarded only by the engine.

TRL GRPOTrainer with vLLM colocated on one GPU (A100 80 GB). The thinking is free-form: no format, language or
readability reward (llm_train/grpo_reward.py). Dr. GRPO loss (no length normalisation bias), no KL term.
The pilot measures: wall-clock per step, rollout lengths, truncation, whether reward rises, and drift of the thinking.

  python -m llm_train.grpo_pilot --out /content/drive/MyDrive/phutball/grpo_pilot/run1 --hours 2.5 --cap 6144
  python -m llm_train.grpo_pilot --out /tmp/smoke --smoke          # 2 tiny steps: catches setup errors cheaply

Resumes from the newest checkpoint in --out. Writes: checkpoints, final/ (LoRA adapter), rollouts.jsonl (every
sample with reward and text), steps.log (one reward summary per generation batch).
"""
from __future__ import annotations

import argparse
import dataclasses
from collections import defaultdict
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


STAGES = {                       # curriculum stages: (subtask keep-probabilities, max win length J)
    "wins_short": ({"win_pos": 1.0, "win_neg": 0.2, "forced": 0, "block": 0, "prevent": 0}, 2),
    "wins":       ({"win_pos": 1.0, "win_neg": 0.5, "forced": 0, "block": 0, "prevent": 0}, 4),
    "wins_place": ({"win_pos": 1.0, "win_neg": 0.5, "forced": 1.0, "block": 1.0, "prevent": 0}, 4),
    "full":       ({}, 4),
}


BUDGET_HINT = ("\n\nYou have a thinking budget of about {n:,} tokens. When you have checked your answer, stop thinking "
               "and give it.")


def load_puzzles(path: Path, seed: int, mix: dict | None, max_j: int = 4, hint: int = 0):
    from expert.engine import State
    from llm_bench.text import prompt
    rows = [json.loads(l) for l in open(path)]
    rng = random.Random(seed); rng.shuffle(rows)
    rows = [r for r in rows if r["subtask"] != "win_pos" or int(r["item"]["meta"].get("J", 1)) <= max_j]
    if mix:                                                             # optional subtask weights, e.g. prevent=0.5
        rows = [r for r in rows if rng.random() < mix.get(r["subtask"], 1.0)]
    out = []
    for r in rows:
        it = r["item"]; s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"])
        text = prompt(it["task"], s) + (BUDGET_HINT.format(n=hint) if hint else "")
        out.append({"prompt": [{"role": "user", "content": text}], "item": json.dumps(it),
                    "task": it["task"], "subtask": r["subtask"]})
    return out


def build_config(GRPOConfig, want: dict):
    """Keep the options this TRL version knows; print the rest (TRL renames options between releases)."""
    fields = {f.name: f for f in dataclasses.fields(GRPOConfig)}
    cfg, dropped = {}, []
    for k, v in want.items():
        if k not in fields: dropped.append(k); continue
        d = fields[k].default
        if k == "scale_rewards" and isinstance(d, str): v = "none"     # newer TRL: "group" | "batch" | "none"
        cfg[k] = v
    if dropped: print("TRL does not know these options (skipped):", dropped, flush=True)
    return GRPOConfig(**cfg)


def find_llm(obj, depth: int = 0, seen=None):
    """The vllm.LLM engine TRL created for colocate mode (attribute names change between TRL releases)."""
    import vllm
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 3: return None
    seen.add(id(obj))
    if isinstance(obj, vllm.LLM): return obj
    for v in (vars(obj).values() if hasattr(obj, "__dict__") else []):
        if isinstance(v, (str, int, float, bool, bytes)) or v is None: continue
        try:
            r = find_llm(v, depth + 1, seen)
        except Exception:                                               # noqa: BLE001
            r = None
        if r is not None: return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--puzzles", type=Path, default=ROOT / "llm_train/puzzles/grpo_train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=6144, help="max completion (thinking + answer) tokens")
    ap.add_argument("--prompts", type=int, default=4, help="puzzles per optimizer step")
    ap.add_argument("--gens", type=int, default=8, help="samples per puzzle (the GRPO group)")
    ap.add_argument("--micro", type=int, default=1, help="completions per forward/backward pass (vLLM stays resident when forcing)")
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--hours", type=float, default=2.5, help="stop (and save) after this much training time")
    ap.add_argument("--max-steps", type=int, default=1000); ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--vllm-mem", type=float, default=0.35); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", default="", help="subtask keep-probabilities, e.g. prevent=0.5,block=0.7")
    ap.add_argument("--force-credit", type=float, default=0.75,
                    help="credit for a correct budget-forced answer on a cut-off rollout (0: off)")
    ap.add_argument("--force-engine", choices=["hf", "vllm"], default="hf",
                    help="hf: the training model completes the forced answer (vLLM can sleep: less memory); "
                         "vllm: the colocated engine (needs it resident: OOM at 6k tokens on one A100)")
    ap.add_argument("--force-batch", type=int, default=2)
    ap.add_argument("--force-budgets", default="1",
                    help="fractions of the cap at which a cut-off rollout is forced, e.g. 0.25,0.5,1 (anytime credit)")
    ap.add_argument("--stage", choices=sorted(STAGES), default=None, help="curriculum stage (sets the puzzle mix)")
    ap.add_argument("--init-adapter", type=Path, default=None, help="start from this LoRA adapter (the previous stage)")
    ap.add_argument("--advance-correct", type=float, default=0.0,
                    help="stop the stage once the finished-and-correct share over the last --advance-window "
                         "generation batches reaches this (0: run for --hours)")
    ap.add_argument("--advance-window", type=int, default=5)
    ap.add_argument("--budget-hint", type=int, default=0,
                    help="append 'You have a thinking budget of about N tokens ...' to every prompt (0: off); "
                         "evaluate with the same hint (llm_bench.run --budget-hint)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.cap, a.prompts, a.gens, a.micro, a.max_steps, a.hours, a.save_every = 512, 2, 4, 2, 2, 0.5, 1000
    a.out.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from llm_train.grpo_reward import PhutballReward
    print("torch", torch.__version__, "transformers", transformers.__version__, "trl", trl.__version__, flush=True)

    mix = dict((k, float(v)) for k, v in (p.split("=") for p in a.mix.split(",") if p)) if a.mix else None
    max_j = 4
    if a.stage: mix, max_j = STAGES[a.stage]
    data = Dataset.from_list(load_puzzles(a.puzzles, a.seed, mix, max_j, a.budget_hint))
    if a.budget_hint: print("budget hint:", BUDGET_HINT.format(n=a.budget_hint).strip(), flush=True)
    print(f"{len(data)} puzzles", flush=True)

    per_step = a.prompts * a.gens
    cfg = build_config(GRPOConfig, dict(
        output_dir=str(a.out), seed=a.seed, report_to="none", logging_steps=1, bf16=True,
        learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=2, max_grad_norm=1.0,
        per_device_train_batch_size=a.micro, gradient_accumulation_steps=per_step // a.micro,
        num_generations=a.gens, max_prompt_length=1536, max_completion_length=a.cap,
        temperature=1.0, top_p=1.0, beta=0.0, loss_type="dr_grpo", scale_rewards=False,
        mask_truncated_completions=False,                               # truncation is penalised, not hidden
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem,
        vllm_enable_sleep_mode=not (a.force_credit and a.force_engine == "vllm"),   # vLLM forcing needs it awake
        vllm_max_model_length=1536 + a.cap + 64,
        gradient_checkpointing=True, max_steps=a.max_steps, save_steps=a.save_every, save_total_limit=2,
        chat_template_kwargs={"enable_thinking": True}, model_init_kwargs={"torch_dtype": torch.bfloat16},
        log_completions=False))

    lora_kw = dict(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    try:
        lora = LoraConfig(**lora_kw, exclude_modules=r".*(visual|vision|mm_projector).*")   # language model only
    except TypeError:
        lora = LoraConfig(**lora_kw)

    reward = PhutballReward(cap=a.cap, log_path=str(a.out / "rollouts.jsonl"), force_credit=a.force_credit,
                            budgets=tuple(float(x) for x in a.force_budgets.split(",")))

    class TimeLimit(transformers.TrainerCallback):
        def __init__(self): self.t0 = time.time()

        def on_step_end(self, args, state, control, **kw):
            el = time.time() - self.t0
            print(f"[step {state.global_step}] {el / 60:.1f} min elapsed, {el / max(state.global_step, 1) / 60:.1f} min/step",
                  flush=True)
            with open(a.out / "steps.log", "a") as f:
                f.write(json.dumps({"step": state.global_step, "elapsed_s": round(el), "reward": reward.history[-1:]}) + "\n")
            done = None
            if el > a.hours * 3600: done = "time"
            st = reward.stats[-a.advance_window:]
            if a.advance_correct and len(st) >= a.advance_window:
                pooled = defaultdict(lambda: [0, 0])                    # per subtask over the window
                for x in st:
                    for k, (c, n) in x.get("per", {}).items(): pooled[k][0] += c; pooled[k][1] += n
                rates = {k: c / n for k, (c, n) in pooled.items() if n}
                if rates and min(rates.values()) >= a.advance_correct:   # every subtask, not the pooled mean
                    done = "advanced"
            if done:
                control.should_training_stop = True; control.should_save = True
                (a.out / "stage_done.json").write_text(json.dumps(
                    {"reason": done, "steps": state.global_step, "elapsed_s": round(el), "last": st}))

    trainer = GRPOTrainer(model=a.model, reward_funcs=[reward], args=cfg, train_dataset=data, peft_config=lora,
                          callbacks=[TimeLimit()])
    if a.force_credit:
        tok = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        tok = getattr(tok, "tokenizer", tok)                            # a processor wraps the tokenizer
        llm = find_llm(trainer) if a.force_engine == "vllm" else None
        if a.force_engine == "hf" and tok is not None:
            model = trainer.model

            def hf_force(raws):
                """Greedy 12-token completions from the current policy (LoRA on), left-padded small batches."""
                outs, was_training, side = [], model.training, tok.padding_side
                model.eval(); tok.padding_side = "left"
                pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                try:
                    with torch.no_grad():
                        for i in range(0, len(raws), a.force_batch):
                            enc = tok(raws[i:i + a.force_batch], return_tensors="pt", padding=True,
                                      add_special_tokens=False).to(model.device)
                            g = model.generate(**enc, max_new_tokens=12, do_sample=False, use_cache=True, pad_token_id=pad)
                            outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                finally:
                    tok.padding_side = side
                    if was_training: model.train()
                    torch.cuda.empty_cache()
                return outs
            reward.forcer = hf_force
            reward.decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
            reward.template = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                                   enable_thinking=True)
            print("budget forcing ON (training model, vLLM sleeps between generations)", flush=True)
        elif llm is None or tok is None:
            print("WARNING: no vLLM engine / tokenizer found: budget forcing OFF", flush=True); reward.force_credit = 0
        else:
            from vllm import SamplingParams
            sp = SamplingParams(max_tokens=12, temperature=0.0)
            reward.forcer = lambda raws: [o.outputs[0].text for o in llm.generate(raws, sp, use_tqdm=False)]
            reward.decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
            reward.template = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                                   enable_thinking=True)
            print("budget forcing ON (vLLM engine found)", flush=True)
    if a.init_adapter:                                                  # continue the previous stage's LoRA
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(trainer.model, load_file(str(a.init_adapter / "adapter_model.safetensors")))
        miss = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
        print(f"loaded adapter {a.init_adapter} (missing LoRA keys: {len(miss)})", flush=True)
    ckpts = sorted(a.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(str(a.out / "final"))
    print("saved", a.out / "final", flush=True)


if __name__ == "__main__":
    main()
