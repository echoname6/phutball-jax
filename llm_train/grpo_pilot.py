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
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_puzzles(path: Path, seed: int, mix: dict | None):
    from expert.engine import State
    from llm_bench.text import prompt
    rows = [json.loads(l) for l in open(path)]
    rng = random.Random(seed); rng.shuffle(rows)
    if mix:                                                             # optional subtask weights, e.g. prevent=0.5
        rows = [r for r in rows if rng.random() < mix.get(r["subtask"], 1.0)]
    out = []
    for r in rows:
        it = r["item"]; s = State(it["rows"], it["cols"], it["board"], it["ball"], it["player"])
        out.append({"prompt": [{"role": "user", "content": prompt(it["task"], s)}], "item": json.dumps(it),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--puzzles", type=Path, default=ROOT / "llm_train/puzzles/grpo_train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=6144, help="max completion (thinking + answer) tokens")
    ap.add_argument("--prompts", type=int, default=4, help="puzzles per optimizer step")
    ap.add_argument("--gens", type=int, default=8, help="samples per puzzle (the GRPO group)")
    ap.add_argument("--micro", type=int, default=2, help="completions per forward/backward pass")
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--hours", type=float, default=2.5, help="stop (and save) after this much training time")
    ap.add_argument("--max-steps", type=int, default=1000); ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--vllm-mem", type=float, default=0.35); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", default="", help="subtask keep-probabilities, e.g. prevent=0.5,block=0.7")
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
    data = Dataset.from_list(load_puzzles(a.puzzles, a.seed, mix))
    print(f"{len(data)} puzzles", flush=True)

    per_step = a.prompts * a.gens
    cfg = build_config(GRPOConfig, dict(
        output_dir=str(a.out), seed=a.seed, report_to="none", logging_steps=1, bf16=True,
        learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=2, max_grad_norm=1.0,
        per_device_train_batch_size=a.micro, gradient_accumulation_steps=per_step // a.micro,
        num_generations=a.gens, max_prompt_length=1536, max_completion_length=a.cap,
        temperature=1.0, top_p=1.0, beta=0.0, loss_type="dr_grpo", scale_rewards=False,
        mask_truncated_completions=False,                               # truncation is penalised, not hidden
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem, vllm_enable_sleep_mode=True,
        vllm_max_model_length=1536 + a.cap + 64,
        gradient_checkpointing=True, max_steps=a.max_steps, save_steps=a.save_every, save_total_limit=2,
        chat_template_kwargs={"enable_thinking": True}, model_init_kwargs={"torch_dtype": torch.bfloat16},
        log_completions=False))

    lora_kw = dict(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    try:
        lora = LoraConfig(**lora_kw, exclude_modules=r".*(visual|vision|mm_projector).*")   # language model only
    except TypeError:
        lora = LoraConfig(**lora_kw)

    reward = PhutballReward(cap=a.cap, log_path=str(a.out / "rollouts.jsonl"))

    class TimeLimit(transformers.TrainerCallback):
        def __init__(self): self.t0 = time.time()

        def on_step_end(self, args, state, control, **kw):
            el = time.time() - self.t0
            print(f"[step {state.global_step}] {el / 60:.1f} min elapsed, {el / max(state.global_step, 1) / 60:.1f} min/step",
                  flush=True)
            with open(a.out / "steps.log", "a") as f:
                f.write(json.dumps({"step": state.global_step, "elapsed_s": round(el), "reward": reward.history[-1:]}) + "\n")
            if el > a.hours * 3600: control.should_training_stop = True; control.should_save = True

    trainer = GRPOTrainer(model=a.model, reward_funcs=[reward], args=cfg, train_dataset=data, peft_config=lora,
                          callbacks=[TimeLimit()])
    ckpts = sorted(a.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(str(a.out / "final"))
    print("saved", a.out / "final", flush=True)


if __name__ == "__main__":
    main()
