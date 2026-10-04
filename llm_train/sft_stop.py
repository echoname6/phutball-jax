#!/usr/bin/env python3
"""Self-stop warm start, step 2: a short LoRA fine-tune on llm_train/self_stop.py's examples.

Loss on the completion only (the model's own thinking up to the cut, the stop, its verified answer). The LoRA
configuration matches llm_train/grpo_pilot.py exactly, so the adapter continues into the GRPO curriculum through
--init-adapter.

  python -m llm_train.sft_stop --data DRIVE/self_stop/sft.jsonl --out DRIVE/self_stop/adapter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm_train.grpo_pilot import build_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--accum", type=int, default=8)
    a = ap.parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in open(a.data)]
    data = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    print(f"{len(data)} examples", flush=True)
    cfg = build_config(SFTConfig, dict(
        output_dir=str(a.out), report_to="none", bf16=True, logging_steps=5, save_strategy="no",
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, num_train_epochs=a.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=a.accum, gradient_checkpointing=True,
        max_length=8192, max_seq_length=8192, packing=False, completion_only_loss=True,
        model_init_kwargs={"torch_dtype": torch.bfloat16}))
    lora_kw = dict(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    try:
        lora = LoraConfig(**lora_kw, exclude_modules=r".*(visual|vision|mm_projector).*")   # same as grpo_pilot
    except TypeError:
        lora = LoraConfig(**lora_kw)
    trainer = SFTTrainer(model=a.model, args=cfg, train_dataset=data, peft_config=lora)
    trainer.train()
    trainer.save_model(str(a.out / "final"))
    print("saved", a.out / "final", flush=True)


if __name__ == "__main__":
    main()
