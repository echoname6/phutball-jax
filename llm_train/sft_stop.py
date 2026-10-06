#!/usr/bin/env python3
"""Self-stop warm start, step 2: a short LoRA fine-tune on llm_train/self_stop.py's examples.

Loss on the completion only (the model's own thinking up to the cut, the stop, its verified answer). The LoRA
configuration matches llm_train/grpo_pilot.py exactly, so the adapter continues into the GRPO curriculum through
--init-adapter.

  python -m llm_train.sft_stop --data DRIVE/self_stop/sft.jsonl --out DRIVE/self_stop/adapter

--init-adapter continues an existing adapter (same rank and modules) instead of a fresh LoRA on the base model: the
v4 curriculum carries the weights from round to round (v3 retrained from base every round, so no round built on the
previous one and even the warm start was not carried over).
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
    ap.add_argument("--max-neg-ratio", type=float, default=0.6,
                    help="keep at most this many no-win examples per win example (the first warm start had 173 no-win "
                         "vs 146 win and started RL leaning to NO WIN: wins 5/16 vs no-wins 12/16 before any update)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-adapter", type=Path, default=None,
                    help="continue this LoRA adapter (trained with the same --rank) instead of a fresh one")
    a = ap.parse_args()
    if a.init_adapter:
        r0 = json.loads((a.init_adapter / "adapter_config.json").read_text())["r"]
        if r0 != a.rank: sys.exit(f"--init-adapter has rank {r0}, not --rank {a.rank}")
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    import random
    rows = [json.loads(l) for l in open(a.data)]
    neg = [r for r in rows if r.get("subtask") == "win_neg"]; pos = [r for r in rows if r.get("subtask") == "win_pos"]
    keep = int(a.max_neg_ratio * len(pos))
    if len(neg) > keep:
        random.Random(a.seed).shuffle(neg)
        rows = [r for r in rows if r.get("subtask") != "win_neg"] + neg[:keep]
    from collections import Counter
    print("examples by subtask:", dict(Counter(r.get("subtask") for r in rows)), flush=True)
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
    if a.init_adapter:                                                  # continue it (same as grpo_pilot)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(str(a.init_adapter / "adapter_model.safetensors"))
        res = set_peft_model_state_dict(trainer.model, sd)
        miss = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
        unexp = [k for k in getattr(res, "unexpected_keys", []) if "lora_" in k]
        if miss or unexp: sys.exit(f"adapter mismatch: {len(miss)} missing, {len(unexp)} unexpected LoRA keys "
                                   f"(e.g. {(miss + unexp)[:3]})")
        print(f"continuing adapter {a.init_adapter} ({len(sd)} tensors)", flush=True)
    trainer.train()
    trainer.save_model(str(a.out / "final"))
    print("saved", a.out / "final", flush=True)


if __name__ == "__main__":
    main()
