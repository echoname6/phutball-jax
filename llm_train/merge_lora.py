#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model (on CPU; Colab high-RAM) so vLLM can serve it like any model.

  python -m llm_train.merge_lora --adapter RUN/final --out /content/merged
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B"); ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import transformers
    from peft import PeftModel
    model = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            model = getattr(transformers, cls).from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map="cpu"); break
        except Exception as e:                                          # noqa: BLE001
            print(cls, "failed:", repr(e)[:200])
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True)
    for name in ("AutoTokenizer", "AutoProcessor"):
        try: getattr(transformers, name).from_pretrained(a.base).save_pretrained(a.out)
        except Exception as e: print(name, "skipped:", repr(e)[:120])   # noqa: E701
    print("merged ->", a.out)


if __name__ == "__main__":
    main()
