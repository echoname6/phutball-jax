#!/usr/bin/env python3
"""Is a GRPO phase's LoRA actually training? Prints, for every adapter under a run folder (checkpoint-N/ and final/),
how many lora_B tensors are nonzero and their largest / mean absolute value. A fresh LoRA starts with every lora_B at
zero (the merged weights equal the base model); after optimizer steps they should be nonzero and growing.

  python -m llm_train.inspect_adapter /content/drive/MyDrive/phutball/grpo_v2_base_curve/p1
"""
import glob
import os
import sys

from safetensors import safe_open


def main():
    root = sys.argv[1]
    paths = sorted(glob.glob(f"{root}/checkpoint-*/adapter_model.safetensors"),
                   key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
    paths += glob.glob(f"{root}/final/adapter_model.safetensors")
    if not paths: print("no adapters under", root); return
    for p in paths:
        n = nz = 0; mx = tot = 0.0; cnt = 0
        with safe_open(p, "pt") as f:
            for k in f.keys():
                if "lora_B" not in k: continue
                t = f.get_tensor(k).float().abs(); n += 1
                nz += int(t.max() > 0); mx = max(mx, float(t.max())); tot += float(t.sum()); cnt += t.numel()
        print(f"{os.path.relpath(p, root):45s} lora_B tensors {n:4d}, nonzero {nz:4d}, max |B| {mx:.2e}, "
              f"mean |B| {tot / max(cnt, 1):.2e}")


if __name__ == "__main__":
    main()
