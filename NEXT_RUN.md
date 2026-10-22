# Next run plan (gated GRPO, run 3 / second seed) — written 2026-10-22

Run 2 (`grpo_gated_v2_colab.ipynb`, START=base, curve reward, 16k, prior correction, headroom targeting, token-level IS,
32-token training forcing) is the reference recipe. Its p1 (clean, after the 12-token fix) beat base on unseen
validation: peak (forced at 16k) 0.858 -> 0.890 (paired +88/-56, z +2.67), finished by 16k 48% -> 84%, forced at 4k on
4-jump wins 49% -> 70%. The items below are for the next clean run / second seed.

## Before any long run: a 5-minute preflight (build it)
Three GRPO steps plus checks that would each have caught one of run 2's silent failures within minutes:
- max |lora_B| grows every step (TRL sequence-level IS at ~1e-18 left the LoRA untrained);
- rollouts change between steps (compare two seeds' rollouts.jsonl with `llm_train.compare_rollouts`);
- forced answers on known puzzles come back complete and correct, including 4-jump chains with two-digit rows
  (the 12-token cap cut them and scored them as false claims);
- vLLM importance-sampling ratios near 1 (no top-k / top-p with IS on);
- no orphaned `grpo_pilot` / `vllm serve` processes holding GPU memory.

## Puzzle data
- **Training pool**: run 2 used the 1,059 puzzles v4 r14 found learnable (an arbitrary filter for a base start, and
  only 67 block / 41 forced / 41 prevent). Draw from the full `pool_v4` (13,864: ~1,900 block, ~1,970 forced, 140
  prevent) or generated puzzles, so placements are not repeated across phases (memorisation risk in the training
  numbers; validation is disjoint, so the gate is unaffected).
- **Constructive placement generator**: build the attacker's threat first, keep positions where no square next to the
  ball works but a farther one does (block: later landing squares / chain breaks; forced; prevent), add decoys, grade by
  chain length, distance of the answer from the ball and number of correct squares. Measured: block and forced in
  val / pool already have 0% adjacent answers (block averages 1.2 correct squares); prevent has 19-23%.

### Prevent (the task that needs a full opponent turn of look-ahead)
- **Test on strict only**: `llm_bench/data/prevent_v2_test.jsonl` (200) and strict validation; never train on these.
- **Train on a mix of strict and relaxed**, with **at least half strict** (mirrored left-right to double them). Reason:
  relaxed puzzles (`prevent_v2_relaxed`, 294) still allow sideways squares next to the ball, which save ~45% of the
  time; with the curve reward paying for early answers, a model can learn "place beside the ball, claim it saves, stop"
  without checking for a remaining unstoppable placement. With >= 50% strict the shortcut succeeds ~20% and has clearly
  negative expected reward (a wrong answer at t costs -w(t), ~-1 early), so verifying is what pays.
- Use relaxed puzzles in order of fewest saving squares next to the ball (as `split_prevent_v2` does); drop the most
  shortcut-prone if needed.
- **Monitor the shortcut**: on the strict test set, track false claims and their length. A learned shortcut shows as
  rising strict false claims arriving at shorter lengths.
- **Later**: adversarial decoys from the constructive generator (a sideways square that looks saving but leaves an
  unstoppable placement) to train the verification step directly.
- **More strict puzzles**: `make_prevent_v2 --strict` yields ~0.45 per worker-minute on 21x15 (~37 core-hours per
  1,000); try a 10-minute yield test at 3-4 board sizes first.

### Board sizes
- All LLM puzzle sets are 21x15. Option: train with a few sizes mixed in (e.g. 19x13, 21x15, 23x17) for generality and
  prevent yield; keep validation at 21x15 for comparability; add a size-transfer eval (e.g. 25x19).

## Measurement
- Held-out test set (`bench_v2`, curve eval) for base vs the final model: numbers the gate never selected on.
- k-sample (k=4) re-evaluation of the hardest buckets (win3b / win4 / win4b, placements) to firm up per-bucket gains.
- **pass@k against base at large k** (Yue et al. 2025 vs ProRL): needed to claim new capability rather than sharpening.
- Second seed of the same recipe.
- Ablations: curve reward vs plain correctness at the same 16k cap vs a standard length penalty (is it the reward or
  the budget?).
- Re-run the transfer sweeps (bridge, ladder, far families) on run 2's best: run 1's "efficiency only" transfer result
  was measured on models trained with the 12-token bias.

## Other options
- Faster hardware: 1x H100/H200 (~2-3x, no code change; on H200 keep vLLM awake and force with vLLM); 8x H100 needs
  vLLM server mode + multi-GPU training (~a day of setup).
- Train in the xy coordinate format as well (every model is most accurate there on the ladder).
- Hints (first jump given, then removed) for puzzles never solved even when forced.

## Interpretability: where the gains live, and when the answer becomes "known" (Qwen3.5-9B first, Beam later)

Question: which circuitry did RL grow, and which did it repurpose? Assets: the change is a LoRA (low rank, switchable
per layer), and puzzles give exact minimal pairs (move one man: a jump appears, a win disappears). Order:

1. **LoRA anatomy + per-layer ablation** (cheap, causal): update norms per layer / module, top singular directions of
   each delta; switch the LoRA off one layer at a time and measure what is lost: commitment (finished at 16k), search
   speed (forced at 4k), ceiling (forced at 16k). Localises the gains before anything else.
2. **J-lens readout of the answer squares across the thinking trace** (Gurnee et al., "Verbalizable Representations
   Form a Global Workspace in Language Models", arXiv:2607.15495; readout analysis arXiv:2608.25347). The Jacobian lens
   gives, per vocabulary token, the residual-stream direction that makes the model more likely to say that token later
   (the "J-space", a global workspace that carries silent intermediate reasoning). Here: is the winning landing square
   already "poised to be said" before the model writes it? Base vs run-2 checkpoints:
   - in J-space earlier -> search efficiency as a mechanism;
   - in J-space but the model keeps thinking -> the commitment gap;
   - present after RL on puzzles where base never has it -> the ceiling gain, from inside.
   Validation: the J-lens onset should track the behavioural forced-accuracy-by-budget curves.
   Practical: squares are multi-token (" h", "1", "0"): read out the column-letter token or score whole squares;
   restrict the Jacobian to the ~30-40 square tokens (and NO / WIN) to keep it affordable; compute with autograd on
   the open weights. Qwen3.5 is a hybrid (linear + full attention): residual-stream methods are unaffected.
   Note the paper's "counterfactual reflection training" (train what the model would say if interrupted) is
   conceptually close to our forced answers.
3. **Causal tests**: activation patching / interchange interventions (distributed alignment search) on the found
   directions with minimal-pair puzzles ("a win exists" swapped in or out); steering with the answer / "found it"
   direction to see whether the model commits earlier.
4. **Crosscoders (model diffing base vs RL)** at the 2-3 layers that step 1 flags: features unique to the RL model
   (grown) vs shared features whose weights / role changed (repurposed). Most expensive (tens of millions of tokens of
   activations per layer from both models).

**Beam (Reflection AI, 501B MoE, 23B active, open weights) only after this works on the 9B model**: the J-lens needs
weights and gradients, so it cannot run through an inference API; it would need the weights hosted on a multi-GPU node
(~1 TB in bf16, ~250 GB in 4-bit), and MoE routing adds a layer (which experts carry the answer). The API-only
experiment for Beam remains the budget sweep (forced vs finished gap), which decides whether a Beam fine-tune is worth it.

## Writeup framing
- The curve reward follows anytime-reasoning work (AnytimeReasoner, arXiv:2505.13438; S-GRPO, arXiv:2505.07686;
  Elastic Reasoning, arXiv:2505.05315); cite them. Ours: an exact expectation over a continuous unknown deadline from
  one rollout.
- New: a procedurally generated, difficulty-graded suite that separates commitment, search efficiency and ceiling;
  a matched before/after showing a truncated forced answer (12 tokens) hid the ceiling gain (peak z +0.72 vs +2.67,
  same seeds and puzzles) and pushed the model to give up with NO WIN (cf. DAPO, arXiv:2503.14476, on truncation noise).
