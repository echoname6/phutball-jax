# Phutball handoff (2026-09-30)

State of the phutball model-training work: the pipeline, results, puzzle infrastructure, the Elo tournament that is
starting, and the plan for grounding the owner's own play against the model. Written for another Claude session picking
this up. Numbers are from the logs; where a figure is an estimate it says so.

## Where things live

- **Repo**: `~/Projects/phutball-jax`, branch `expert-dagger` (pushed to `github.com/echoname6/phutball-jax`; the active
  `gh` account is `kalen6k`, so push with `git push "https://x-access-token:$(gh auth token --user echoname6)@github.com/echoname6/phutball-jax.git" expert-dagger`;
  do not switch the active account). Commits: `user.name=kalen`, `user.email=echoname6@gmail.com`.
- **Local Python**: `~/Projects/phutball/venv/bin/python` (JAX 0.8.1). Colab runs JAX 0.11.1 / Flax 0.11.2; scripts carry a
  two-line shim for `jax.core.get_opaque_trace_state`.
- **Compute**: Colab Pro+ on the `kalen6k` Google account, A100 at about 7 credits per hour per session (about 1,570
  credits at the start; roughly 200-300 used so far, estimate). Everything on Colab reads and writes `My Drive/phutball/`:
  `data/expert_data/` (pools, checkpoints, `puzzle_eval_deep.npz`), `selfplay/`, `grpo2/`, `looped/`, `elo/`.
- **Local data**: `expert_data/` in the repo (gitignored): pools, placement pools, games, curriculum/ladder/refresh
  checkpoints, `puzzle_eval.npz`, `puzzle_eval_deep.npz`.
- **Browser app**: `~/Projects/phutball` (separate repo, React frontend in `phutball-frontend/`).

## Pipeline (in order)

1. **Puzzle curriculum on the laptop** (`expert/train_curriculum.py`): the repo's 6-layer x 128 transformer (805k params)
   on engine-relabelled jump-chain puzzles, stage k = chains of 1..k jumps (k up to 4). Every state of a winning chain is
   labelled; targets are all winning moves, softmax-weighted toward shorter wins.
2. **Placement stage** (`expert/train_mix.py`): jump puzzles plus placement puzzles (unstoppable threat, block, prevent).
3. **Expert-game board ladder** (`expert/train_ladder.py`): imitation of the 2-ply scripted expert (`expert/oracle.py`)
   on 13x9 -> 15x11 -> 17x13 -> 21x15, one size-agnostic parameter set (per-cell tokens, goal-distance positional
   encoding), KL anchors to the puzzle model and to the previous rung, plateau promotion.
4. **Puzzle refresh** (500 steps) to restore jump-chain accuracy lost during imitation.
5. **Self-play board ladder on Colab** (`expert/selfplay_ladder.py`, `expert/selfplay_fast.py`,
   `selfplay_ladder_colab.ipynb`): Gumbel-MCTS self-play in batched JAX with continuous slot refilling; gating against the
   best network (AlphaGo Zero style, new best at >= 55%); search schedule 32x16 on small boards, 17x13 at 32 then 64 sims,
   21x15 at 64 then 128 sims; each plateau raises the budget, and the final plateau ends the run. Live `control.json` in
   the run folder (slots, games, batch, eval_every, patience, eval_sims, eval_max_turns=360, reload -> exit 3, stop).
6. **PBT over GRPO** (`expert/grpo.py`, `expert/grpo_pbt.py`, `grpo_pbt_colab.ipynb`, run folder `grpo2/`): tried to push
   past the self-play final. Stopped; see results.
7. **Looped transformer experiment** (`expert/looped.py`, `expert/train_looped.py`, `looped_colab.ipynb`): learned
   iteration versus explicit search. Done; see results.
8. **Elo tournament and PUCT calibration** (`expert/elo_tournament.py`, `expert/puct_calibration.py`, `elo_colab.ipynb`):
   starting now.

## Puzzle infrastructure (all engine-proven)

- `expert/engine.py`: fast Python mirror of the JAX env (landing in either end zone ends the game; goal rows 1 and R-2
  hold zone markers but are legal placement squares, which several early bugs got wrong).
- `expert/puzzle_data.py`: jump-chain puzzles, relabelled exhaustively (`best_completions`), mirror plus real-game clutter.
- `expert/puzzle_backward.py`: chains that require sideways or backward jumps, with decoy forward jumps; a puzzle is kept
  only if "always jump toward the goal" fails under both tie-breaks. The original win puzzles were 96% solvable by that rule.
- `expert/puzzle_place.py`: `make_forced` (unstoppable threats, checked against every opponent jump and every
  landing-square placement), `make_block` (pure-placement blocks), `make_prevent` (engine-verified denial: stop an
  unstoppable threat one turn early). All searches are strict: a search that hits its cap raises `CapHit` and the puzzle
  is dropped. The repo's original loop-then-advance denial puzzles were rejected (skipping the loop costs nothing one or
  two moves deep).
- Eval sets: `expert_data/puzzle_eval.npz` (standard, by true difficulty) and `expert_data/puzzle_eval_deep.npz`
  (423 puzzles, true depth 5-10, never trained on). The deep set has only about 40 puzzles per depth (about +/-8 points);
  a 200-per-depth version is needed before claiming 5-10 point differences.

## Results

**Self-play ladder** (about 300k+ games in total; estimate from the logs):

| Rung | Iterations | Notes |
| --- | --- | --- |
| 13x9 | 30 | plateau |
| 15x11 | 80 | hit the iteration cap; promoted the best (it 75) |
| 17x13 | 125 | 32 sims plateaued at it 25-40; 64 sims gave six new bests in a row; promoted best it 110 |
| 21x15 | 120 | 64 sims: bests to it 50; 128 sims: bests at it 80, 90, 100; it 110 and 120 lost to it 100 (45.5%, 29.1%); run stopped, final = best it 100 |

Checkpoints in `selfplay/`: `selfplay_21x15_switch_s128_it50.pkl` (the Elo reference), `selfplay_21x15_best_it80/90/100.pkl`,
`selfplay_21x15_final.pkl` (= it 100), rung finals (`selfplay_{13x9,15x11,17x13}_final.pkl`), replay `buffers_21x15.npz`
(300k positions with 128-sim targets, from roughly iterations 111-120).

Tactics stayed at 98-100% on held-out jump chains and 90-99% on placement puzzles throughout self-play.

**Search on deep chains** (self-play final, noisy root; 40 puzzles per depth):

| Depth | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- |
| no search (6 blocks) | 92% | 88% | 80% | 60% | 46% | 58% |
| 32 sims (198 blocks) | 92% | 88% | 82% | 70% | 54% | 67% |
| 128 sims (774 blocks) | 95% | 88% | 92% | 88% | 77% | 75% |

**PBT over GRPO**: the first attempt (plain REINFORCE, 50 steps per batch) collapsed the policy in one round (KL 0.56,
5.5% vs the anchor). With a PPO clip (eps 0.2, lr about 2e-5, 16 steps per batch) it was stable (KL 0.01-0.04) but no
member beat the anchor over two rounds: 40.6-53.5% in round 1, 48.0-52.3% in round 2 (+/-4.4). Conclusion: GRPO
fine-tuning did not improve the converged self-play network. Together with self-play degrading after its best, this
points to a plateau of this network or setup; not yet tested: lower-lr self-play (about 3e-5), league play against
past bests, a larger network.

**Looped transformer** (all trained 6,000 steps on identical data: jump puzzles <= 4 jumps, placement puzzles, the
128-sim search targets from the replay; blocks = transformer block applications per move):

| Model | Params | Blocks/move | Search-target top-1 | Deep 5j | 6j | 7j | 8j | 9j |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed 6-layer | 805k | 6 | 63.6% | 95% | 88% | 75% | 52% | 42% |
| 6-block looped, T=2 | 839k | 12 | 63.2% | 92% | 85% | 52% | 52% | 42% |
| 6-block looped, T=1 | 839k | 6 | 62.3% | 90% | 82% | 52% | 48% | 42% |
| 2-block looped, T=8 | 309k | 16 | 59.8% | 92% | 75% | 50% | 30% | 35% |

Weight-shared looping did not help at this scale: the fixed network is as good or better everywhere, loops beyond the
trained range degrade ("overthinking"), and the stop head learned to stop after about one loop. Note: the 6-block run's
logged "blocks/move" labels say 2T (bug, fixed in `cd236ba`); the real cost is 6T. The one untested fix is progressive
training (random no-gradient warm-up loops, a wider loop range) on jump chains only.

## Browser play (phutball app)

- Models are exported with `~/Projects/phutball/scripts/export_onnx.py` (JAX-vs-ONNX parity checked: same top move on
  32/32 positions). A local copy of the app is served at `http://127.0.0.1:8765` from a scratch build (python
  `http.server`), with the self-play final (it 100) loaded and the AlphaZero provider re-enabled in the bundle.
- Bugs found and fixed in `phutball-frontend` (uncommitted; the worker files are untracked in that repo):
  - `public/alphaZeroWorker.js` and `src/workers/alphaZeroWorker.js`: PUCT scored children from the wrong side after
    turn-ending moves (fixed search beat the old one 10-0); player-2 moves were mirrored (priors in rotated coordinates
    applied to the physical board, then a second transform); a halt as the first move produced the path `[null]`
    ("e is not iterable").
  - `src/App.js`: the AI-move effect listed `declareNoWin` before its definition (render crash since the Sep 21 commit);
    the win-probability chart line is now driven by the value head through a second worker.
- The browser opponent is PUCT (c=2.5) at N playouts, greedy on visits: not the Gumbel search used in training. At 400
  playouts a move takes about 20 s (45 ms per network call in WASM).
- The owner lost one game to the it 50 model at 400 playouts (fixed search): the first real human data point.

## Elo tournament (starting now)

`elo_colab.ipynb` runs `expert/elo_tournament.py`: round robin between it50 (reference, 1000), it80, it90, it100 (final),
the 17x13 final (played on 21x15) and the best PBT member (`grpo2/best_latest.pkl`); 32x16 Gumbel search without root
noise, 64 games per colour per pair, 360-turn cap (draws = 0.5); Bradley-Terry fit with 95% bootstrap intervals. Output:
`My Drive/phutball/elo/elo.json`. The pre-self-play model is excluded on purpose (it loses every game, so it only costs
time). Expected: about 1.5-2 A100 hours.

## Grounding the owner's play

- Plan: 10 games against the self-play final in the browser at 400 playouts, 5 as P1 and 5 as P2 (P1 wins about 55% in
  self-play, so the colours must be balanced). Ten games give a rough placement (about +/-200 Elo).
- Problem: the browser search (PUCT-400) differs from the tournament's (Gumbel 32x16), so a result against the browser
  opponent is not directly on the tournament scale.
- Fix: `expert/puct_calibration.py` (last cell of `elo_colab.ipynb`) plays the same network with PUCT-400 against
  Gumbel 32/64/128x16, 10 games each (one process per budget, about 45-60 min), and converts each score to an Elo gap
  (about +/-150 with 10 games). Output: `phutball/elo/puct_vs_gumbel{32,64,128}.json`.
- To finish: rating(final with PUCT-400) = rating(it100 at 32x16) + gap(PUCT-400 vs Gumbel-32); then the owner's record
  goes in the notebook's `HUMAN` list (`"you:W-L-D vs it100"`) and the tournament fit is rerun. The `--human` option
  treats those games as games against it100 at 32x16, so apply the calibration gap to the owner's rating afterwards
  (or add a separate "it100-puct400" player).

## Open ideas, not started

Lower-lr self-play from the final; league play against past bests; a larger network through the same pipeline;
progressive-training looped model on jump chains only; a 200-per-depth deep set and multiple seeds for any looped-model
claim; a second domain (Dots and Boxes is the best fit; the owner's tile game Auger, `~/Projects/auger-puzzle`, fits the
looped-model question through its cascade depth but needs determinization for full-game search).
