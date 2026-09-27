"""Decode every policy target back to a physical action and check it equals the expert's action;
check the encoded ball channel sits where the (rotated) ball should be."""
import sys; sys.path.insert(0, ".")
import numpy as np
from expert.dataset import play_record, to_examples

games = [play_record(s, 21, 15, 0.1, 200) for s in range(4)]
S, P, V = to_examples(games)
rows, cols = 21, 15; n = rows * cols; k = 0; bad = 0
for rec, w in games:
    for s, a in rec:
        v = int(P[k].argmax())
        phys = v if v == 2 * n else ((n - 1 - v) if v < n else (n + (n - 1 - (v - n)))) if s.player == 2 else v
        bad += phys != a
        br, bc = divmod(s.ball, cols)
        if s.player == 2: br, bc = rows - 1 - br, cols - 1 - bc
        bad += S[k][0, br, bc] != 1.0
        k += 1
print(f"{k} examples checked, {bad} mismatches; channels {S.shape[1]}; values in {sorted(set(V.tolist()))}")
