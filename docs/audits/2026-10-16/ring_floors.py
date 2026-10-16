"""Map-free floors on the ring (environment_cancel.GridWorldCancel, 32 cells, T = 128, map redrawn per walk), scored on
revisit targets as train_cancel.evaluate: always-blank, and the one-level retrace rule of
docs/audits/2026-10-05/score_rank_floors.py (while a run of actions reverses the previous run, step j copies the
observation 2j tokens back; else predict blank). 2000 walks per p_plus, np seed 4242 (any stream; a floor, not a pairing)."""
import sys
import numpy as np
sys.path.insert(0, "/home/prashr")
from mapformer.environment_cancel import GridWorldCancel  # noqa: E402

for pp in (0.5, 0.75, 0.9):
    env = GridWorldCancel(size=32, p_plus=pp); BL = env.blank_token + env.obs_offset
    np.random.seed(4242); n = blank = ret = back = 0
    for _ in range(2000):
        tok, _om, rev = env.generate_trajectory(128)
        tok = tok.numpy(); a = tok[0::2]; o = tok[1::2]; r = rev[1::2].numpy()
        prev_run = cur = 0
        for t in range(len(a)):
            if t > 0 and a[t] == a[t - 1]:
                cur += 1
            else:
                prev_run = cur if t > 0 else 0          # two actions only: any change reverses
                cur = 1
            if r[t]:
                n += 1; blank += o[t] == BL
                pr = o[t - 2 * cur] if (prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0) else BL
                ret += pr == o[t]; back += prev_run > 0 and cur <= prev_run
    print(f"p_plus {pp}: revisits {n}  blank {blank / n:.4f}  retrace {ret / n:.4f}  (share of revisits inside a retrace {back / n:.3f})")
