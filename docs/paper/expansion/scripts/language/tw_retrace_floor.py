"""The paper's retrace floor on the text world (POST HOC, CPU). The text world's registered floors were the best constant
(0.505) and a one-step reversal-copy rule (0.597); the main paper's torus floor is RETRACE (sign_floors.py: while a run of
moves reverses the previous run, the j-th step copies the observation 2j steps back; blank otherwise), which scores
0.843 on the 1024-move torus. Here it is computed on the text world's registered eval stream (held-out map 10000, np
seed 0, 200 walks x 1024 words = TEXTWORLD_SECONDARY S5) and on TW_NORMSTEP's (np seed 10**6, 200 walks)."""
import sys, numpy as np
sys.path.insert(0, '/home/prashr/mapformer/docs/paper/expansion/scripts/language')
from tw_lib import TextWorld, ROLES, gen
for tag, seed in (("TEXTWORLD (np seed 0)", 0), ("TW_NORMSTEP (np seed 10**6)", 10**6)):
    te = TextWorld(size=64, seed=10000); np.random.seed(seed)
    n = blank = ret = rc = 0
    nothing = te.idx["nothing"]
    for _ in range(200):
        g = gen(te, 1024)
        slots = np.nonzero(g["obs"])[0]                       # full clauses only
        a = g["dirs"][:len(slots)]; o = g["tok"][slots]; r = g["rev"][slots]
        prev_run = cur = 0
        for t in range(len(slots)):
            if t > 0 and a[t] == a[t - 1]:
                cur += 1
            else:
                prev_run = cur if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
                cur = 1
            if r[t]:
                n += 1; blank += o[t] == nothing
                pr = nothing
                if prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0:
                    pr = o[t - 2 * cur]
                ret += pr == o[t]
                rcp = o[t - 2] if (t >= 2 and a[t] == (a[t - 1] ^ 1)) else nothing
                rc += rcp == o[t]
    print(f"{tag}: n {n} | always-'nothing' {blank / n:.4f} | one-step reversal-copy {rc / n:.4f} | retrace {ret / n:.4f}")
