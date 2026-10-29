"""MODEL-FREE, CPU (Opus 2026-10-29). Bounded-room SURROGATE for MiniWorld OneRoom (4 moves, runs of 1-10 stopped by walls,
fresh obs map per walk: 16 types, p_empty 0.5) and the 64-torus for reference. Floors on revisits: blank, retrace, in-context
longest-suffix match ('induction', windows of 2..24 tokens, most recent continuation), and their combination
(retrace if it applies, else induction if a match of >= 6 tokens, else blank). Question: does a sequence-recognition
(induction) floor rise steeply as the map shrinks, where the cover (tree/plane) shares do not?"""
import sys
import numpy as np
sys.path.insert(0, "/home/prashr")
from mapformer.environment import GridWorld  # noqa
sys.path.insert(0, "/home/prashr/mapformer/docs/research/scripts_opus")
from q1_regime_sweep import room_walk, INV, DELTA  # noqa
BLANK = 20; MAXM = 24


def floors(acts, cells, obs_of):
    o = np.array([obs_of(c) for c in cells]); tok = np.empty(2 * len(acts), int); tok[0::2] = acts; tok[1::2] = o
    dicts = [dict() for _ in range(MAXM + 1)]; seen = set(); prev_run = cur = 0; res = []
    for t in range(len(acts)):
        a = acts[t]
        if t > 0 and a == acts[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a == INV[acts[t - 1]]) else 0
            cur = 1
        i = 2 * t + 1; im, ip = 0, BLANK
        for m in range(min(MAXM, i), 1, -1):
            h = dicts[m].get(tok[i - m:i].tobytes())
            if h is not None:
                im, ip = m, h; break
        if cells[t] in seen:
            rt_app = prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0
            rtp = o[t - 2 * cur] if rt_app else BLANK
            comb = rtp if rt_app else (ip if im >= 6 else BLANK)
            res.append((o[t] == BLANK, rtp == o[t], (ip if im >= 2 else BLANK) == o[t], comb == o[t]))
        seen.add(cells[t])
        for m in range(2, min(MAXM, i) + 1):
            dicts[m][tok[i - m:i].tobytes()] = tok[i]
    return res


rng = np.random.default_rng(1); N = 40
print("regime | blank retrace induction comb(retrace->induction>=6)")
for L, T in ((6, 512), (8, 512), (11, 512), (16, 512), (23, 512), (32, 512), (16, 1024), (32, 1024)):
    R = []
    for _ in range(N):
        om = np.where(rng.random((L, L)) < 0.5, BLANK, rng.integers(4, 20, (L, L)))
        a, c = room_walk(L, T, rng)
        R += floors(a, c, lambda x: om[x])
    R = np.array(R, float).mean(0)
    print(f"room {L}x{L} ({L*L//2} occ.) T {T}: blank {R[0]:.3f} retrace {R[1]:.3f} induction {R[2]:.3f} comb {R[3]:.3f}")
for size, T in ((64, 128), (64, 1024)):
    env = GridWorld(size=size, n_obs_types=16, p_empty=0.5, seed=10000); np.random.seed(1); R = []
    for _ in range(N):
        tok, _o, _r = env.generate_trajectory(T); a = tok.numpy()[0::2]; U = np.cumsum(DELTA[a], 0) % size
        om = env.obs_map.numpy() + env.obs_offset
        R += floors(a, [tuple(u) for u in U], lambda x: om[x])
    R = np.array(R, float).mean(0)
    print(f"torus {size} T {T}: blank {R[0]:.3f} retrace {R[1]:.3f} induction {R[2]:.3f} comb {R[3]:.3f}")
