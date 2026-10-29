"""POST HOC, CPU (Opus 2026-10-29). Is Selective RoPE's wrap-only deficit a torus-commensurability failure?
Per trained SROPE_TORUS model: mean angle per move per channel (pair increments: theta at observation k minus at
observation k-1, keyed by action k; the conv's lag is absorbed), split into map part m = (a(+) - a(-))/2 and clock part
k = a(+) + a(-) per axis. Predicted coherence of a wrap-only revisit (displacement 64 along one axis, zero along the other)
in channel c: cos(64 m_c) (wrap-aware); of a zero-displacement out-and-back of n moves per direction: cos(n k_c).
Usage-weighted (|q_c| at moves x |k_c| at observations) means, per seed, against the measured wrap-only accuracy.
For SRoPE arms: usage share on channels whose fixed temperature is < 0.01 (they cannot reach 2 pi / 64 per move with
steps of order 1)."""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/research/scripts_opus")
sys.path.insert(0, "/home/prashr")
from q2_srope_decomp import load, angles, usage, ARMS  # noqa: E402
from mapformer.environment import GridWorld  # noqa: E402
import json  # noqa: E402

REPO = "/home/prashr/mapformer"
SEEDS = range(1700, 1708)
torch.set_num_threads(8)
strata = {}
for arm, V in ARMS.items():
    st = json.load(open(f"{REPO}/SROPE_TORUS_{arm}_STRATA.json"))
    strata[arm] = {s: st[f"{V}|{s}|1024"]["wrap"]["acc"] for s in SEEDS}
env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
lines = []
P = lambda *x: (print(*x), lines.append(" ".join(str(y) for y in x)))
for arm in ARMS:
    rows = []
    for s in SEEDS:
        m = load(arm, s); np.random.seed(1234 + s)
        tok = torch.stack([env.generate_trajectory(1024)[0] for _ in range(4)])[:, :-1]
        if hasattr(m, "path_integrator"):                           # unwrapped (angles() returns atan2-wrapped for MapWM)
            with torch.no_grad():
                th = torch.cumsum(m.action_to_lie(m.token_emb(tok)), 1) * m.path_integrator.omega
        else:
            th = angles(m, tok).transpose(1, 2)                     # (B, L, H, nb)
        U = usage(m, tok).numpy()                                   # (H, nb)
        ob = th[:, 1::2]; pinc = (ob[:, 1:] - ob[:, :-1]).reshape(-1, *th.shape[2:]).numpy()
        act = tok[:, 0::2][:, 1:ob.shape[1]].reshape(-1).numpy()
        A = np.stack([pinc[act == a].mean(0) for a in range(4)])    # (4, H, nb): N, S, W, E
        wrap = lambda x: np.arctan2(np.sin(x), np.cos(x))
        mx, my = (A[1] - A[0]) / 2, (A[3] - A[2]) / 2
        kx, ky = A[0] + A[1], A[2] + A[3]
        w = U / U.sum()
        cw = 0.5 * ((w * np.cos(64 * mx)).sum() + (w * np.cos(64 * my)).sum())
        ck = 0.5 * ((w * np.cos(8 * kx)).sum() + (w * np.cos(8 * ky)).sum())
        res = dict(seed=s, wrap_pred=cw, clock8=ck, wrap_acc=strata[arm][s])
        if hasattr(m, "srope"):
            lowT = (m.srope[0].temperature.numpy() < 0.01)
            res["lowT_usage"] = float(w[:, lowT].sum())
            res["map_usage_hiT"] = float((w * (np.abs(wrap(mx)) + np.abs(wrap(my)) > 0.02))[:, ~lowT].sum())
        rows.append(res)
    wp = np.array([r["wrap_pred"] for r in rows]); wa = np.array([r["wrap_acc"] for r in rows])
    extra = ""
    if "lowT_usage" in rows[0]:
        extra = f"; usage on temperature < 0.01 channels {np.mean([r['lowT_usage'] for r in rows]):.3f}"
    r_ = np.corrcoef(wp, wa)[0, 1] if wp.std() > 1e-9 and wa.std() > 1e-9 else float("nan")
    P(f"{arm:13s} predicted wrap coherence (usage-wtd cos 64 m) {wp.mean():.3f} [{wp.min():.3f}, {wp.max():.3f}]; "
      f"out-and-back x8 coherence (cos 8 k) {np.mean([r['clock8'] for r in rows]):.3f}; wrap-only acc {wa.mean():.3f}; "
      f"r(pred, acc) over seeds {r_:+.2f}{extra}")
    P("   per seed: " + " ".join(f"{r['seed']}:{r['wrap_pred']:.2f}/{r['wrap_acc']:.2f}" for r in rows))
open(f"{REPO}/docs/research/scripts_opus/out/q2_srope_wrap_out.txt", "w").write("\n".join(lines) + "\n")
