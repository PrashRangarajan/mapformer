"""Fix-14 R2 (post hoc, CPU, eval-only; PREDICTIONS.md): does the one-channel flow (THEORY_REVIEW potential.py, Remark R's
~28% capture from U(1.7, 4.5)) predict WHICH channels of the trained AO_high (and SO_high) models end cancelling?
For each of the 64 channels x 8 runs (MECHANISM, seeds 800-807), the flow is run from that channel's own initial move
angles (eval.json init_angles), |.| for AO_high and signed for SO_high, with potential.py's settings (pair_hist n = 2000,
plain gradient ascent on V, lr 0.005, 4000 steps). Predicted cancelling: |wrap(a+ + a-)| < 0.1 at the flow's end; observed
cancelling: the same on the trained checkpoint's angles. Channels counted where the trained |wrap(a(+1))| > 0.05 (the
registered selection). The flow is potential.flow itself, called per channel (a batched re-implementation agreed to
1e-15 over 20 steps but drifted by up to 2e-3 over 400 steps near the |.| fold, where chattering amplifies rounding, so it
is not used). Check first: potential.flow on constructed starts (a channel at the wrap stays cancelled; a channel at the
cancelling line of the signed class stays there).
  cd /home/prashr && OMP_NUM_THREADS=4 PYTHONPATH=/home/prashr python3 \
      mapformer/docs/audits/2026-10-15/reanalyses/capture_by_channel.py"""
import os, sys

import numpy as np

sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-14/theory_review")
import potential as P                                                       # noqa: E402
from mapformer.analyze_wrap_mech import load                                # noqa: E402
from mapformer.model_wrap_exist import action_angles                        # noqa: E402

R = "/home/prashr/mapformer/runs/wrap_mech/p0"; SEEDS = range(800, 808); HERE = os.path.dirname(os.path.abspath(__file__))
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def flow_batch(W, A0, mode, steps=4000, lr=0.005):
    """potential.flow (plain gradient ascent on V) for N channels at once. A0: (N, 2)."""
    m = np.arange(W.shape[0]); s = np.array(A0, float)
    for _ in range(steps):
        a = np.abs(s) if mode == "abs" else s
        ep = np.exp(1j * a[:, :1] * m); em = np.exp(1j * a[:, 1:] * m)
        gp = (((1j * m * ep) @ W) * em).sum(1).real; gm = ((ep @ W) * (1j * m * em)).sum(1).real
        g = np.stack([gp, gm], 1)
        if mode == "abs":
            g = g * np.where(s >= 0, 1.0, -1.0)
        s = s + lr * g
    return np.abs(s) if mode == "abs" else s


def kappa(p, o):
    p = p.astype(bool); o = o.astype(bool); n = len(p); po = (p == o).mean()
    pe = p.mean() * o.mean() + (1 - p.mean()) * (1 - o.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    Ws, Wa, _ = P.pair_hist()                                               # n = 2000, seed 0 (potential.py's main)
    W = Ws - Wa
    rng = np.random.RandomState(1); T0 = rng.uniform(0, 5, (12, 2))
    for mode in ("abs", "signed"):                                           # the batched version, short horizon only
        ref = np.array([P.flow(W, a, mode, steps=20) for a in T0]); bat = flow_batch(W, T0, mode, steps=20)
        print(f"CHECK batched vs potential.flow ({mode}, 12 channels, 20 steps): max |diff| {np.abs(ref - bat).max():.2e}")
    ok = True                                                                # constructed starts for the predicate
    e = P.flow(W, (2.0, 2 * np.pi - 2.0), "abs"); ok &= abs(wrap(e.sum())) < 0.1 and abs(wrap(e[0])) > 0.05
    e = P.flow(W, (1.3, -1.3), "signed"); ok &= abs(wrap(e.sum())) < 0.1 and abs(wrap(e[0])) > 0.05
    e = P.flow(W, (0.4, 0.5), "abs"); ok &= not (abs(wrap(e.sum())) < 0.1 and abs(wrap(e[0])) > 0.05)   # small same-sign start
    print("CHECK constructed starts (wrap held, through-zero held, small |.| start not captured):", "PASS" if ok else "FAIL")
    if not ok:
        sys.exit(1)
    lines = []
    for arm, mode in (("AO_high", "abs"), ("SO_high", "signed")):
        P_, O_, D_, S_ = [], [], [], []
        for s in SEEDS:
            r = load(R, arm, s); A0 = np.array(r["ev"]["init_angles"]).T            # (C, 2)
            A1 = action_angles(r["model"]).double().numpy().T                         # (C, 2)
            sel = np.abs(wrap(A1[:, 0])) > 0.05
            E = np.array([P.flow(W, a, mode) for a in A0])
            pred = (np.abs(wrap(E.sum(1))) < 0.1) & (np.abs(wrap(E[:, 0])) > 0.05)
            obs = np.abs(wrap(A1.sum(1))) < 0.1
            P_.append(pred[sel]); O_.append(obs[sel]); D_.append(np.abs(A0.sum(1) - 2 * np.pi)[sel]); S_.append(np.full(sel.sum(), s))
        p, o, d, sd = map(np.concatenate, (P_, O_, D_, S_))
        tab = [[int((p & o).sum()), int((p & ~o).sum())], [int((~p & o).sum()), int((~p & ~o).sum())]]
        per = [kappa(p[sd == s], o[sd == s]) for s in SEEDS]
        lines.append(f"{arm} ({mode} flow), {len(p)} channels over 8 runs: predicted cancelling {p.mean():.3f}, observed {o.mean():.3f}")
        lines.append(f"  2x2 [pred & obs, pred & not obs; not pred & obs, neither] = {tab}; agreement {(p == o).mean():.3f}; "
                     f"Cohen's kappa {kappa(p, o):+.3f} (per run {' '.join(f'{k:+.2f}' for k in per)})")
        lines.append(f"  P(observed | predicted) {o[p].mean() if p.any() else float('nan'):.3f}; P(observed | not predicted) "
                     f"{o[~p].mean():.3f}; share of observed cancelling channels that the flow predicts {p[o].mean():.3f}")
        lines.append(f"  observed cancelling among channels with |S0 - 2 pi| < 0.5: {o[d < 0.5].mean():.3f} (n {int((d < 0.5).sum())}); "
                     f">= 0.5: {o[d >= 0.5].mean():.3f} (n {int((d >= 0.5).sum())}); predicted in the same bins "
                     f"{p[d < 0.5].mean():.3f} / {p[d >= 0.5].mean():.3f}")
    txt = "\n".join(lines); print(txt); open(f"{HERE}/capture_by_channel_out.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
