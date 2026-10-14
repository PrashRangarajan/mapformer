"""POST HOC (CPU, after WRAP_EXISTENCE's registered analysis): which channels stop cancelling during the continuation?
For SO_bal and AO_bal (seeds 600-607), at construction vs after 300 epochs: per band of |wrap(angle(+1))| the number of
channels and the share whose |wrap(angle(+1) + angle(-1))| < 0.1 rad; the share of total |angle(+1)| carried by cancelling
channels; the cancelling share among fast channels (|omega| > 1)."""
import numpy as np, torch
from mapformer.analyze_wrap_existence import load
from mapformer.train_wrap_exist import load_so, model_kw
from mapformer.model_wrap_exist import wrap_construct, action_angles
from mapformer.environment_cancel import GridWorldCancel
R = "/home/prashr/mapformer/runs/wrap_existence/p0"


def stats(m):
    A = action_angles(m).double(); w = torch.angle(torch.exp(1j * (A[0] + A[1]))).abs()
    a = torch.angle(torch.exp(1j * A[0])).abs(); om = m.path_integrator.omega.detach().reshape(-1).double().abs()
    out = {}
    for lo, hi in ((0.05, 0.5), (0.5, 4)):
        sel = (a > lo) & (a <= hi)
        out[f"{lo}-{hi}"] = (int(sel.sum()), float((w[sel] < 0.1).double().mean()) if sel.any() else float("nan"))
    out["mass_cancel"] = float((a * (w < 0.1)).sum() / a.sum())
    out["fast_share"] = float((w[(om > 1) & (a > 0.05)] < 0.1).double().mean())
    return out


for arm, mode in (("SO_bal", "signed_balanced"), ("AO_bal", "balanced")):
    rows = []
    for s in range(600, 608):
        env = GridWorldCancel(size=32, seed=s, p_plus=0.5); kw = model_kw(env, 32)
        m0, _ = wrap_construct(load_so(f"{R}/SO_s{s}/SO.pt", env, 32), kw.pop("vocab_size"), kw.pop("action_ids"), mode=mode, **kw)
        rows.append((stats(m0), stats(load(R, arm, s)["model"])))
    for k in ("0.05-0.5", "0.5-4"):
        print(f"{arm} channels |angle(+1)| in {k}: n {np.mean([r[0][k][0] for r in rows]):.1f} -> {np.mean([r[1][k][0] for r in rows]):.1f}; "
              f"cancel share {np.nanmean([r[0][k][1] for r in rows]):.2f} -> {np.nanmean([r[1][k][1] for r in rows]):.2f}")
    print(f"{arm} angle-mass in cancelling channels {np.mean([r[0]['mass_cancel'] for r in rows]):.2f} -> "
          f"{np.mean([r[1]['mass_cancel'] for r in rows]):.2f}; fast (|omega|>1) channel share "
          f"{np.mean([r[0]['fast_share'] for r in rows]):.2f} -> {np.mean([r[1]['fast_share'] for r in rows]):.2f}")
