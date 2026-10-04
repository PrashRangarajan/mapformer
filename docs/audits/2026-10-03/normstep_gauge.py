"""NormStep's observation step = shared part c_bar + object-dependent part delta(object) (2026-10-03).
Per trained model (runs/leak/p0, seeds 0 and 3): |c_bar| / |action step|, the object-identity spread |delta| /
|action step|, and whether opposite actions cancel alone vs once the two interleaved observations' shared step is
added (the per-move gauge: with exactly one action and one observation per move, c_bar can be absorbed into the
actions). omega-scaled steps; c_bar = 0.5 * blank + 0.5 * mean over 200 test-pool objects."""
import torch
from mapformer.leak_eval import load

for arm in ("NormStep", "MapWM"):
    for s in (0, 3):
        m, _ = load(f"/home/prashr/mapformer/runs/leak/p0/{arm}_s{s}/{arm}.pt", arm, "cpu")
        toks = torch.tensor([[0, 1, 2, 3, 4] + list(range(1005, 1205))])
        with torch.no_grad():
            x = m.token_emb(toks); d = (m.step(toks, x) if hasattr(m, "step") else m.action_to_lie(x))[0].flatten(1)
            d = d * m.path_integrator.omega.reshape(-1)
        a = {k: d[k] for k in range(4)}; blank = d[4]; obj = d[5:]
        c_obs = 0.5 * blank + 0.5 * obj.mean(0); nrm = (a[0].norm() + a[1].norm()) / 2
        raw = ((a[0] + a[1]).norm() / nrm, (a[2] + a[3]).norm() / nrm)
        corr = ((a[0] + a[1] + 2 * c_obs).norm() / nrm, (a[2] + a[3] + 2 * c_obs).norm() / nrm)
        spread = (obj - obj.mean(0)).norm(dim=1).mean() / nrm
        print(f"{arm:8s} s{s}: shared obs step {c_obs.norm() / nrm:.4f} | object-identity spread {spread:.4f} | "
              f"cancel raw N+S {raw[0]:.3f} W+E {raw[1]:.3f} | incl. 2 obs steps {corr[0]:.3f} {corr[1]:.3f}")
