"""Do the forget gate and PoPE bound the accumulator?

PRE-REGISTERED, before running. The capability synthesis claims one quantity --
how fast range(theta) grows with sequence length -- explains the OOD-length
signature of the sign and rank results. It ALSO asserts, without measurement, that
the forget gate and PoPE do not bound the accumulator and so must have some other
cause. That assertion is what this tests.

  P1 (positive control). Level15's wrapped theta_hat must have a much smaller range
      than its own theta_path. If not, the claim that a wrapped filter bounds the
      accumulator is itself unmeasured and must be withdrawn.
  P2. Forget - Vanilla, same batch: predicted NO change in alpha.
  P3. MapPoPE - Vanilla, same batch: predicted NO change in alpha.
  P4. The forget gate's own magnitude accumulator sum(log gamma): measure its
      growth. A monotone log-gamma gives a second, ballistic accumulator.

Refuted if P2 or P3 shows a materially lower alpha -- then the growth law covers all
four mechanisms and the "two causes" statement comes out of both documents.
"""
import argparse, json, os
import numpy as np, torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


@torch.no_grad()
def stats(ck, T, n, dev, seed):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = blob["config"]; v = os.path.basename(ck)[:-3]
    m = VARIANT_MAP[v](vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                       n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                       grid_size=cfg["grid_size"])
    m.load_state_dict(blob["model_state_dict"]); m = m.to(dev).eval()
    env = GridWorld(size=cfg["grid_size"], n_obs_types=cfg.get("n_obs_types", 16),
                    p_empty=cfg.get("p_empty", 0.5), seed=10000)
    np.random.seed(seed)
    om = m.path_integrator.omega.detach().abs().to(dev)
    accS = accTh = accHat = accG = None
    for _ in range(n):
        tok, _o, _r = env.generate_trajectory(T)
        tok = tok.unsqueeze(0).to(dev)
        x = m.token_emb(tok)
        S = torch.cumsum(m.action_to_lie(x), dim=1)[0]          # (T,H,nb)
        th = S * om                                             # theta_path
        rS = (S.max(0).values - S.min(0).values)
        rT = (th.max(0).values - th.min(0).values)
        accS = rS if accS is None else torch.maximum(accS, rS)
        accTh = rT if accTh is None else torch.maximum(accTh, rT)
        _ = m(tok[:, :-1])                                      # populate stashes
        if hasattr(m, "last_theta_hat"):
            h = m.last_theta_hat[0]
            rH = (h.max(0).values - h.min(0).values)
            accHat = rH if accHat is None else torch.maximum(accHat, rH)
        for mod in m.modules():
            if hasattr(mod, "last_log_gamma"):
                g = torch.cumsum(mod.last_log_gamma[0].float(), 0)
                rG = (g.max(0).values - g.min(0).values)
                accG = rG if accG is None else torch.maximum(accG, rG)
    out = dict(variant=v, T=T, range_S=float(accS.mean()), range_theta=float(accTh.mean()))
    if accHat is not None:
        out["range_theta_hat"] = float(accHat.mean())
    if accG is not None:
        out["range_loggamma"] = float(accG.mean())
    del m; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True,
                    help="label=runs_dir:Variant triples")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 1024])
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    R = {}
    for spec in a.specs:
        lab, rest = spec.split("=", 1); rd, v = rest.rsplit(":", 1)
        for s in a.seeds:
            ck = os.path.join(rd, f"{v}_s{s}", f"{v}.pt")
            if not os.path.exists(ck):
                ck = os.path.join(rd, f"seed{s}", f"{v}.pt")
            if not os.path.exists(ck):
                continue
            for T in a.lengths:
                r = stats(ck, T, a.n_trials, dev, 7 + s); r["seed"] = s
                R.setdefault(f"{lab}|{T}", []).append(r)
                print(lab, s, T, {k: round(v_, 2) for k, v_ in r.items()
                                  if isinstance(v_, float)}, flush=True)
    json.dump(R, open(a.out.replace(".md", ".json"), "w"), indent=2)

    labs = []
    for k in R:
        l = k.split("|")[0]
        if l not in labs:
            labs.append(l)
    o = ["# Do the forget gate and PoPE bound the accumulator?", "",
         "Pre-registered in `probe_accumulator.py`. Eval-only.", "",
         "`alpha` is the exponent of `range ~ T^alpha` between the two lengths;",
         "0.5 is a diffusive random walk, 1.0 is ballistic drift.", "",
         "| arm | range(S) T=128 | T=1024 | **alpha(S)** | range(theta) T=1024 | extra |",
         "|---|---|---|---|---|---|"]
    T0, T1 = a.lengths[0], a.lengths[-1]
    lr = np.log(T1 / T0)
    for l in labs:
        A, B = R.get(f"{l}|{T0}", []), R.get(f"{l}|{T1}", [])
        if not A or not B:
            continue
        s0 = np.mean([r["range_S"] for r in A]); s1 = np.mean([r["range_S"] for r in B])
        al = np.log(s1 / s0) / lr
        extra = []
        if "range_theta_hat" in B[0]:
            h0 = np.mean([r["range_theta_hat"] for r in A])
            h1 = np.mean([r["range_theta_hat"] for r in B])
            th1 = np.mean([r["range_theta"] for r in B])
            extra.append(f"theta_hat {h0:.1f}->{h1:.1f} (alpha {np.log(h1/h0)/lr:+.3f}); "
                         f"vs theta_path {th1:.1f}")
        if "range_loggamma" in B[0]:
            g0 = np.mean([r["range_loggamma"] for r in A])
            g1 = np.mean([r["range_loggamma"] for r in B])
            extra.append(f"sum log-gamma {g0:.2f}->{g1:.2f} (alpha {np.log(max(g1,1e-9)/max(g0,1e-9))/lr:+.3f})")
        o.append(f"| `{l}` | {s0:.1f} | {s1:.1f} | **{al:.3f}** | "
                 f"{np.mean([r['range_theta'] for r in B]):.1f} | {'; '.join(extra) or '—'} |")
    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
