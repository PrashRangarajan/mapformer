"""Does the OOD-length benefit localise to under-trained (low-frequency) channels?

Pre-registered in LOCALISATION_PREREG.md. Eval-only; no training.

theta_c = omega_c * cumsum(Delta). A channel is UNDER-TRAINED if it does not
complete a full cycle within the training distribution of the accumulator:
    n_cycles(c, T) = omega_c * range_T(S) / (2 pi)  <  1
The long-context literature's critical-dimension argument says such channels are
read at unseen phases when the input grows. P2 ablates them at eval time.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP


def load(ck, dev):
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg = blob["config"]; v = os.path.basename(ck)[:-3]
    m = VARIANT_MAP[v](vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                       n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                       grid_size=cfg["grid_size"])
    m.load_state_dict(blob["model_state_dict"])
    return m.to(dev).eval(), cfg


@torch.no_grad()
def accumulator_stats(m, env, T, n, dev, seed):
    """range of S = cumsum(Delta) per channel, and the cycle count."""
    np.random.seed(seed)
    lo = hi = None
    for _ in range(n):
        tok, _om, _rv = env.generate_trajectory(T)
        d = m.action_to_lie(m.token_emb(tok.unsqueeze(0).to(dev)))
        S = torch.cumsum(d, dim=1)[0]                      # (T, H, nb)
        a, b = S.min(0).values, S.max(0).values
        lo = a if lo is None else torch.minimum(lo, a)
        hi = b if hi is None else torch.maximum(hi, b)
    rng = (hi - lo)                                        # (H, nb)
    om = m.path_integrator.omega.detach().abs().to(dev)             # (H, nb)
    return rng.cpu().numpy(), (om * rng / (2 * np.pi)).cpu().numpy()


@torch.no_grad()
def evaluate(m, env, T, n, dev, seed, kill=None):
    """Held-out revisit accuracy. `kill` is a flat index array of channels whose
    phase is zeroed (that channel then applies no rotation)."""
    orig = m.path_integrator.forward
    if kill is not None and len(kill):
        H, nb = m.n_heads, m.n_blocks
        mask = torch.ones(H * nb, device=dev); mask[torch.as_tensor(kill, device=dev)] = 0.0
        mask = mask.view(1, 1, H, nb)
        om = m.path_integrator.omega.detach()
        def patched(delta):
            ang = (torch.cumsum(delta, 1) * om.unsqueeze(0).unsqueeze(0) * mask).transpose(1, 2)
            return torch.cos(ang), torch.sin(ang)
        m.path_integrator.forward = patched
    try:
        np.random.seed(seed); ok = tot = 0
        for _ in range(n):
            tok, _om, rev = env.generate_trajectory(T)
            tok = tok.unsqueeze(0).to(dev)
            lp = F.log_softmax(m(tok[:, :-1]).float(), -1)
            pred = lp.argmax(-1)[0]; tgt = tok[0, 1:]; msk = rev[1:].to(dev)
            if msk.sum() == 0:
                continue
            ok += (pred[msk] == tgt[msk]).sum().item(); tot += int(msk.sum())
    finally:
        m.path_integrator.forward = orig
    return ok / tot if tot else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(6)))
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 1024])
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    R = {"cycles": {}, "abl": {}, "base": {}}

    for v in a.variants:
        for s in a.seeds:
            ck = os.path.join(a.runs_dir, f"{v}_s{s}", f"{v}.pt")
            if not os.path.exists(ck):
                continue
            m, cfg = load(ck, dev)
            env = GridWorld(size=cfg["grid_size"], n_obs_types=cfg.get("n_obs_types", 16),
                            p_empty=cfg.get("p_empty", 0.5), seed=10000)
            om = m.path_integrator.omega.detach().abs().flatten().cpu().numpy()
            order = np.argsort(om)                      # ascending: low freq first
            rng = np.random.default_rng(1234 + s)

            for T in a.lengths:
                rg, cyc = accumulator_stats(m, env, T, 20, dev, seed=7 + s)
                R["cycles"].setdefault(f"{v}|{T}", []).append(
                    dict(seed=s, range=float(rg.mean()),
                         cycles=cyc.flatten().tolist(),
                         under=int((cyc.flatten() < 1).sum())))
                base = evaluate(m, env, T, a.n_trials, dev, 4242 + s)
                R["base"].setdefault(f"{v}|{T}", []).append((s, base))
                for k in a.ks:
                    for lab, idx in (("low", order[:k]), ("high", order[-k:]),
                                     ("rand", rng.choice(len(om), k, replace=False))):
                        acc = evaluate(m, env, T, a.n_trials, dev, 4242 + s, kill=idx)
                        R["abl"].setdefault(f"{v}|{T}|{lab}|{k}", []).append((s, acc - base))
                print(f"{v} s{s} T={T} base {base:.4f} under {int((cyc.flatten()<1).sum())}/{len(om)}",
                      flush=True)
            del m; torch.cuda.empty_cache()

    json.dump(R, open(a.out.replace(".md", ".json"), "w"), indent=2)

    def st(x):
        x = np.asarray([v for v in x if v is not None], float)
        if len(x) < 2:
            return dict(m=float("nan"), sd=float("nan"), mde=float("nan"), n=len(x))
        sd = x.std(ddof=1)
        return dict(m=float(x.mean()), sd=float(sd), mde=float(2.8*sd/np.sqrt(len(x))), n=len(x))

    o = ["# Does the OOD benefit localise to under-trained channels?", "",
         "Pre-registered in `LOCALISATION_PREREG.md`. Eval-only.", "",
         "## P1 — are any channels under-trained? (`n_cycles < 1`)", "",
         "| arm | T | mean range(S) | under-trained channels |", "|---|---|---|---|"]
    for k, rows in sorted(R["cycles"].items()):
        v, T = k.split("|")
        o.append(f"| `{v}` | {T} | {np.mean([r['range'] for r in rows]):.2f} | "
                 f"{np.mean([r['under'] for r in rows]):.1f} / 64 |")
    o += ["", "## P2 — ablating channels, accuracy change vs unablated", "",
          "Negative = ablation hurts. **Predicted: `low` hurts LESS at T=1024 than at "
          "T=128, and less than `high`.**", "",
          "| arm | T | k | low | high | rand |", "|---|---|---|---|---|---|"]
    for v in a.variants:
        for T in a.lengths:
            for k in a.ks:
                cells = []
                for lab in ("low", "high", "rand"):
                    d = [x[1] for x in R["abl"].get(f"{v}|{T}|{lab}|{k}", [])]
                    s_ = st(d)
                    cells.append("—" if not np.isfinite(s_["m"]) else
                                 f"{s_['m']:+.3f} ± {s_['sd']:.3f}")
                if cells[0] != "—":
                    o.append(f"| `{v}` | {T} | {k} | " + " | ".join(cells) + " |")
    o += ["", "## Baselines", "", "| arm | " + " | ".join(f"T={T}" for T in a.lengths) + " |",
          "|---" * (len(a.lengths) + 1) + "|"]
    for v in a.variants:
        cs = []
        for T in a.lengths:
            r = [x[1] for x in R["base"].get(f"{v}|{T}", []) if x[1] is not None]
            cs.append(f"{np.mean(r):.3f} ± {np.std(r, ddof=1):.3f}" if len(r) > 1 else "—")
        o.append(f"| `{v}` | " + " | ".join(cs) + " |")
    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
