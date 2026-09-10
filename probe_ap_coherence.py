"""Coherence of the EM position kernel at init, and its relation to final loss.

A_P[t,s] = sum_i a_i cos(omega_i * dS + phi_i) =: kappa(dS), with a_i=|q0_i||k0_i|
and phi_i the per-block angle between q0_i and k0_i. The coherence

    rho = kappa(0) / sum_i a_i = (sum_i a_i cos phi_i) / sum_i a_i

is 1 exactly when q0 == k0 (every block peaks at dS=0 together), and has
E[rho]=0, sd ~ sqrt(sum a^2/2)/sum a when q0, k0 are drawn independently.
THEORY_KERNEL.md Thm 2. N4 tests whether sign(rho) predicts the seed's fate.

Absolute REPO constant: `python3 -m mapformer.X` runs from the PARENT dir.
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/prashr/mapformer")


def coherence(q0, k0, n_heads):
    """rho per head. q0,k0 are [n_heads, d_head]."""
    H = n_heads
    qa = q0.view(H, -1, 2).double()
    ka = k0.view(H, -1, 2).double()
    a = qa.norm(dim=-1) * ka.norm(dim=-1)
    cosphi = (qa * ka).sum(-1) / a.clamp_min(1e-30)
    rho = (a * cosphi).sum(-1) / a.sum(-1)
    sd_pred = (a.pow(2).sum(-1) / 2).sqrt() / a.sum(-1)
    return rho.numpy(), sd_pred.numpy()


def from_checkpoint(pt):
    """Two trainers, two key names: train_recency writes `model_state`,
    train_variant writes `model_state_dict`. `config` is present only in the
    latter; when absent the caller's --d-model/--n-layers/--vocab apply."""
    d = torch.load(pt, map_location="cpu", weights_only=False)
    sd = d.get("model_state", d.get("model_state_dict"))
    if sd is None:
        raise KeyError(f"no state dict in {pt}: keys {list(d)}")
    return sd, d.get("losses"), d.get("config", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(REPO / "runs/recency_em"))
    ap.add_argument("--arm", default="VanillaEM_r4")
    ap.add_argument("--suffix", default="_recency.pt")
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=89)
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--env-name", default=None,
                    help="MiniGrid env to construct before the model, so the RNG "
                         "state matches the trainer. Verified not to advance "
                         "torch's RNG, so it is belt-and-braces.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for d in sorted(Path(a.runs_dir).glob(f"{a.arm}_s*")):
        s = int(re.search(r"_s(\d+)$", d.name).group(1))
        pt = d / f"{a.arm}{a.suffix}"
        if not pt.exists():
            print(f"missing {pt}")
            continue
        sd, losses, cfg = from_checkpoint(pt)
        d_model = cfg.get("d_model", a.d_model)
        n_layers = cfg.get("n_layers", a.n_layers)
        n_heads = cfg.get("n_heads", a.n_heads)
        vocab = cfg.get("vocab_size", a.vocab)
        grid = cfg.get("grid_size", a.grid_size)
        # rho at INIT is not stored, so recompute it from the construction seed.
        torch.manual_seed(s); np.random.seed(s)
        from mapformer.train_variant import VARIANT_MAP
        if a.env_name:
            from mapformer.minigrid_env import MiniGridWorld_Cached
            MiniGridWorld_Cached(env_name=a.env_name, seed=s)
        m = VARIANT_MAP[a.arm](vocab_size=vocab, d_model=d_model,
                               n_heads=n_heads, n_layers=n_layers,
                               grid_size=grid)
        if hasattr(m, "q0_pos"):
            r0, sp = coherence(m.q0_pos.detach(), m.k0_pos.detach(), n_heads)
        else:                                   # single p_0 -> rho == 1 by construction
            p = m.p0_pos.detach()
            r0, sp = coherence(p, p, n_heads)
        # and rho AFTER training, from the checkpoint
        if "q0_pos" in sd:
            r1, _ = coherence(sd["q0_pos"], sd["k0_pos"], n_heads)
        else:
            r1, _ = coherence(sd["p0_pos"], sd["p0_pos"], n_heads)
        rows.append(dict(seed=s, rho_init=r0.tolist(), rho_final=r1.tolist(),
                         rho_init_min=float(r0.min()), rho_init_mean=float(r0.mean()),
                         rho_final_mean=float(r1.mean()),
                         sd_pred=float(sp.mean()),
                         final_loss=float(losses[-1]) if losses else None))

    if not rows:
        print("no rows"); return
    ri = np.array([r["rho_init_mean"] for r in rows])
    rf = np.array([r["rho_final_mean"] for r in rows])
    fl = np.array([r["final_loss"] for r in rows])
    allinit = np.array([v for r in rows for v in r["rho_init"]])

    print(f"arm {a.arm}, n={len(rows)} seeds, {allinit.size} head-values")
    print(f"  rho at init : mean {allinit.mean():+.4f}  sd {allinit.std(ddof=1):.4f}"
          f"   theory: E=0, sd={rows[0]['sd_pred']:.4f}")
    print(f"  rho final   : mean {rf.mean():+.4f}  sd {rf.std(ddof=1):.4f}")
    print()
    print(f"{'seed':>4} {'rho_init(mean)':>15} {'rho_init(min)':>14} {'rho_final':>10} {'final_loss':>11}")
    for r in rows:
        print(f"{r['seed']:>4} {r['rho_init_mean']:>+15.4f} {r['rho_init_min']:>+14.4f}"
              f" {r['rho_final_mean']:>+10.4f} {r['final_loss']:>11.4f}")
    if len(rows) > 2:
        print()
        print(f"  N4: r(rho_init, final_loss)  = {np.corrcoef(ri, fl)[0,1]:+.3f}  (n={len(rows)})")
        print(f"      r(rho_final, final_loss) = {np.corrcoef(rf, fl)[0,1]:+.3f}")
        print(f"      drift |rho_final - rho_init| mean = {np.abs(rf-ri).mean():.4f}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
