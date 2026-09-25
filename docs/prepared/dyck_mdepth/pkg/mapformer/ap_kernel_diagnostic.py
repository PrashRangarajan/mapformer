"""Is EM's deficit explained by the geometry of its position kernel A_P?

Hypothesis under test. The paper's main text defines a single origin, giving
A_P = P.P^T with P = R_theta P* -- an AUTOCORRELATION kernel, necessarily
maximal and positive at zero displacement. Appendix A.4 says the paper actually
uses two vectors k0p/q0p and *suspects* the separation helps ("it would create
sparser attention values"). With two vectors,

    A_P[i,j] = sum_c |q0_c||k0_c| cos(dtheta_c + psi_c)

where psi_c is the phase offset between them, so the peak is DISPLACED and the
same-cell value can be negative -- the model scores "same place" as a mismatch.

PREDICTION: the same-cell kernel quality should track where EM underperforms.
Separate-q0/k0 should look bad wherever EM loses (n_obs=256, compositional) and
single-p_0 should not. If kernel quality is equally bad in regimes where EM WINS
(n_obs=16), the parameterization story is wrong and the deficit is something else.

Measured on REVISIT PAIRS (i,j), i>j, where the agent was at the same grid cell
-- true zero displacement, not the trivial diagonal:
  same_cell_mean   mean A_P at those pairs (want: high, positive)
  frac_negative    fraction scored NEGATIVE (want: 0)
  frac_rowmax      fraction that are the causal row argmax (want: high)
"""
import argparse
import json
from pathlib import Path

import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.model import _apply_rope

_REPO = Path(__file__).resolve().parent


@torch.no_grad()
def kernel_stats(model, env, n_traj, n_steps, device):
    """A_P statistics at same-cell (zero-displacement) revisit pairs."""
    tot = neg = rowmax = 0
    ssum = 0.0
    for _ in range(n_traj):
        tokens, _om, _rm, locs = env.generate_batch(1, n_steps)
        tokens = tokens.to(device)
        x = model.token_emb(tokens)
        cos_a, sin_a = model.path_integrator(model.action_to_lie(x))
        B, L = tokens.shape

        if hasattr(model, "p0_pos"):          # single-p_0 ablation
            q0 = k0 = model.p0_pos
        else:                                  # paper-faithful separate vectors
            q0, k0 = model.q0_pos, model.k0_pos
        qp = _apply_rope(q0.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1), cos_a, sin_a)
        kp = _apply_rope(k0.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1), cos_a, sin_a)
        A_P = torch.matmul(qp, kp.transpose(-1, -2)).mean(1)[0]   # avg heads -> (L, L)

        # observation token at step t sits at input index 2t+1 -> location[t]
        loc = locs[0]
        by_cell = {}
        for t, c in enumerate(loc):
            by_cell.setdefault(tuple(c), []).append(2 * t + 1)
        for idxs in by_cell.values():
            for a in range(1, len(idxs)):
                i, j = idxs[a], idxs[a - 1]
                if i >= L or j >= L:
                    continue
                v = A_P[i, j]
                ssum += v.item(); tot += 1
                neg += int(v.item() < 0)
                rowmax += int(A_P[i, :i].argmax().item() == j)
    return dict(n_pairs=tot,
                same_cell_mean=ssum / max(tot, 1),
                frac_negative=neg / max(tot, 1),
                frac_rowmax=rowmax / max(tot, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True,
                    help="label=path/to/ckpt.pt:n_obs")
    ap.add_argument("--n-traj", type=int, default=40)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--env-seed", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "AP_KERNEL_DIAGNOSTIC.md"))
    args = ap.parse_args()

    dev = torch.device(args.device)
    rows = []
    for spec in args.specs:
        label, rest = spec.split("=", 1)
        path, nobs = rest.rsplit(":", 1)
        p = Path(path)
        if not p.exists():
            print(f"MISSING {p} -- skipping")
            continue
        blob = torch.load(p, map_location="cpu", weights_only=False)
        sd, cfg = blob["model_state_dict"], blob["config"]
        variant = blob.get("variant", "VanillaEM")
        model = VARIANT_MAP[variant](
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            grid_size=cfg["grid_size"]).to(dev)
        model.load_state_dict(sd); model.eval()
        env = GridWorld(size=cfg["grid_size"], n_obs_types=int(nobs),
                        p_empty=cfg["p_empty"], n_landmarks=0, seed=args.env_seed)
        st = kernel_stats(model, env, args.n_traj, args.n_steps, dev)
        st["label"] = label
        rows.append(st)
        print(f"{label:34s} mean={st['same_cell_mean']:+.4f} "
              f"neg={st['frac_negative']:.1%} rowmax={st['frac_rowmax']:.1%} "
              f"(n={st['n_pairs']})", flush=True)

    lines = ["# A_P kernel geometry at zero displacement", "",
             "Measured on same-cell revisit pairs. A well-formed position kernel is "
             "positive and maximal at zero displacement.", "",
             "| model | same-cell A_P | % negative | % row-max | n pairs |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['same_cell_mean']:+.4f} | "
                     f"{r['frac_negative']:.1%} | {r['frac_rowmax']:.1%} | {r['n_pairs']} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines)); print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
