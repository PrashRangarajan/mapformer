"""Extract and compare the COARSE-level position angles of the three designs.

  MapWM-Hier      coarse angle = mean-pool of the fine cum_delta  (pooled)
  CoarsePI        coarse angle = the coarse level's OWN cumsum    (learned)
  CoarseIdx       coarse angle = omega * coarse_index             (ordinal)

Dumps, per variant and per eval length, the pre-rotation angle theta = omega*cum
at the coarse level: magnitude over coarse-token index, per-frequency spread,
and how far it travels outside the range seen at training length. That is the
proposed mechanism for why pooling the fine angle hurts.
"""
import json
from pathlib import Path

import numpy as np
import torch

from mapformer.environment_hier_goal import HierGoalGridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.model_hourglass import _causal_shorten

_REPO = Path(__file__).resolve().parent
OUT = _REPO / "coarse_angles.json"
CKPT = _REPO / "runs/hiergoal_multiseed/seed0"
VARIANTS = {"Hourglass_k2": "pooled", "Hourglass_CoarsePI": "learned",
            "Hourglass_CoarseIdx": "ordinal"}


@torch.no_grad()
def coarse_theta(model, tokens, kind):
    """Return theta (B, S, H, nb) at the coarse level, pre-cos/sin."""
    B, L = tokens.shape
    k = model.k
    x = model.token_emb(tokens)
    delta = model.action_to_lie(x)
    cum = torch.cumsum(delta, dim=1)
    pad = (-L) % k
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        cum = torch.nn.functional.pad(cum, (0, 0, 0, 0, 0, pad))
        cum[:, L:] = cum[:, L - 1:L]
    Lp, S = L + pad, (L + pad) // k
    om = model.path_integrator.omega.unsqueeze(0).unsqueeze(0)      # (1,1,H,nb)
    if kind == "pooled":
        cc = _causal_shorten(cum.reshape(B, Lp, -1), k).view(B, S, model.n_heads, model.n_blocks)
    elif kind == "learned":
        xc = _causal_shorten(model.pre_layers[0](
            x, *model._angles(cum), model._causal_mask(Lp, tokens.device)), k)
        cc = torch.cumsum(model.coarse_action_to_lie(xc), dim=1)
    else:  # ordinal
        idx = torch.arange(S, device=tokens.device, dtype=cum.dtype)
        cc = idx.view(1, S, 1, 1).expand(B, S, model.n_heads, model.n_blocks)
    return cc * om                                                   # theta


def main():
    env = HierGoalGridWorld(size=64, room_size=8, seed=10000)
    rng = np.random.RandomState(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = {}
    for v, kind in VARIANTS.items():
        p = CKPT / f"{v}_hiergoal.pt"
        if not p.exists():
            print(f"MISSING {p}"); continue
        c = torch.load(p, map_location=dev, weights_only=False)
        m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"],
                           n_heads=c["n_heads"], n_layers=c["n_layers"], grid_size=64).to(dev).eval()
        m.load_state_dict(c["model_state"])
        rec = {"kind": kind, "lengths": {}}
        for T in (64, 256):                       # trained length vs OOD
            toks, _, _, _ = env.generate_hier_batch(64, T, 64, rng)
            th = coarse_theta(m, toks[:, :-1].to(dev), kind).float().cpu()   # (B,S,H,nb)
            mag = th.abs().mean(dim=(2, 3))                                   # (B,S) mean |theta|
            rec["lengths"][str(T)] = {
                # magnitude of the rotation angle along the coarse sequence
                "mag_mean": mag.mean(0).tolist(),
                "mag_p10": mag.quantile(0.10, dim=0).tolist(),
                "mag_p90": mag.quantile(0.90, dim=0).tolist(),
                # spread ACROSS episodes at each coarse position: does the angle
                # actually carry path-dependent (spatial) information?
                "across_episode_std": mag.std(0).tolist(),
                # per-frequency-band magnitude at the final coarse token
                "band_mag": th[:, -1].abs().mean(0).flatten().tolist(),
                "max_abs": float(th.abs().max()),
            }
        out[v] = rec
        print(f"{v:22s} kind={kind:8s} "
              f"|theta|max T=64: {rec['lengths']['64']['max_abs']:8.1f}  "
              f"T=256: {rec['lengths']['256']['max_abs']:8.1f}")
    Path(OUT).write_text(json.dumps(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
