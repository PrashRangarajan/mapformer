"""E0 side diagnostic: why zeroing BLANK steps costs object accuracy at ID (zob < zobj in e0_results.json).

Hypothesis: opposite actions do not cancel exactly (a(+x) + a(-x) = r_x != 0), so a closed loop carries a
residual phase proportional to its length; the learned blank step b (a constant; object steps are zero-mean
because the codes are zero-mean and the encoder has no bias) can offset part of that drift, so removing b
re-exposes it. Reports, per path arm, in step units and in angle units (omega * delta):
  |a_k|, |r_x|/|a|, |r_y|/|a|, |b|/|a|, cos(b, r_x), cos(b, r_y), R^2 of b regressed on span(r_x, r_y),
  the RMS object step (train-pool codes) / |a|, and the fraction of a random code's step energy that the
  rank-4 object-step map passes (= sum s_i^2 / (64 * mean |A c|^2 ...) reported as tr(M M^T)/64 / |a|^2).
Eval only, CPU. Run: python3 /home/prashr/mapformer/docs/audits/2026-10-03/e0_step_geometry.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/prashr")
import torch

from mapformer.environment_newobj import N_SPECIAL, BLANK
from mapformer.model_codes import use_object_codes
from mapformer.train_newobj import ARMS

REPO = Path("/home/prashr/mapformer")
OUT = REPO / "docs/audits/2026-10-03"
P = 1000


@torch.no_grad()
def main():
    res = {}
    for arm in ("MapWM", "MapPoPE", "MapEM", "PosOnly"):
        ck = torch.load(REPO / f"runs/newobj_pilot/{arm}_s100/{arm}.pt", map_location="cpu", weights_only=False)
        a = ck["config"]["args"]
        m = ARMS[arm](vocab_size=ck["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=a["n_layers"],
                      grid_size=a["size"])
        use_object_codes(m, N_SPECIAL, P)
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        om = m.path_integrator.omega.detach().flatten()
        tok = torch.arange(N_SPECIAL)[None]
        d_sp = m.action_to_lie(m.token_emb(tok))[0].flatten(1)              # (5, H*nb)
        objs = torch.arange(N_SPECIAL, N_SPECIAL + P)[None]
        d_ob = m.action_to_lie(m.token_emb(objs))[0].flatten(1)             # (P, H*nb) train pool
        row = {}
        for unit, w in (("step", torch.ones_like(om)), ("angle", om)):
            A = d_sp[:4] * w; b = d_sp[BLANK] * w; O = d_ob * w
            an = A.norm(dim=1).mean()
            rx, ry = A[0] + A[1], A[2] + A[3]
            S = torch.stack([rx, ry], 1)
            coef = torch.linalg.lstsq(S, b[:, None]).solution
            resid = b - (S @ coef)[:, 0]
            cos = lambda u, v: float(u @ v / (u.norm() * v.norm()))
            row[unit] = {"mean_|a|": float(an), "|r_x|/|a|": float(rx.norm() / an), "|r_y|/|a|": float(ry.norm() / an),
                         "|b|/|a|": float(b.norm() / an), "cos(b,r_x)": cos(b, rx), "cos(b,r_y)": cos(b, ry),
                         "R2_b_on_r": float(1 - resid.pow(2).sum() / b.pow(2).sum()),
                         "lstsq_coef_b_on_(r_x,r_y)": coef[:, 0].tolist(),
                         "rms_obj/|a|": float(O.pow(2).sum(1).mean().sqrt() / an),
                         "|mean_obj|/|a|": float(O.mean(0).norm() / an)}
        res[arm] = row
        print(arm, json.dumps(row, indent=1))
    json.dump(res, open(OUT / "e0_step_geometry.json", "w"), indent=1)


if __name__ == "__main__":
    main()
