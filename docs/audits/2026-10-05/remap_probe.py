"""Remapping-type probe (docs/theory/2026-10-05/neuro_positional.md, M3). Post hoc, eval-only, CPU.

For a 1-layer torus model the pre-softmax score between query action a and key observation o, as a function of the
key cell's displacement d from the query cell, is exactly S(a, o, d) (built by docs/audits/2026-09-27/probe_whatwhere.py's
`scores`, verified against the model's own logits). Each of the 68 content pairs (4 actions x 17 observations) has a
position kernel K_c(d) over the window d in [-16, 16]^2. Neural reading: K_c is the "place field" of a stored token as
seen by query a; how content changes it is the remapping type.

Decomposition of the content x position INTERACTION I_c(d) = S - row mean - column mean + grand mean, per head:
  gain   the part of I explained by scaling the shared kernel k(d) (top right singular vector of the row-centred S):
         I_c ~ (b_c - mean b) k(d)                                      -> rate remapping (height only)
  even   residual symmetric under d -> -d                               -> width / shape change about the same centre
  odd    residual antisymmetric under d -> -d                           -> peak shift / skew (partial remapping)
Shares are of ||I||^2 (they sum to 1). Per content pair, from the row-centred kernel K_c - mean_d K_c:
  shift   torus distance (cells) of argmax_d K_c from d = 0            -> 0 = field stays put
  height  K_c(peak) - median_d K_c
  width   number of cells with K_c above half-height (median + height/2), i.e. field area in cells
and across content pairs: corr(height, width) and corr(height, shift) -- the neural predictions N1 (MapPoPE-like recall:
rate changes co-vary with width; MapEM-like: amplitude only; MapWM-like: centroid shifts).
Also the interaction's share of the position-dependent variance (inter_of_pos), for scale.
"""
import glob, json, os, sys
import numpy as np
import torch
torch.set_num_threads(4)
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-09-27")
import probe_whatwhere as PW
from mapformer import train_variant
from mapformer.model_pope_pair import MapFormerWM_PoPEPair, MapFormerWM_PoPEPair_r4
from mapformer.environment import GridWorld
from mapformer.model import _apply_rope
train_variant.VARIANT_MAP["MapPoPE-Pair"] = MapFormerWM_PoPEPair
train_variant.VARIANT_MAP["MapPoPE-Pair_r4"] = MapFormerWM_PoPEPair_r4
PW.VARIANT_MAP = train_variant.VARIANT_MAP
_phases = PW.phases_seq


def _phases_pair(m, tokens):                      # the pairwise model shares each angle across an element pair
    c, s, qp, kp = _phases(m, tokens)
    if type(m).__name__.startswith("MapFormerWM_PoPEPair"):
        c, s = c.repeat_interleave(2, -1), s.repeat_interleave(2, -1)
    return c, s, qp, kp


PW.phases_seq = _phases_pair

REPO = "/home/prashr/mapformer"; R = PW.R; NA, NO = PW.N_ACT, PW.N_OBS


@torch.no_grad()
def kernel_matrix(m):
    """S (H, 68, P) over displacements, the probe_whatwhere construction (key's own step excluded)."""
    layer = m.layers[0]; H = m.n_heads; E = m.token_emb.weight
    hq = layer.norm1(E[:NA])[None]; hk = layer.norm1(E[NA:NA + NO])[None]
    st = m.action_to_lie(E[None])[0] * m.path_integrator.omega
    Nn, Ss, Ww, Ee = st[0], st[1], st[2], st[3]
    disp = [(dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1)]
    ang = torch.stack([(dx * Ee if dx > 0 else -dx * Ww) + (dy * Nn if dy > 0 else -dy * Ss) for dx, dy in disp])
    k = PW.kind(m); aq = torch.zeros(1, H, NA, st.shape[-1]); rows = []
    for p in range(len(disp)):
        ak = ang[p][:, None, :].expand(H, NO, -1)[None]
        if k == "em":
            qp = _apply_rope(m.q0_pos[None, :, None, :].expand(1, H, NA, -1), aq.cos(), aq.sin())
            kp = _apply_rope(m.k0_pos[None, :, None, :].expand(1, H, NO, -1), ak.cos(), ak.sin())
            Sx, _, _ = PW.scores(m, layer, hq, hk, None, None, None, None, qp, kp)
        else:
            cq, sq, ckk, skk = aq.cos(), aq.sin(), ak.cos(), ak.sin()
            if type(m).__name__.startswith("MapFormerWM_PoPEPair"):        # angles shared by element pairs
                cq, sq, ckk, skk = [z.repeat_interleave(2, -1) for z in (cq, sq, ckk, skk)]
            Sx, _, _ = PW.scores(m, layer, hq, hk, cq, sq, ckk, skk)
        rows.append(Sx[0])
    return torch.stack(rows, -1).reshape(H, NA * NO, len(disp)).double().numpy(), disp


def decompose(X, disp):
    P = len(disp); i0 = disp.index((0, 0)); mirror = np.array([disp.index((-dx, -dy)) for dx, dy in disp])
    g = X.mean(); rm = X.mean(1, keepdims=True); cm = X.mean(0, keepdims=True)
    I = X - rm - cm + g; pos = ((cm - g) ** 2).mean(); inter = (I ** 2).mean()
    Xc = X - rm
    u, s, vt = np.linalg.svd(Xc, full_matrices=False); kv = vt[0]
    b = Xc @ kv; Igain = np.outer(b - b.mean(), kv)
    Rr = I - Igain; Re, Ro = (Rr + Rr[:, mirror]) / 2, (Rr - Rr[:, mirror]) / 2
    tot = (I ** 2).sum()
    out = {"inter_of_pos": float(inter / (pos + inter)), "gain": float((Igain ** 2).sum() / tot),
           "even": float((Re ** 2).sum() / tot), "odd": float((Ro ** 2).sum() / tot)}
    D = np.array(disp)
    Kc = Xc - np.median(Xc, axis=1, keepdims=True)
    pk = Kc.argmax(1); height = Kc.max(1)
    width = (Kc > height[:, None] / 2).sum(1).astype(float)
    shift = np.abs(D[pk]).sum(1).astype(float)                     # L1 cells from d = 0 (window, no wrap at R=16)
    out.update(shift_mean=float(shift.mean()), shift0=float((shift == 0).mean()), height_cv=float(height.std() / height.mean()),
               width_cv=float(width.std() / width.mean()), width_mean=float(width.mean()),
               r_height_width=float(np.corrcoef(height, width)[0, 1]) if width.std() > 0 else float("nan"),
               r_height_shift=float(np.corrcoef(height, shift)[0, 1]) if shift.std() > 0 else float("nan"))
    return out


def main():
    env = GridWorld(size=64, n_obs_types=16, seed=10000); np.random.seed(0)
    toks = torch.stack([env.generate_trajectory(64)[0] for _ in range(2)])
    sets = {a: sorted(glob.glob(f"{REPO}/runs/paper2x2/p0/{a}_s[0-7]/{a}.pt")) for a in ("Vanilla", "Vanilla_r4", "MapPoPE-Flat", "MapPoPE_r4")}
    sets["MapPoPE-Pair (mappope_pair, s10-17)"] = sorted(glob.glob(f"{REPO}/runs/mappope_pair/p0/MapPoPE-Pair_s1[0-7]/MapPoPE-Pair.pt"))
    sets["MapPoPE-Pair_r4 (mappope_pair)"] = sorted(glob.glob(f"{REPO}/runs/mappope_pair/p0/MapPoPE-Pair_r4_s1[0-7]/MapPoPE-Pair_r4.pt"))
    sets["VanillaEM_r4 (dof batch)"] = sorted(glob.glob(f"{REPO}/runs/dof/torus/VanillaEM_r4_s[0-7]/VanillaEM_r4.pt"))
    res = {}
    keys = ["inter_of_pos", "gain", "even", "odd", "shift0", "shift_mean", "height_cv", "width_cv", "width_mean", "r_height_width", "r_height_shift"]
    print(f"{'arm':38s} " + " ".join(f"{k:>14s}" for k in keys) + "   verify")
    for name, paths in sets.items():
        assert len(paths) == 8, (name, len(paths))
        rows, errs = [], []
        for p in paths:
            m, v = PW.load(p); errs.append(PW.verify(m, toks))
            X, disp = kernel_matrix(m)
            for h in range(X.shape[0]):
                rows.append(decompose(X[h], disp))
        res[name] = rows
        med = {k: float(np.nanmedian([r[k] for r in rows])) if not all(np.isnan(r[k]) for r in rows) else float('nan') for k in keys}
        print(f"{name:38s} " + " ".join(f"{med[k]:14.3f}" for k in keys) + f"   {max(errs):.1e}", flush=True)
    json.dump(res, open(f"{REPO}/docs/audits/2026-10-05/remap_probe.json", "w"), indent=1)
    print("\n(medians over 8 seeds x 2 heads; gain + even + odd = 1 of the interaction; shift in cells L1; width in cells)")


if __name__ == "__main__":
    main()
