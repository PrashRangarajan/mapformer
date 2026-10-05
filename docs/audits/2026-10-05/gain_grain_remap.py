"""Remap-probe decomposition (docs/audits/2026-10-05/remap_probe.py, unchanged decompose()) on the GAIN_GRAIN arms:
declared secondary of GAIN_GRAIN_PREREG.md, eval-only, CPU. Adds the two new score kinds to probe_whatwhere's
re-implementation of the layer score:
  gain   GainKernelLayer (GainScalar, GainMod4): S = (2/sqrt dh) sum_c mu^q_m(c) mu^k_m(c) A_c cos(dphase_c)
  em_nn  EMTransformerLayer_NonNeg: S = softplus(A_X) * A_P
and checks every rebuilt forward against the model's own logits (verify column, max |dlogit|). Per arm (medians over
seeds x heads): gain / even / odd shares of the content x position interaction, peak at d = 0 (shift0), width cv,
r(height, width); plus, for the gain arms, the learned spectrum (share of sum_c A_c in each 8-channel omega band, fine
-> coarse) and the spread of the observation-key gains (cv of mu^k over the 17 observation tokens); for MapEM, the
share of (action query, observation key) pairs with A_X < 0 (content inverting the kernel).

  python3 docs/audits/2026-10-05/gain_grain_remap.py            # the batch checkpoints (after the batch)
  python3 docs/audits/2026-10-05/gain_grain_remap.py --untrained  # pipeline check on untrained models (no outcome read)
Output: gain_grain_remap_out.txt / .json (batch); gain_grain_remap_untrained_out.txt (pipeline check)."""
import glob
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-05")
sys.path.insert(0, "/home/prashr")
import remap_probe as RP                       # also imports probe_whatwhere (PW) and patches its pairwise phases
from mapformer import train_variant
from mapformer.model_em_pope import register
from mapformer.model import _apply_rope
from mapformer.environment import GridWorld

PW = RP.PW
register(train_variant.VARIANT_MAP); PW.VARIANT_MAP = train_variant.VARIANT_MAP
REPO = "/home/prashr/mapformer"; R = PW.R; NA, NO = PW.N_ACT, PW.N_OBS
ARMS = ["Vanilla", "MapPoPE-Pair", "GainScalar", "GainMod4", "VanillaEM", "VanillaEM_NonNeg"]
_kind0, _scores0, _phases0 = PW.kind, PW.scores, PW.phases_seq


def kind(m):
    L = m.layers[0]
    if hasattr(L, "q_gain"):
        return "gain"
    if type(L).__name__ == "EMTransformerLayer_NonNeg":
        return "em_nn"
    return _kind0(m)


def scores(m, layer, hq, hk, cq, sq, ck, sk, qp=None, kp=None):
    k = kind(m)
    if k == "gain":
        return layer.score(hq, hk, cq, sq, ck, sk), None, None
    if k == "em_nn":
        B, Tq, _ = hq.shape; Tk = hk.shape[1]; H, dh = layer.n_heads, layer.d_head
        sh = lambda z, T: z.view(B, T, H, dh).transpose(1, 2)
        AX = sh(layer.q_content(hq), Tq) @ sh(layer.k_content(hk), Tk).transpose(-1, -2) / math.sqrt(dh)
        AP = qp @ kp.transpose(-1, -2) / math.sqrt(dh)
        return F.softplus(AX) * AP, AX, AP
    return _scores0(m, layer, hq, hk, cq, sq, ck, sk, qp, kp)


def phases_seq(m, tokens):
    if kind(m) == "em_nn":
        B, L = tokens.shape; c, s = m.path_integrator(m.action_to_lie(m.token_emb(tokens)))
        q0 = m.q0_pos[None, :, None, :].expand(B, -1, L, -1); k0 = m.k0_pos[None, :, None, :].expand(B, -1, L, -1)
        return c, s, _apply_rope(q0, c, s), _apply_rope(k0, c, s)
    return _phases0(m, tokens)            # gain: path phases, one per channel (no pair duplication)


PW.kind, PW.scores, PW.phases_seq = kind, scores, phases_seq


@torch.no_grad()
def kernel_matrix(m):
    """S (H, 68, P) over displacements: remap_probe.kernel_matrix with the new kinds."""
    layer = m.layers[0]; H = m.n_heads; E = m.token_emb.weight
    hq = layer.norm1(E[:NA])[None]; hk = layer.norm1(E[NA:NA + NO])[None]
    st = m.action_to_lie(E[None])[0] * m.path_integrator.omega
    Nn, Ss, Ww, Ee = st[0], st[1], st[2], st[3]
    disp = [(dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1)]
    ang = torch.stack([(dx * Ee if dx > 0 else -dx * Ww) + (dy * Nn if dy > 0 else -dy * Ss) for dx, dy in disp])
    k = kind(m); aq = torch.zeros(1, H, NA, st.shape[-1]); rows = []
    for p in range(len(disp)):
        ak = ang[p][:, None, :].expand(H, NO, -1)[None]
        if k in ("em", "em_nn"):
            qp = _apply_rope(m.q0_pos[None, :, None, :].expand(1, H, NA, -1), aq.cos(), aq.sin())
            kp = _apply_rope(m.k0_pos[None, :, None, :].expand(1, H, NO, -1), ak.cos(), ak.sin())
            Sx, _, _ = scores(m, layer, hq, hk, None, None, None, None, qp, kp)
        else:
            cq, sq, ckk, skk = aq.cos(), aq.sin(), ak.cos(), ak.sin()
            if type(m).__name__.startswith("MapFormerWM_PoPEPair"):
                cq, sq, ckk, skk = [z.repeat_interleave(2, -1) for z in (cq, sq, ckk, skk)]
            Sx, _, _ = scores(m, layer, hq, hk, cq, sq, ckk, skk)
        rows.append(Sx[0])
    return torch.stack(rows, -1).reshape(H, NA * NO, len(disp)).double().numpy(), disp


@torch.no_grad()
def extras(m):
    L = m.layers[0]; E = m.token_emb.weight; out = {}
    if kind(m) == "gain":
        A = L.amplitude().numpy(); band = A.reshape(A.shape[0], 4, -1).sum(-1) / A.sum(-1, keepdims=True)   # (H, 4)
        _, gk = L.gains(L.norm1(E[NA:NA + NO])[None]); g = gk[0, :, :, ::L.n_blocks // L.n_modules].numpy()   # (H, 17, M)
        out = {"band_share": band.tolist(), "key_gain_cv": (g.std(1) / g.mean(1)).mean(-1).tolist()}
    if kind(m) in ("em", "em_nn"):
        h = L.norm1(E); H, dh = L.n_heads, L.d_head
        AX = torch.einsum("ahd,ohd->hao", L.q_content(h[:NA]).view(NA, H, dh), L.k_content(h[NA:NA + NO]).view(NO, H, dh)) / math.sqrt(dh)
        out = {"neg_AX_share": (AX < 0).double().mean((1, 2)).tolist()}
    return out


def main():
    untrained = "--untrained" in sys.argv
    env = GridWorld(size=64, n_obs_types=16, seed=10000); np.random.seed(0)
    toks = torch.stack([env.generate_trajectory(64)[0] for _ in range(2)])
    keys = ["inter_of_pos", "gain", "even", "odd", "shift0", "shift_mean", "height_cv", "width_cv", "width_mean", "r_height_width"]
    print(f"{'arm':18s} " + " ".join(f"{k:>14s}" for k in keys) + "   verify")
    res = {}
    for a in ARMS:
        if untrained:
            models = [PW.load(None, a, untrained_seed=s)[0] for s in (0, 1)]
        else:
            paths = sorted(glob.glob(f"{REPO}/runs/gain_grain/p0/{a}_s*/{a}.pt"))
            assert len(paths) == 20, (a, len(paths))
            models = [PW.load(p)[0] for p in paths]
        rows, errs, ex = [], [], []
        for m in models:
            errs.append(PW.verify(m, toks)); X, disp = kernel_matrix(m); ex.append(extras(m))
            rows += [RP.decompose(X[h], disp) for h in range(X.shape[0])]
        res[a] = {"heads": rows, "extras": ex}
        med = {k: float(np.nanmedian([r[k] for r in rows])) if not all(np.isnan(r[k]) for r in rows) else float("nan") for k in keys}
        print(f"{a:18s} " + " ".join(f"{med[k]:14.3f}" for k in keys) + f"   {max(errs):.1e}", flush=True)
    print()
    for a in ARMS:
        ex = res[a]["extras"]
        if ex and "band_share" in ex[0]:
            b = np.array([e["band_share"] for e in ex]).reshape(-1, 4); g = np.array([e["key_gain_cv"] for e in ex]).ravel()
            print(f"{a:18s} spectrum share per omega band (fine -> coarse), median over runs x heads: "
                  + " ".join(f"{v:.3f}" for v in np.median(b, 0)) + f"; key-gain cv median {np.median(g):.3f}")
        if ex and "neg_AX_share" in ex[0]:
            v = np.array([e["neg_AX_share"] for e in ex]).ravel()
            print(f"{a:18s} share of (action, observation) pairs with A_X < 0: median {np.median(v):.3f}, heads all-negative "
                  f"{int((v > 0.99).sum())}/{len(v)}")
    if not untrained:
        json.dump(res, open(f"{REPO}/docs/audits/2026-10-05/gain_grain_remap.json", "w"), indent=1)
    print("\n(medians over runs x 2 heads; gain + even + odd = 1 of the interaction; verify = max |dlogit| of the rebuilt forward)")


if __name__ == "__main__":
    main()
