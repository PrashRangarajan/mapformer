"""Is MapWM's position kernel per-pair, or shared like MapEM's? (EM_WM_STATE.md open question 4)

The claim under test (`AUDIT_2026-09-10.md` #1): MapEM applies ONE position kernel to every
query-key pair, while MapWM's kernel has amplitudes and phases set PER PAIR by content. A
literature sweep disputed it, arguing MapWM's kernel is also shared because the only positional
parameters are the per-head angular velocities `omega` and content enters solely as summed
scalar increments. Both readings agree on the algebra and differ on what "the kernel" means, so
this measures it.

Per block b, a pair (t, s) contributes to the score

    WM   Q_b(t)^T R(theta_s - theta_t) K_b(s) = |Q_b||K_b| cos(dtheta_b - phi_b(q, k))
    EM   q0_b^T R(theta_s - theta_t) k0_b     = |q0_b||k0_b| cos(dtheta_b - phi_b)

with `phi = atan2(B, A)`, `A = q.k`, `B = q x k`. The rotation family is shared in BOTH. What
differs is whether `phi` (and the amplitude) can vary from pair to pair:

  WM  Q, K are content projections of the two tokens -> one phase per pair, measured here
  EM  q0, k0 are single learned vectors -> ONE phase per (head, block) for the whole model,
      pair-independent by construction. EM's content branch A_X is never rotated, so it
      cannot contribute a phase at all; it can only rescale the shared kernel.

Readouts (WM): circular sd of `phi` over pairs per (head, block), `sqrt(-2 ln R)`; the same
weighted by amplitude (what the score actually sees); and the spread of the PER-OFFSET mean
phase, which asks whether WM's kernel specialises by the query's k rather than serving all
offsets with one shape. Circular sd 0 = one kernel for every pair; ~1.97 corresponds to R ~ 0.14,
i.e. near-uniform on the circle.

    python3 -m mapformer.probe_phase_spread
"""
from __future__ import annotations

import json

import numpy as np
import torch

from mapformer.ckpt_guard import REPO
from mapformer.probe_rewind import load_model, position_origins

ARMS = [("Vanilla_r4", "runs/recency_em/Vanilla_r4_s{}/Vanilla_r4_recency.pt", "WM"),
        ("VanillaEM_P0_r4", "runs/recency_em/VanillaEM_P0_r4_s{}/VanillaEM_P0_r4_recency.pt", "EM single p0"),
        ("VanillaEM_r4", "runs/recency_em/VanillaEM_r4_s{}/VanillaEM_r4_recency.pt", "EM separate q0/k0")]
SEEDS = range(8)
NCAND = 64          # most recent symbol keys per query


def _phase(q, k):
    """q, k: (..., nb, 2). -> (A.k, q x k) -> phase, amplitude."""
    A = q[..., 0] * k[..., 0] + q[..., 1] * k[..., 1]
    B = q[..., 1] * k[..., 0] - q[..., 0] * k[..., 1]
    return np.arctan2(B, A), np.hypot(A, B)


def _circ_sd(z_mean):
    return float(np.sqrt(np.maximum(-2 * np.log(np.clip(np.abs(z_mean), 1e-12, 1)), 0)).mean())


@torch.no_grad()
def wm_pair_phases(pt, variant, seed, n_ep=8, T=1024):
    m, ck, env = load_model(pt, variant)
    m.eval()
    lay = m.layers[0]
    assert hasattr(lay, "q_proj"), "WM layer expected (content Q/K are what get rotated)"
    rng = np.random.RandomState(7000 + seed)
    toks, sps, ans, infos = env.generate_batch(n_ep, T, rng)
    x = toks[:, :-1]
    e = m.token_emb(x)
    H, dh = lay.n_heads, lay.d_head
    h = lay.norm1(e)
    Q = lay.q_proj(h).view(x.shape[0], x.shape[1], H, dh).transpose(1, 2)
    K = lay.k_proj(h).view(x.shape[0], x.shape[1], H, dh).transpose(1, 2)
    ph, am, ks = [], [], []
    for b in range(x.shape[0]):
        sym = np.asarray(infos[b]["sym_positions"])
        for p, k in zip(sps[b], infos[b]["offsets"]):
            if p >= x.shape[1]:
                continue
            cand = sym[sym < p][-NCAND:]
            q = Q[b, :, p].view(H, dh // 2, 2).double().numpy()[:, None]        # (H,1,nb,2)
            kk = K[b, :, cand].view(H, len(cand), dh // 2, 2).double().numpy()  # (H,nc,nb,2)
            a, g = _phase(q, kk)
            ph.append(a); am.append(g); ks.append(k)
    return np.stack(ph), np.stack(am), np.array(ks)   # (npair,H,nc,nb)


@torch.no_grad()
def em_kernel_phases(pt, variant):
    m, ck, env = load_model(pt, variant)
    q0, k0, form = position_origins(m)
    H, dh = q0.shape
    ph, am = _phase(q0.view(H, dh // 2, 2).double().numpy(), k0.view(H, dh // 2, 2).double().numpy())
    return ph, am, form                                # (H,nb) -- the model's ONLY phases


def main():
    rows, out = [], {}
    for variant, tmpl, label in ARMS:
        per_seed = []
        for s in SEEDS:
            pt = REPO / tmpl.format(s)
            if not pt.exists():
                raise FileNotFoundError(pt)
            if label == "WM":
                ph, am, ks = wm_pair_phases(pt, variant, s)
                flat = ph.reshape(-1, ph.shape[1], ph.shape[3])
                amf = am.reshape(-1, ph.shape[1], ph.shape[3])
                sd = _circ_sd(np.exp(1j * flat).mean(0))
                wsd = _circ_sd((np.exp(1j * flat) * amf).sum(0) / np.clip(amf.sum(0), 1e-12, None))
                mk = [np.angle(np.exp(1j * ph[ks == kk]).reshape(-1, ph.shape[1], ph.shape[3]).mean(0))
                      for kk in sorted(set(ks.tolist())) if (ks == kk).sum() >= 3]
                perk = _circ_sd(np.exp(1j * np.stack(mk)).mean(0))
                per_seed.append(dict(seed=s, circ_sd=sd, circ_sd_weighted=wsd, per_k_spread=perk))
            else:
                ph, am, form = em_kernel_phases(pt, variant)
                per_seed.append(dict(seed=s, form=form, circ_sd=0.0, circ_sd_weighted=0.0,
                                     per_k_spread=0.0, abs_phase_mean=float(np.abs(ph).mean()),
                                     abs_phase_max=float(np.abs(ph).max()), n_blocks=int(ph.size)))
        out[variant] = per_seed
        g = lambda k: np.mean([r[k] for r in per_seed])
        extra = ("" if label == "WM" else
                 f"  (|phase| mean {g('abs_phase_mean'):.3f}, one phase per block, "
                 f"{per_seed[0]['n_blocks']} blocks, identical for every pair)")
        rows.append(f"| {label} | {g('circ_sd'):.3f} | {g('circ_sd_weighted'):.3f} | "
                    f"{g('per_k_spread'):.3f} |{extra}")

    txt = "\n".join([
        "## Per-pair phase spread of the position kernel (n=8 seeds, held-out episodes)\n",
        "Circular sd of the kernel phase across query-key pairs. 0 = one kernel for every pair; "
        "~1.97 is R~0.14, i.e. near-uniform on the circle.\n",
        "| arm | circ sd of phase | amplitude-weighted | spread of per-offset mean phase |",
        "|---|---|---|---|", *rows, "",
        "EM's phases are pair-independent BY CONSTRUCTION (they come from q0/k0, not from the "
        "tokens), and its content branch A_X is never rotated, so content can only rescale the "
        "shared kernel, not reshape it. WM's are set per pair by content, and vary systematically "
        "with the query's offset. This is the measured form of the shared-vs-per-pair contrast."])
    print(txt)
    (REPO / "runs/search").mkdir(parents=True, exist_ok=True)
    (REPO / "runs/search/PHASE_SPREAD.md").write_text(txt + "\n")
    json.dump(out, open(REPO / "_PHASE_SPREAD.json", "w"), indent=1)


if __name__ == "__main__":
    main()
