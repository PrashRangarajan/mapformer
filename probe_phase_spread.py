"""Is MapWM's position kernel per-pair, and does training BUILD that? (EM_WM_STATE.md open q4)

Per block b, a pair (t, s) contributes `|Q_b||K_b| cos(dtheta_b - phi_b)` with
`phi = atan2(q x k, q.k)`. The rotation family is shared in both models; what differs is
whether the phase can vary from pair to pair:

  WM  Q, K are content projections of the two tokens -> a phase per pair
  EM  q0, k0 are single learned vectors -> one phase per (head, block) for the whole model.
      EM's content branch A_X is never rotated, so content can only rescale the kernel.

**Three corrections after an adversarial review (2026-09-11), each of which changed a reading:**

1. The EM rows used to be written as the LITERAL 0.0 while the report called them "measured".
   They are now computed over the same pairs as WM. They come out 0 because q0/k0 do not
   depend on the tokens -- which is the claim, and it must be measured, not asserted (rule 23).
2. `per_k_spread` was computed over whatever offsets had >= 3 samples, which at 8 episodes was
   2 of 40. It now needs MIN_BINS offsets with >= MIN_PER_BIN samples or reports NaN.
3. "circular sd ~2.0 = near-uniform" was WRONG. The finite-sample uniform null is ~2.95 at
   this N (printed below), so 2.0 is CONCENTRATED, about 9x the null's R.

**The control that decides the interpretation.** An UNTRAINED WM is included. If training built
the per-pair structure, the trained model should be more spread than the untrained one. It is
not: the untrained model scores HIGHER. So this statistic reflects the parameterisation (random
content projections give near-uniform phases), not something training discovers -- it supports
"WM CAN reshape per pair" and refutes "WM's advantage comes from reshaping per pair".

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
NCAND = 64            # most recent symbol keys per query
N_EP = 64             # episodes; at ~7 scored queries each this gives ~7 samples per offset
MIN_PER_BIN, MIN_BINS = 5, 16


def _phase(q, k):
    A = q[..., 0] * k[..., 0] + q[..., 1] * k[..., 1]
    B = q[..., 1] * k[..., 0] - q[..., 0] * k[..., 1]
    return np.arctan2(B, A), np.hypot(A, B)


def _circ_sd(z_mean):
    return float(np.sqrt(np.maximum(-2 * np.log(np.clip(np.abs(z_mean), 1e-12, 1)), 0)).mean())


def _flat(a):
    """(pairs, H, ncand, nb) -> (pairs*ncand, H, nb).

    BUG FIXED 2026-09-11: this used to be `a.reshape(-1, H, nb)`, which flattens in C order and
    so interleaves the CANDIDATE axis across heads and blocks -- every column was a scramble of
    different blocks' phases. It produced a non-zero spread for separate-q0/k0 EM, where the
    algebra says exactly 0, and a non-zero amplitude-weighted value for single-p0 EM whose
    unweighted spread was 0. Both impossibilities are what exposed it. Transpose first.
    """
    return a.transpose(0, 2, 1, 3).reshape(-1, a.shape[1], a.shape[3])


def _per_k(ph, ks):
    """Spread of the per-offset MEAN phase, or NaN when the bins are too thin to mean anything."""
    keep = [k for k in sorted(set(ks.tolist())) if (ks == k).sum() >= MIN_PER_BIN]
    if len(keep) < MIN_BINS:
        return float("nan")
    mk = [np.angle(np.exp(1j * _flat(ph[ks == k])).mean(0)) for k in keep]
    return _circ_sd(np.exp(1j * np.stack(mk)).mean(0))


def uniform_null(n_pairs, H=2, nb=32, reps=5, seed=0):
    """circ_sd of n_pairs phases drawn uniformly -- the reference 'no structure' value at this N."""
    rng = np.random.RandomState(seed)
    return float(np.mean([_circ_sd(np.exp(1j * rng.uniform(-np.pi, np.pi, (n_pairs, H, nb))).mean(0))
                          for _ in range(reps)]))


@torch.no_grad()
def _episode_pairs(m, env, seed, per_pair_qk, n_ep=N_EP, T=1024):
    """Phases over real (query, symbol-key) pairs. `per_pair_qk(lay, e, x)` -> (Q, K) or None
    for the EM case, where the origins replace the content projections."""
    rng = np.random.RandomState(7000 + seed)
    toks, sps, ans, infos = env.generate_batch(n_ep, T, rng)
    x = toks[:, :-1]
    e = m.token_emb(x)
    lay = m.layers[0]
    H, dh = lay.n_heads, lay.d_head
    QK = per_pair_qk(lay, e, x, H, dh)
    ph, am, ks = [], [], []
    for b in range(x.shape[0]):
        sym = np.asarray(infos[b]["sym_positions"])
        for p, k in zip(sps[b], infos[b]["offsets"]):
            if p >= x.shape[1]:
                continue
            cand = sym[sym < p][-NCAND:]
            if QK is None:                                  # EM: origins, same for every pair
                q0, k0, _ = position_origins(m)
                q = q0.view(H, dh // 2, 2).double().numpy()[:, None]
                kk = np.broadcast_to(k0.view(H, dh // 2, 2).double().numpy()[:, None],
                                     (H, len(cand), dh // 2, 2))
            else:
                Q, K = QK
                q = Q[b, :, p].view(H, dh // 2, 2).double().numpy()[:, None]
                kk = K[b, :, cand].view(H, len(cand), dh // 2, 2).double().numpy()
            a, g = _phase(q, kk)
            ph.append(a); am.append(g); ks.append(k)
    return np.stack(ph), np.stack(am), np.array(ks)


def _wm_qk(lay, e, x, H, dh):
    h = lay.norm1(e)
    return (lay.q_proj(h).view(x.shape[0], x.shape[1], H, dh).transpose(1, 2),
            lay.k_proj(h).view(x.shape[0], x.shape[1], H, dh).transpose(1, 2))


@torch.no_grad()
def pair_phases(pt, variant, seed, model=None):
    m, ck, env = load_model(pt, variant) if model is None else model
    m.eval()
    is_wm = hasattr(m.layers[0], "q_proj")
    return _episode_pairs(m, env, seed, _wm_qk if is_wm else (lambda *a: None))


@torch.no_grad()
def untrained_control(pt, variant, seed, init_seed):
    """The same architecture with parameters RE-INITIALISED: does training build the spread?"""
    m, ck, env = load_model(pt, variant)
    torch.manual_seed(init_seed)
    for mod in m.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    m.eval()
    return _episode_pairs(m, env, seed, _wm_qk)


def stats(ph, am, ks):
    """Unweighted and amplitude-weighted circular sd, plus the per-offset spread.

    BUG FIXED 2026-09-11 (second one): the weighted line divided by
    `clip(sum(amplitude), 1e-12, None)`. A trained single-`p0` kernel has DEAD blocks -- measured
    amplitudes down to 5e-86 -- so for those blocks a ~1e-81 numerator was divided by the 1e-12
    floor, giving |z| ~ 1e-69, which the statistic read as maximal spread. That inflated
    single-p0's weighted column to 3.496 while its unweighted column was 0.000, which is
    impossible and is what exposed it. Blocks carrying no amplitude are now excluded and counted.
    """
    flat, amf = _flat(ph), _flat(am)
    den = amf.sum(0)
    live = den > 1e-6 * den.max()
    z = (np.exp(1j * flat) * amf).sum(0)[live] / den[live]
    return dict(circ_sd=_circ_sd(np.exp(1j * flat).mean(0)),
                circ_sd_weighted=_circ_sd(z), n_blocks_live=int(live.sum()),
                n_blocks_total=int(live.size), per_k_spread=_per_k(ph, ks),
                n_pairs=int(flat.shape[0]))


def main():
    out, rows = {}, []
    npairs = None
    for variant, tmpl, label in ARMS:
        per_seed = []
        for s in SEEDS:
            pt = REPO / tmpl.format(s)
            if not pt.exists():
                raise FileNotFoundError(pt)
            r = stats(*pair_phases(pt, variant, s))
            if label.startswith("EM"):
                q0, k0, form = position_origins(load_model(pt, variant)[0])
                H, dh = q0.shape
                pk, _ = _phase(q0.view(H, dh // 2, 2).double().numpy(),
                               k0.view(H, dh // 2, 2).double().numpy())
                r.update(form=form, abs_phase_mean=float(np.abs(pk).mean()), n_blocks=int(pk.size))
            r["seed"] = s
            per_seed.append(r)
            npairs = r["n_pairs"]
            print(f"{label:20s} s{s} circ_sd {r['circ_sd']:.3f} per_k {r['per_k_spread']:.3f}", flush=True)
        out[variant] = per_seed
        g = lambda k: np.mean([x[k] for x in per_seed])
        extra = "" if label == "WM" else (f"  ({per_seed[0]['n_blocks']} blocks, one phase each; "
                                          f"\\|phase\\| mean {g('abs_phase_mean'):.3f})")
        rows.append(f"| {label} | {g('circ_sd'):.3f} | {g('circ_sd_weighted'):.3f} | "
                    f"{g('per_k_spread'):.3f} |{extra}")

    ctrl = [stats(*untrained_control(REPO / ARMS[0][1].format(0), "Vanilla_r4", 0, 555 + i))
            for i in range(3)]
    out["WM_untrained"] = ctrl
    gc = lambda k: np.mean([c[k] for c in ctrl])
    rows.append(f"| **WM, UNTRAINED (control)** | **{gc('circ_sd'):.3f}** | {gc('circ_sd_weighted'):.3f} | "
                f"{gc('per_k_spread'):.3f} |  (3 inits; the same architecture, random weights)")
    null = uniform_null(npairs)
    out["uniform_null"] = dict(circ_sd=null, n_pairs=npairs)

    txt = "\n".join([
        "## Per-pair phase spread of the position kernel (n=8 seeds; corrected 2026-09-11)\n",
        f"Circular sd of the kernel phase across query-key pairs, {npairs} pairs per model. "
        f"0 = one kernel for every pair. **The finite-sample uniform ('no structure') null at "
        f"this N is {null:.3f}**, so a value near 2.0 is CONCENTRATED, not near-uniform.\n",
        "| arm | circ sd of phase | amplitude-weighted | spread of per-offset mean phase |",
        "|---|---|---|---|", *rows, "",
        "**What this does and does not show.** EM's phases are pair-independent (measured 0.000, "
        "not assumed): they come from q0/k0, and EM's content branch is never rotated, so content "
        "can only rescale the shared kernel. WM's are set per pair. **But the untrained control "
        "scores HIGHER than the trained model**, so the spread is a property of the "
        "parameterisation, not something training discovers or uses. This supports 'WM CAN "
        "reshape per pair' and REFUTES any claim that WM's advantage comes from doing so."])
    print("\n" + txt)
    (REPO / "runs/search").mkdir(parents=True, exist_ok=True)
    (REPO / "runs/search/PHASE_SPREAD.md").write_text(txt + "\n")
    json.dump(out, open(REPO / "_PHASE_SPREAD.json", "w"), indent=1)


if __name__ == "__main__":
    main()
