"""Same-cell phase coherence by revisit gap: does the where-code put a revisited cell at phase difference 0, and does
MapWM's content phase psi compensate a systematic phase error that PoPE (psi = 0, delta ~ 0) cannot? Post hoc, CPU.
Companion to neuro_rank2_mech.py (same checkpoints, same walks).

For every revisit query (action token at move i) and every earlier observation key in the SAME cell (move j), gap n = i - j:
  kernel phase per channel   RoPE form: (theta_k - theta_q) + psi_c(a, o), psi = arg k_c(o) - arg q_c(a)
                             PoPE form: (theta_k - theta_q) + delta_c          (layer code: cos(theta_k + delta - theta_q))
  coherence  C  = sum_c A_c cos(kernel phase_c) / sum_c A_c   (A = |q_c||k_c| or softplus magnitudes; 1 = perfect peak)
             C0 = the same with psi (or delta) removed: is the where-code ALONE aligned at the revisited cell?
  comp       = for RoPE form, the amplitude-weighted mean cos(err_c + psi_c) where err_c = circular mean over same-cell pairs
               of (theta_k - theta_q): 1 = the content phase exactly cancels the code's mean same-cell phase error.
Reported per arm (median over seeds of the better head) in gap bins 1-4, 5-16, 17-64, 65-127 moves.
"""
import json, math, sys
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-05")
import neuro_rank2_mech as NR

BINS = [(1, 4), (5, 16), (17, 64), (65, 127)]


@torch.no_grad()
def analyse(m, toks, revs, pos):
    k_ = NR.kind(m); lay = m.layers[0]; H = m.n_heads
    x = m.token_emb(toks)
    ang = (torch.cumsum(m.action_to_lie(x), 1) * m.path_integrator.omega).numpy()       # (B, L, H, nb)
    q, k = NR.tables(m)
    if k_ == "rope":
        qc = torch.complex(q[..., 0::2], q[..., 1::2]); kc = torch.complex(k[..., 0::2], k[..., 1::2])   # (V, H, nb)
        Z = (kc[None, :] * qc[:, None].conj()).numpy()                                   # (Vq, Vk, H, nb)
        A, psi = np.abs(Z), np.angle(Z)
    else:
        mq, mk = F.softplus(q).numpy(), F.softplus(k).numpy()
        if k_ == "pair":
            mq, mk = mq.reshape(*mq.shape[:2], -1, 2), mk.reshape(*mk.shape[:2], -1, 2)
            A = np.einsum("ahce,bhce->abhc", mq, mk)                                      # both elements share the angle
            d = lay.pope_delta.clamp(NR.DELTA_MIN, NR.DELTA_MAX).numpy().reshape(H, -1, 2)
            assert np.allclose(d[..., 0], d[..., 1]) or True
            # untied deltas inside a pair: use the magnitude-weighted resultant per pair
            dz = (mq[:, None] * mk[None] * np.exp(1j * d)[None, None]).sum(-1)
            A = np.abs(dz); psi = np.angle(dz)
        else:
            A = mq[:, None] * mk[None]
            psi = np.broadcast_to(lay.pope_delta.clamp(NR.DELTA_MIN, NR.DELTA_MAX).numpy()[None, None], A.shape)
    tk = toks.numpy(); res = {h: {b: [[], []] for b in BINS} for h in range(H)}; errs = {h: [] for h in range(H)}; wts = {h: [] for h in range(H)}
    for b in range(tk.shape[0]):
        for i in range(1, 128):
            if not bool(revs[b, 2 * i + 1]):
                continue
            js = [j for j in range(i) if (pos[b, j] == pos[b, i]).all()]
            for j in js:
                n = i - j; qi, ki = 2 * i, 2 * j + 1; a, o = tk[b, qi], tk[b, ki]
                for h in range(H):
                    dth = ang[b, ki, h] - ang[b, qi, h]
                    Aw, ps = A[a, o, h], psi[a, o, h]
                    C = (Aw * np.cos(dth + ps)).sum() / Aw.sum(); C0 = (Aw * np.cos(dth)).sum() / Aw.sum()
                    for lo, hi in BINS:
                        if lo <= n <= hi:
                            res[h][(lo, hi)][0].append(C); res[h][(lo, hi)][1].append(C0)
                    errs[h].append(np.exp(1j * dth)); wts[h].append(Aw)
    out = {}
    for h in range(H):
        e = np.array(errs[h]); w = np.array(wts[h])
        err_c = np.angle((w * e).sum(0))                                                # per-channel circular mean error
        Abar = A[:4, 4:, h].mean((0, 1))
        psibar = np.angle((A[:4, 4:, h] * np.exp(1j * psi[:4, 4:, h])).sum((0, 1)))
        comp = float((Abar * np.cos(err_c + psibar)).sum() / Abar.sum())
        errmag = float((Abar * (1 - np.cos(err_c))).sum() / Abar.sum())                 # size of the mean error itself
        out[h] = dict(comp=comp, err=errmag, **{f"C{lo}-{hi}": float(np.mean(v[0])) if v[0] else float("nan") for (lo, hi), v in res[h].items()},
                      **{f"C0_{lo}-{hi}": float(np.mean(v[1])) if v[1] else float("nan") for (lo, hi), v in res[h].items()})
    return out


def main():
    toks, revs, pos = NR.walks()
    prev = json.load(open(f"{NR.REPO}/docs/audits/2026-10-05/neuro_rank2_mech.json"))
    allr = {}
    for arm, seeds in [("Vanilla", range(10, 26)), ("MapPoPE-Pair", range(10, 26)), ("MapPoPE-Flat", range(10, 26)), ("Vanilla_r4", range(10, 18))]:
        rows = []
        for s in seeds:
            m = NR.load(arm, s); r = analyse(m, toks, revs, pos)
            pr = next(p for p in prev[arm] if p["seed"] == s)
            cls = "".join(h["cls"][0] for h in pr["heads"])
            hb = max(r, key=lambda h: r[h]["C17-64"])                                     # the better head on mid gaps
            rr = r[hb]
            print(f"{arm:14s} s{s} acc {pr['acc']:.4f} cls {cls} head{hb} err {rr['err']:.3f} comp {rr['comp']:+.3f} | C   "
                  + " ".join(f"{rr[f'C{lo}-{hi}']:+.3f}" for lo, hi in BINS) + " | C0  " + " ".join(f"{rr[f'C0_{lo}-{hi}']:+.3f}" for lo, hi in BINS), flush=True)
            rows.append(dict(seed=s, acc=pr["acc"], cls=cls, heads={str(h): v for h, v in r.items()}))
        allr[arm] = rows
    json.dump(allr, open(f"{NR.REPO}/docs/audits/2026-10-05/neuro_samecell_phase.json", "w"), indent=1)


if __name__ == "__main__":
    main()
