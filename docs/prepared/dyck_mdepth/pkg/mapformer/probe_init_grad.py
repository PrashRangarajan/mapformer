"""SEARCH_PREREG.md S2 -- the gradient at initialisation, and the ruggedness of the path to the rewind.

No training. Models are constructed exactly as train_recency does (torch.manual_seed(seed),
then VARIANT_MAP); the loss is the training loss on the first 8 training batches
(RandomState(seed)), in EVAL mode (no dropout) so the gradient is deterministic.

Readouts per (arm, seed), mean and t over the 8 batches:
  rate      slope over k of y_k = <g_k, s>/<s, s>, g_k = -dL/dDelta(q_k) (function space: the
            Delta table is a leaf), s = mean symbol Delta. Negative = gradient flow moves the
            code toward a linear rewind.
  cos_ker   cos(g_k, dJ_k), J = sum over queries of sum_h A_P(query, answer): does the loss
            gradient point toward raising the kernel at the answer?
  ratio     ||dL/d position pathway|| / ||dL/d content branch|| (parameter gradients)
  AP, AX    rms of A_P and A_X over causal entries
Ruggedness per k: strict interior local maxima of the mean (over queries with that k)
summed-head A_P at the answer, along the straight line from the current Delta(q_k) to the
exact linear rewind -(k-1) s, 2001 points. At init and at the trained P0 checkpoints.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

from mapformer.ckpt_guard import REPO, require_checkpoints
from mapformer.environment_recency import RecencyWorld
from mapformer.model import _apply_rope
from mapformer.probe_anatomy import em_forward, origins, query_mask

ARMS = ["VanillaEM_P0_r4", "EMDoF_alignfree", "EMDoF_magonly"]
NB, BS, T = 8, 16, 1024
KS = list(range(1, 65))


def recency_loss(logits, env, sps, ans):
    tl, tt = [], []
    for b in range(logits.shape[0]):
        for p, a in zip(sps[b], ans[b]):
            if p < logits.shape[1]:
                tl.append(logits[b, p, env.sym_offset:env.sym_offset + env.n_symbols])
                tt.append(a - env.sym_offset)
    return F.cross_entropy(torch.stack(tl), torch.tensor(tt, device=logits.device))


def delta_leaf(m):
    V = m.token_emb.num_embeddings if hasattr(m.token_emb, "num_embeddings") else m.token_emb.base.num_embeddings
    with torch.no_grad():
        D = m.action_to_lie(m.token_emb(torch.arange(V, device=m.out_proj.weight.device))[None])[0]
    return D.detach().clone().requires_grad_(True)       # (V, H, nb)


def slope(y):
    k = np.arange(len(y), dtype=float)
    return float(((k - k.mean()) * (y - y.mean())).sum() / ((k - k.mean()) ** 2).sum())


def path_maxima(m, D, x_list, info_list, sps_list, env, npts=2001):
    """Local maxima of the mean summed-head A_P at the answer along Delta(q_k): now -> -(k-1)s."""
    q0, k0 = origins(m)
    q0, k0 = q0.detach(), k0.detach()
    om = m.path_integrator.omega.detach()                       # (H, nb)
    s = D[env.sym_offset:env.sym_offset + env.n_symbols].mean(0)  # (H, nb)
    rest = {k: [] for k in KS}
    for x, infos, sps in zip(x_list, info_list, sps_list):
        S = torch.cumsum(D[x], dim=1)                            # (B, L, H, nb)
        for b in range(x.shape[0]):
            sympos = np.asarray(infos[b]["sym_positions"])
            for p, k in zip(sps[b], infos[b]["offsets"]):
                if p >= x.shape[1]:
                    continue
                a = sympos[sympos < p][-k]
                rest[k].append(S[b, p] - D[x[b, p]] - S[b, a])  # dS without the query's own step
    al = torch.linspace(0, 1, npts, device=D.device)
    scale = float((q0 * k0).sum().abs() / math.sqrt(q0.shape[-1]))   # kappa(0), summed heads
    out = {}
    for k in KS:
        if not rest[k]:
            continue
        R = torch.stack(rest[k])                                 # (nq, H, nb)
        d0 = D[env.query_offset + k - 1]
        path = d0[None] + al[:, None, None] * (-(k - 1) * s - d0)[None]   # (npts, H, nb)
        dS = R[None] + path[:, None]                             # (npts, nq, H, nb)
        ang = dS * om                                            # theta_p - theta_a
        qr = _apply_rope(q0[None, None].expand(npts, R.shape[0], -1, -1), torch.cos(ang), torch.sin(ang))
        f = (qr * k0).sum(-1).sum(-1).mean(1) / math.sqrt(q0.shape[-1])   # (npts,)
        f = f.cpu().numpy()
        nraw = int(((f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:])).sum())
        # DEVIATION from SEARCH_PREREG S2 (recorded in the results file): the registered
        # strict-interior count reads float noise on flat paths (k=1 at trained P0: f0 = f1 =
        # 0.993 and 143 "maxima"). Count only peaks with prominence >= 1% of the kernel peak.
        pk, _ = find_peaks(f, prominence=0.01 * scale)
        out[k] = dict(maxima=int(len(pk)), maxima_raw=nraw, f0=float(f[0]), f1=float(f[-1]),
                      fmax=float(f.max()), scale=scale, curve=f[::20].tolist())
    return out


def init_probe(arm, seed, dev):
    from mapformer.train_variant import VARIANT_MAP
    torch.manual_seed(seed); np.random.seed(seed)
    env = RecencyWorld(n_symbols=16, k_max=64, p_query=0.25, min_gap=None, seed=seed)
    m = VARIANT_MAP[arm](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2,
                         n_layers=1, grid_size=64).to(dev).eval()
    rng = np.random.RandomState(seed)
    pathway = [p for n_, p in m.named_parameters()
               if n_.startswith(("action_to_lie", "path_integrator")) or n_ in
               ("p0_pos", "q0_pos", "k0_pos", "k0_u", "k0_scale")]
    content = [p for n_, p in m.named_parameters() if n_.startswith("layers.0.")
               and not n_.startswith(("layers.0.norm",))]
    rates, coss, ratios, aps, axs = [], [], [], [], []
    xs, infs, spl = [], [], []
    for _ in range(NB):
        toks, sps, ans, infos = env.generate_batch(BS, T, rng)
        x = toks[:, :-1].to(dev)
        xs.append(x); infs.append(infos); spl.append(sps)
        # parameter pass
        m.zero_grad()
        o = em_forward(m, x)
        recency_loss(o["logits"], env, sps, ans).backward()
        gp = torch.sqrt(sum((p.grad ** 2).sum() for p in pathway if p.grad is not None))
        gc = torch.sqrt(sum((p.grad ** 2).sum() for p in content if p.grad is not None))
        ratios.append((gp / gc).item())
        causal = torch.tril(torch.ones(x.shape[1], x.shape[1], device=dev, dtype=torch.bool))
        aps.append(o["AP"][:, :, causal].pow(2).mean().sqrt().item())
        axs.append(o["AX"][:, :, causal].pow(2).mean().sqrt().item())
        # function-space pass
        D = delta_leaf(m)
        o = em_forward(m, x, delta=D[x])
        loss = recency_loss(o["logits"], env, sps, ans)
        J = 0.0
        for b in range(x.shape[0]):
            sympos = np.asarray(infos[b]["sym_positions"])
            for p, k in zip(sps[b], infos[b]["offsets"]):
                if p < x.shape[1]:
                    J = J + o["AP"][b, :, p, int(sympos[sympos < p][-k])].sum()
        gL, = torch.autograd.grad(loss, D, retain_graph=True)
        gJ, = torch.autograd.grad(J, D)
        g = (-gL).flatten(1).double(); gj = gJ.flatten(1).double()
        s = D.detach()[env.sym_offset:env.sym_offset + env.n_symbols].flatten(1).double().mean(0)
        qrows = [env.query_offset + k - 1 for k in KS]
        y = (g[qrows] @ s / (s @ s)).cpu().numpy()
        rates.append(slope(y))
        used = [r for r in qrows if gj[r].norm() > 0]
        coss.append(float(F.cosine_similarity(g[used], gj[used], dim=-1).mean()))
    with torch.no_grad():
        D = delta_leaf(m).detach()
        rug = path_maxima(m, D, xs, infs, spl, env)
    r = np.array(rates)
    return dict(arm=arm, seed=seed, rate=float(r.mean()), rate_t=float(r.mean() / (r.std(ddof=1) / math.sqrt(NB))),
                cos_ker=float(np.mean(coss)), ratio=float(np.mean(ratios)),
                AP_rms=float(np.mean(aps)), AX_rms=float(np.mean(axs)), rugged=rug)


@torch.no_grad()
def trained_rugged(dev):
    from mapformer.probe_rewind import load_model
    f = require_checkpoints("runs/dof/recency", "recency", ["VanillaEM_P0_r4"], range(8))
    out = []
    for (arm, s), pt in sorted(f.found.items()):
        m, ck, env = load_model(pt, arm)
        m = m.to(dev).eval()
        rng = np.random.RandomState(7000 + s)
        xs, infs, spl = [], [], []
        for _ in range(8):
            toks, sps, ans, infos = env.generate_batch(BS, T, rng)
            xs.append(toks[:, :-1].to(dev)); infs.append(infos); spl.append(sps)
        out.append(dict(arm=arm, seed=s, rugged=path_maxima(m, delta_leaf(m).detach(), xs, infs, spl, env)))
        print(f"trained {arm} s{s} done", flush=True)
    return out


def summarise(init_rows, trained_rows):
    from mapformer.stats_guard import from_diffs
    L = ["## S2  gradient at initialisation (8 batches, eval mode)\n",
         "| arm | seed | rate (slope of y_k) | t | cos(g, dJ) | pathway/content grad | rms A_P | rms A_X |",
         "|---|---|---|---|---|---|---|---|"]
    for r in init_rows:
        L.append(f"| {r['arm']} | {r['seed']} | {r['rate']:+.3e} | {r['rate_t']:+.2f} | {r['cos_ker']:+.3f} | "
                 f"{r['ratio']:.3e} | {r['AP_rms']:.2e} | {r['AX_rms']:.2e} |")
    for arm in ARMS:
        ts = [r["rate_t"] for r in init_rows if r["arm"] == arm]
        L.append(f"\n{arm}: |t| < 2 on {sum(abs(t) < 2 for t in ts)}/8 seeds (H-rugged (ii) needs >= 6/8 for P0); "
                 f"sign of rate negative on {sum(r['rate'] < 0 for r in init_rows if r['arm'] == arm)}/8")
    L.append("\n## Ruggedness: interior local maxima on the straight path to the linear rewind\n")
    L.append("Peaks with prominence >= 1% of kappa(0); the registered raw strict count in parentheses "
             "(it reads float noise on flat paths -- a recorded deviation).\n")
    L.append("| set | k<=4 median | k 8-16 median | k 32 median | k 60-64 median | f(1) > f(0) at k>=32 |")
    L.append("|---|---|---|---|---|---|")
    def row(name, rows):
        def med(ks):
            v = [r["rugged"][k]["maxima"] for r in rows for k in ks if k in r["rugged"]]
            return float(np.median(v)) if v else float("nan")
        def medraw(ks):
            v = [r["rugged"][k]["maxima_raw"] for r in rows for k in ks if k in r["rugged"]]
            return float(np.median(v)) if v else float("nan")
        up = [r["rugged"][k]["f1"] > r["rugged"][k]["f0"] for r in rows for k in range(32, 65) if k in r["rugged"]]
        return (f"| {name} | {med(range(1, 5))} ({medraw(range(1, 5))}) | {med(range(8, 17))} | "
                f"{med([32])} | {med(range(60, 65))} ({medraw(range(60, 65))}) | {np.mean(up):.3f} |"), \
            med(range(1, 5)), med(range(60, 65))
    verdicts = []
    for arm in ARMS:
        s_, a_, b_ = row(f"init {arm}", [r for r in init_rows if r["arm"] == arm]); L.append(s_)
        if arm == "VanillaEM_P0_r4":
            verdicts.append(("init P0", a_, b_))
    s_, a_, b_ = row("trained P0 (s0-7)", trained_rows); L.append(s_); verdicts.append(("trained P0", a_, b_))
    for name, a_, b_ in verdicts:
        L.append(f"\nH-rugged (i) {name}: k<=4 median {a_} (<= 1 needed), k 60-64 median {b_} (>= 5 needed) -> "
                 f"**{'MET' if a_ <= 1 and b_ >= 5 else 'NOT MET'}**")
    txt = "\n".join(L)
    print(txt)
    (REPO / "runs/search/S2_report.md").write_text(txt + "\n")


def main():
    dev = "cuda:1" if torch.cuda.is_available() else "cpu"
    init_rows = []
    for arm in ARMS:
        for s in range(8):
            init_rows.append(init_probe(arm, s, dev))
            print(f"init {arm} s{s}: rate {init_rows[-1]['rate']:+.3e} t {init_rows[-1]['rate_t']:+.2f}", flush=True)
    trained = trained_rugged(dev)
    json.dump(dict(init=init_rows, trained=trained), open(REPO / "_INIT_GRAD.json", "w"))
    summarise(init_rows, trained)


if __name__ == "__main__":
    main()
