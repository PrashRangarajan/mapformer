"""How good is each model's POSITION CODE, read without its attention block?

The loop (L) and depth (L4) leave the step map's shape untouched (shared r=2, theta computed once from
the token embeddings, re-used every pass). So the question that splits "the loop found a better code"
(search on the step map) from "the loop compensates downstream for the same poor code" is: how much
cell identity does the learned phase theta carry by itself?

Reader: for a scored revisit query at input position i = 2t, score every earlier observation token
j = 2s+1 (s < t) by a phase-only kernel  k(i, j) = sum_{h,c} w_{hc} cos(theta_{hc}(i) - theta_{hc}(j))
(wrap-aware by construction), and call it a hit if the top-scoring key sits at the same wrapped cell.
  uniform : w = 1 on every channel
  fitted  : w >= 0 (softplus) and a temperature fitted by cross-entropy (log of the softmax mass on
            same-cell keys) on 24 TRAINING-seed trajectories (np seed 777 + s), scored on the 50
            held-out evaluation trajectories (np seed 1234 + s, the stored eval stream). This is the best
            content-free, one-pass, cosine reader of the code: an upper bound on what one attention pass
            can get from position alone (a real head also has content; it cannot see more position).
Also reported: the observation-token step (obs |delta| / action |delta|, the leak; obs tokens should not
move the phase) and the same-cell vs different-cell separation of the fitted kernel (AUC on pairs).

usage: python3 phase_code.py ARM
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import OUT, load, env_for, phases, step_info, stratum, trajectories  # noqa

torch.set_num_threads(4)
arm = sys.argv[1]
T = 1024; N_FIT = 24; N_EVAL = 50
STR = ("all", "short", "long", "wrap")


def prep(m, c, tok, rev):
    inp = tok[:-1]
    th = phases(m, inp)                                   # (H, L, nb)
    H, L, nb = th.shape
    feat_c = torch.cos(th).permute(1, 0, 2).reshape(L, H * nb)
    feat_s = torch.sin(th).permute(1, 0, 2).reshape(L, H * nb)
    msk = rev[1:]
    idx = torch.nonzero(msk).flatten().numpy()
    cell, new_u, lag = step_info(tok, c["grid_size"])
    st = idx // 2
    nstep = len(cell)
    kpos = 2 * np.arange(nstep) + 1
    kpos = kpos[kpos < L]
    nk = len(kpos)
    same = (cell[st][:, None] == cell[None, :nk]) & (np.arange(nk)[None, :] < st[:, None])
    valid = np.arange(nk)[None, :] < st[:, None]
    obs_t = tok[1:][idx].numpy(); obs_k = inp[kpos].numpy()
    return dict(qc=feat_c[idx], qs=feat_s[idx], kc=feat_c[kpos], ks=feat_s[kpos],
                same=torch.from_numpy(same), valid=torch.from_numpy(valid),
                strat=stratum(new_u[st], lag[st]), obs_t=obs_t, obs_k=obs_k)


def scores(d, w):
    return (d["qc"] * w) @ d["kc"].T + (d["qs"] * w) @ d["ks"].T


res = {}
t0 = time.time()
for s in range(8):
    m, c, _ = load(arm, s)
    env = env_for(c)
    with torch.no_grad():
        emb = m.token_emb.weight
        dl = m.action_to_lie(emb[None])[0].reshape(emb.shape[0], -1)   # (V, H*nb), omega not applied
        om = m.path_integrator.omega.reshape(-1)
        dth = (dl * om).norm(dim=-1)                                    # per-token |omega * delta|
    leak = float(dth[4:].mean() / dth[:4].mean())                      # obs tokens / action tokens
    fit = [prep(m, c, *x) for x in trajectories(env, T, N_FIT, 777 + s)]
    ev = [prep(m, c, *x) for x in trajectories(env, T, N_EVAL, 1234 + s)]
    nch = fit[0]["qc"].shape[1]
    raw = torch.zeros(nch, requires_grad=True); logt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([raw, logt], lr=0.05)
    for it in range(300):
        d = fit[it % N_FIT]
        w = torch.nn.functional.softplus(raw)
        sc = scores(d, w) * torch.exp(logt)
        sc = sc.masked_fill(~d["valid"], float("-inf"))
        lse_all = torch.logsumexp(sc, -1)
        lse_ok = torch.logsumexp(sc.masked_fill(~d["same"], float("-inf")), -1)
        loss = (lse_all - lse_ok).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    w_fit = torch.nn.functional.softplus(raw).detach()
    out = {"leak_obs_over_action": leak}
    for name, w in (("uniform", torch.ones(nch)), ("fitted", w_fit)):
        hits = {k: [0, 0] for k in STR}; ohits = {k: [0, 0] for k in STR}
        pos, neg = [], []
        with torch.no_grad():
            for d in ev:
                sc = scores(d, w).masked_fill(~d["valid"], float("-inf"))
                top = sc.argmax(-1)
                hit = d["same"].gather(1, top[:, None])[:, 0].numpy()
                ohit = d["obs_k"][top.numpy()] == d["obs_t"]
                for k in STR:
                    sel = np.ones(len(hit), bool) if k == "all" else d["strat"] == k
                    hits[k][0] += int(hit[sel].sum()); hits[k][1] += int(sel.sum())
                    ohits[k][0] += int(ohit[sel].sum()); ohits[k][1] += int(sel.sum())
                if name == "fitted":
                    v = sc[d["valid"]].numpy(); sm = d["same"][d["valid"]].numpy()
                    pos.append(v[sm]); neg.append(np.random.default_rng(0).choice(v[~sm], 4000))
        out[name] = {k: {"cell_top1": hits[k][0] / max(hits[k][1], 1),
                         "obs_top1": ohits[k][0] / max(ohits[k][1], 1), "n": hits[k][1]} for k in STR}
        if name == "fitted":
            p = np.concatenate(pos); n = np.concatenate(neg)
            ranks = np.argsort(np.argsort(np.concatenate([p, n])))
            auc = (ranks[:len(p)].sum() - len(p) * (len(p) - 1) / 2) / (len(p) * len(n))
            out["fitted_auc_same_vs_diff"] = float(auc)
            out["fitted_temperature"] = float(torch.exp(logt))
            ww = w_fit.reshape(c["n_heads"], -1)
            out["fitted_weight_share_by_head"] = (ww.sum(1) / ww.sum()).tolist()
    res[s] = out
    print(f"{arm} s{s} {time.time()-t0:.0f}s leak {leak:.3f} | uniform cell " +
          " ".join(f"{k} {out['uniform'][k]['cell_top1']:.3f}" for k in STR) + " | fitted cell " +
          " ".join(f"{k} {out['fitted'][k]['cell_top1']:.3f}" for k in STR) +
          f" | auc {out['fitted_auc_same_vs_diff']:.3f}", flush=True)
json.dump(res, open(f"{OUT}/phase_code_{arm}.json", "w"), indent=1)
