"""Per-pass (per-layer) revisit accuracy and attention, by revisit stratum, at T=1024.

For L (one block looped) the readout after pass j is exactly the model run with n_loops=j (the
eval-time loop-count sweep the LOOP_RANK pre-registration listed as a free secondary); passes 5-8
are beyond training. For L4 the readout after layer j is a logit lens (out_norm + out_proj applied to
an intermediate residual; the head was trained only on layer 4). A and C have one pass.

Attention, per head, for each scored revisit query (input position 2t, the action token that predicts
o_t): mass on earlier OBSERVATION tokens at the same cell, mass on earlier ACTION tokens that arrived
at the same cell (these hold the observation only if an earlier pass copied it there: a relay), and
whether the single most-attended key is a same-cell token.

usage: python3 pass_strata.py ARM [n_trials] [max_passes]
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/recurrence")
from common import (OUT, load, env_for, forward_passes, check_forward, step_info, stratum,  # noqa
                    trajectories)

torch.set_num_threads(4)
arm = sys.argv[1]
n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100
max_passes = int(sys.argv[3]) if len(sys.argv) > 3 else (8 if arm == "L" else None)
T = 1024
STR = ("all", "short", "long", "wrap")
res = {}
t0 = time.time()
for s in range(8):
    m, c, _ = load(arm, s)
    env = env_for(c)
    trs = trajectories(env, T, n_trials, 1234 + s)
    check_forward(m, arm, trs[0][0][:-1])
    acc = None
    for tok, rev in trs:
        inp = tok[:-1]
        lg, at = forward_passes(m, arm, inp, n_passes=max_passes)
        P = len(lg)
        if acc is None:
            acc = {k: np.zeros((P, 2)) for k in STR}
            att = {k: np.zeros((P, 2, 4)) for k in STR}   # obs mass, act mass, argmax hit, count
        tgt = tok[1:]; msk = rev[1:]
        idx = torch.nonzero(msk).flatten().numpy()
        if len(idx) == 0:
            continue
        cell, new_u, lag = step_info(tok, c["grid_size"])
        st = idx // 2
        strat = stratum(new_u[st], lag[st])
        nstep = len(cell)
        # same-cell key masks for every scored query: obs key 2s+1, action key 2s, s < t
        same = cell[st][:, None] == cell[None, :nstep]                     # (Q, nstep)
        earlier = np.arange(nstep)[None, :] < st[:, None]
        same &= earlier
        L = len(inp)
        obs_mask = np.zeros((len(idx), L), dtype=bool)
        act_mask = np.zeros((len(idx), L), dtype=bool)
        cols = np.arange(nstep)
        oc = 2 * cols + 1; ac = 2 * cols
        ok_o = oc < L
        obs_mask[:, oc[ok_o]] = same[:, ok_o]
        act_mask[:, ac] = same
        om = torch.from_numpy(obs_mask); am = torch.from_numpy(act_mask)
        anym = om | am
        for p in range(P):
            hit = (lg[p].argmax(-1)[idx] == tgt[idx]).numpy()
            A = at[p][:, idx, :]                                            # (H, Q, L)
            mo = (A * om[None]).sum(-1).numpy()                             # (H, Q)
            ma = (A * am[None]).sum(-1).numpy()
            amax = A.argmax(-1)                                             # (H, Q)
            hitk = anym.gather(1, amax.T).T.numpy()                         # (H, Q)
            for k in STR:
                sel = np.ones(len(idx), bool) if k == "all" else (strat == k)
                if not sel.any():
                    continue
                acc[k][p] += [hit[sel].sum(), sel.sum()]
                for h in range(A.shape[0]):
                    att[k][p, h] += [mo[h, sel].sum(), ma[h, sel].sum(), hitk[h, sel].sum(), sel.sum()]
    r = {}
    for k in STR:
        r[k] = {"acc": (acc[k][:, 0] / acc[k][:, 1]).tolist(), "n": int(acc[k][0, 1]),
                "obs_mass": (att[k][:, :, 0] / att[k][:, :, 3]).tolist(),
                "act_mass": (att[k][:, :, 1] / att[k][:, :, 3]).tolist(),
                "argmax_same_cell": (att[k][:, :, 2] / att[k][:, :, 3]).tolist()}
    res[s] = r
    print(f"{arm} s{s} {time.time()-t0:.0f}s  acc by pass " +
          " ".join(f"{x:.3f}" for x in r["all"]["acc"]) +
          "  wrap " + " ".join(f"{x:.3f}" for x in r["wrap"]["acc"]), flush=True)
json.dump(res, open(f"{OUT}/pass_strata_{arm}.json", "w"), indent=1)
