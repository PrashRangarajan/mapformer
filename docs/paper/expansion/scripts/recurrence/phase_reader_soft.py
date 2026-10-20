"""A stronger one-pass, position-only reader of each model's phase code: an idealised single attention
head whose ONLY per-position information is the model's own theta.

  score(i, j) = sum_c w_c cos(theta_c(i) + phi^q_{a_i, c} - theta_c(j) - phi^k_{o_j, c}) + b_{o_j}
  p(o | i)    = sum_j softmax_j(score) [o_j = o]          (soft vote over earlier observation tokens)

w_c >= 0 (softplus; also sets the temperature), per-(query action, channel) and per-(key observation,
channel) phase offsets (a one-layer head can rotate q and k by token-type-dependent angles, e.g. to
cancel a constant per-observation phase step), and a per-observation key bias. Fitted by cross-entropy
on the observation at scored revisits of 24 training-seed trajectories (np seed 777 + s); scored on the
50 evaluation trajectories (np seed 1234 + s). Accuracy = argmax_o p(o | i) == target, by stratum.

A one-layer model's query at an action token depends on the action identity alone and its keys on the
token identity alone, so this reader has the same information a single pass has (it lacks the second
head and the MLP). If a model scores far above its own code's soft reader, its extra accuracy comes from
computation downstream of the code (compensation), not from the code.

usage: python3 phase_reader_soft.py ARM
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
    ang = th.permute(1, 0, 2).reshape(L, H * nb)
    idx = torch.nonzero(rev[1:]).flatten().numpy()
    cell, new_u, lag = step_info(tok, c["grid_size"])
    st = idx // 2
    kpos = 2 * np.arange(len(cell)) + 1
    kpos = kpos[kpos < L]
    valid = np.arange(len(kpos))[None, :] < st[:, None]
    return dict(qa=ang[idx], qtok=inp[idx], ka=ang[kpos], ktok=inp[kpos],
                valid=torch.from_numpy(valid), tgt=tok[1:][idx], strat=stratum(new_u[st], lag[st]))


class Reader(torch.nn.Module):
    def __init__(self, V, nch):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.zeros(nch))
        self.phq = torch.nn.Parameter(torch.zeros(V, nch))
        self.phk = torch.nn.Parameter(torch.zeros(V, nch))
        self.b = torch.nn.Parameter(torch.zeros(V))
        self.V = V

    def forward(self, d):
        w = torch.nn.functional.softplus(self.raw) * 2.0
        aq = d["qa"] + self.phq[d["qtok"]]
        ak = d["ka"] + self.phk[d["ktok"]]
        sc = (torch.cos(aq) * w) @ torch.cos(ak).T + (torch.sin(aq) * w) @ torch.sin(ak).T
        sc = sc + self.b[d["ktok"]][None]
        sc = sc.masked_fill(~d["valid"], float("-inf"))
        a = torch.softmax(sc, -1)
        onehot = torch.nn.functional.one_hot(d["ktok"], self.V).float()
        return (a @ onehot).clamp_min(1e-9)               # (Q, V) vote


res = {}
t0 = time.time()
for s in range(8):
    m, c, _ = load(arm, s)
    env = env_for(c)
    fit = [prep(m, c, *x) for x in trajectories(env, T, N_FIT, 777 + s)]
    ev = [prep(m, c, *x) for x in trajectories(env, T, N_EVAL, 1234 + s)]
    torch.manual_seed(0)
    R = Reader(c["vocab_size"], fit[0]["qa"].shape[1])
    opt = torch.optim.Adam(R.parameters(), lr=0.03)
    for it in range(600):
        d = fit[it % N_FIT]
        p = R(d)
        loss = -torch.log(p.gather(1, d["tgt"][:, None])).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    hits = {k: [0, 0] for k in STR}
    with torch.no_grad():
        for d in ev:
            h = (R(d).argmax(-1) == d["tgt"]).numpy()
            for k in STR:
                sel = np.ones(len(h), bool) if k == "all" else d["strat"] == k
                hits[k][0] += int(h[sel].sum()); hits[k][1] += int(sel.sum())
    res[s] = {k: hits[k][0] / max(hits[k][1], 1) for k in STR}
    res[s]["final_fit_loss"] = float(loss)
    print(f"{arm} s{s} {time.time()-t0:.0f}s soft reader " +
          " ".join(f"{k} {res[s][k]:.3f}" for k in STR) + f" fit loss {float(loss):.3f}", flush=True)
json.dump(res, open(f"{OUT}/phase_reader_soft_{arm}.json", "w"), indent=1)
