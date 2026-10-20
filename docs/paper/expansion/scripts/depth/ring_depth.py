"""POST HOC (CPU, eval-only): what do 1 / 2 / 3 index-RoPE layers compute on the biased 1D ring (CANCEL batch,
runs/cancel; 32 cells, T = 128, maps redrawn per walk)? Arms RoPE_L1/L2/L3 and Vanilla_L1 (path), p_plus 0.5 / 0.75 /
0.9 / 1.0, seeds 0-7 (0-1 are the pilot; read 2-7). Stream = train_cancel.evaluate exactly (GridWorldCancel(32, seed
10000, p_plus), np.random.seed(0), 200 walks), so the overall accuracy is asserted equal to each run's eval.json "128".

Per revisit (observation after move t, predicted at input position 2t), s = last earlier visit to the cell:
  gap        moves since s
  wrap-only  every earlier visit to the cell is a lap away in unwrapped position (ring_strata definition)
  retrace    the current run reverses the previous run and has not overshot it (s = t - 2 cur)
  lap        the walk's displacement over the last 32 moves is 0 mod 32 (a fixed 32-move lag lands on the cell)
  nruns      action runs spanned by moves s+1..t
Attention per pass and head at query 2t: mass on earlier observation tokens of the same cell, mass on the last visit's
tokens, normalised entropy, mean distance (moves). Probes (ridge, walks 0-149 train / 150-199 test, action positions
t >= 8): unwrapped displacement P_t, cos/sin(2 pi P_t / 32), P_t / (t + 1), 1 / (t + 1), d_w for w = 4, 16, 64.

  cd /home/prashr && python3 mapformer/docs/paper/expansion/scripts/depth/ring_depth.py
writes out/ring_depth.json (checks, probes) and $DEPTH_SCRATCH/ring_depth.npz (per-revisit rows)"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import forward_capture, check_logits, runs_info, ridge_r2, REPO, OUTDIR  # noqa: E402
from mapformer.environment_cancel import GridWorldCancel  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402

SCRATCH = os.environ.get("DEPTH_SCRATCH", OUTDIR)
torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
SIZE, T, NW = 32, 128, 200
ARMS = [("RoPE", 1), ("RoPE", 2), ("RoPE", 3), ("Vanilla", 1)]
PS = (0.5, 0.75, 0.9, 1.0)
WINDOWS = (4, 16, 64)


def labels(moves):
    step = np.where(moves == 0, 1, -1)
    u = np.cumsum(step); Tn = len(u)
    seen = {}; lastt = {}
    rev = np.zeros(Tn, bool); gap = np.full(Tn, 10 ** 6); wo = np.zeros(Tn, bool); last = np.full(Tn, -1)
    for t in range(Tn):
        c = int(u[t] % SIZE)
        if c in seen:
            rev[t] = True; gap[t] = t - lastt[c]; last[t] = lastt[c]
            wo[t] = all(p != u[t] for p in seen[c]); seen[c].append(int(u[t]))
        else:
            seen[c] = [int(u[t])]
        lastt[c] = t
    cur, prev, rid = runs_info(moves, lambda x, y: x != y)
    retr = (prev > 0) & (cur <= prev)
    lap = np.zeros(Tn, bool)
    lap[32:] = ((u[32:] - u[:-32]) % SIZE) == 0
    nruns = np.zeros(Tn, int)
    ok = last >= 0
    nruns[ok] = rid[ok] - rid[np.clip(last[ok] + 1, 0, Tn - 1)] + 1
    return dict(u=u, rev=rev, gap=gap, wo=wo, last=last, retr=retr, lap=lap, nruns=nruns, cell=u % SIZE)


def main():
    rows = []; probes = {}; checks = {}
    for p in PS:
        for v, nl in ARMS:
            for d in sorted(glob.glob(f"{REPO}/runs/cancel/p0/{v}_L{nl}_p{p}_s[0-9]")):
                s = int(d.rsplit("_s", 1)[1])
                b = torch.load(f"{d}/{v}.pt", map_location="cpu", weights_only=False)
                m = VARIANT_MAP[v](vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=nl,
                                   grid_size=SIZE)
                m.load_state_dict(b["model_state_dict"]); m.eval()
                want = json.load(open(f"{d}/eval.json"))["eval"]["128"]["acc"]
                te = GridWorldCancel(size=SIZE, seed=10000, p_plus=p); np.random.seed(0)
                toks = []; L = []
                for _ in range(NW):
                    tok, _o, rev = te.generate_trajectory(T)
                    lb = labels((tok[0::2] - te.action_offset).numpy())
                    assert (lb["rev"] == rev[1::2].numpy()).all()
                    toks.append(tok); L.append(lb)
                X = torch.stack(toks)[:, :-1]
                key = f"{v}_L{nl}|{p}|{s}"
                checks[key] = check_logits(m, X[:20])
                logits, res, att = forward_capture(m, X)
                pred = logits.argmax(-1)
                hit = n = 0
                Ls = 2 * T - 1; kidx = torch.arange(Ls)
                for i, lb in enumerate(L):
                    ts = np.nonzero(lb["rev"])[0]
                    if len(ts) == 0:
                        continue
                    corr = (pred[i, 2 * ts] == toks[i][2 * ts + 1]).numpy()
                    hit += int(corr.sum()); n += len(ts)
                    q = torch.as_tensor(2 * ts)
                    tcell = torch.as_tensor(np.repeat(lb["cell"], 2)[:Ls])
                    same_obs = (tcell[None] == torch.as_tensor(lb["cell"][ts])[:, None]) & (kidx[None] < q[:, None]) \
                        & (kidx[None] % 2 == 1)
                    lv = torch.as_tensor(lb["last"][ts])
                    nk = (q + 1).float(); dm = (q[:, None] - kidx[None]).clamp_min(0).float() / 2
                    st = []
                    for A in att:
                        Aq = A[i][:, q, :]
                        same = (Aq * same_obs[None]).sum(-1)
                        lm = Aq.gather(-1, (2 * lv + 1)[None, :, None].expand(Aq.shape[0], -1, 1))[..., 0] + \
                            Aq.gather(-1, (2 * lv)[None, :, None].expand(Aq.shape[0], -1, 1))[..., 0]
                        ent = -(Aq.clamp_min(1e-30) * Aq.clamp_min(1e-30).log()).sum(-1) / nk.log()[None]
                        md = (Aq * dm[None]).sum(-1)
                        st.append(torch.stack([same, lm, ent, md], -1).numpy())
                    st = np.stack(st, 0)                                   # (P, H, nq, 4)
                    stp = np.zeros((len(ts), 3, 2, 4), np.float32)          # pad passes to 3
                    stp[:, :st.shape[0]] = np.transpose(st, (2, 0, 1, 3))
                    rows.append(dict(arm=np.full(len(ts), f"{v}_L{nl}"), p=np.full(len(ts), p), seed=np.full(len(ts), s),
                                     c=corr, gap=lb["gap"][ts], wo=lb["wo"][ts], retr=lb["retr"][ts], lap=lb["lap"][ts],
                                     nruns=lb["nruns"][ts], chance=(same_obs.sum(-1).float() / nk).numpy(), st=stp))
                acc = hit / n
                assert abs(acc - want) < 2e-3, (key, acc, want)
                checks[key + "|acc"] = (acc, want)
                # probes
                tt = np.arange(8, T)
                Y = []
                for lb in L:
                    u = lb["u"]; cols = [u[tt], np.cos(2 * np.pi * u[tt] / SIZE), np.sin(2 * np.pi * u[tt] / SIZE),
                                         u[tt] / (tt + 1), 1.0 / (tt + 1)]
                    for w in WINDOWS:
                        cols.append(u[tt] - np.where(tt - w >= 0, u[np.clip(tt - w, 0, None)], 0))
                    Y.append(np.stack(cols, 1))
                Y = np.stack(Y)                                            # (NW, nt, ncol)
                r2 = []
                for r in res:
                    Xr = r[:, 2 * tt].numpy()
                    r2.append(ridge_r2(Xr[:150].reshape(-1, 128), Y[:150].reshape(-1, Y.shape[-1]),
                                       Xr[150:].reshape(-1, 128), Y[150:].reshape(-1, Y.shape[-1])).tolist())
                probes[key] = r2
                print(key, f"acc {acc:.4f} (stored {want:.4f})", flush=True)
    json.dump(dict(checks=checks, probes=probes,
                   probe_cols=["P", "cosP", "sinP", "P/t", "1/t"] + [f"d{w}" for w in WINDOWS]),
              open(f"{OUTDIR}/ring_depth.json", "w"), indent=1)
    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(f"{SCRATCH}/ring_depth.npz", **cat)


if __name__ == "__main__":
    main()
