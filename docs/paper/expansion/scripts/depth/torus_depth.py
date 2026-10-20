"""POST HOC (CPU, eval-only): what does depth buy index attention on the torus at T = 1024? FACTORIAL checkpoints
(runs/factorial, seeds 500-503): RoPE 1 layer (L1), RoPE 4 layers (L4), RoPE one block looped x4 (LOOP), path MapWM r4
1 layer (PATH). Same stream as eval_rank_strata (env seed 10000 = held-out map, np.random.seed(1234 + s), 100 walks of
T = 1024); the 'all' accuracy is asserted against FACTORIAL_*_STRATA.json.

Per revisit target (observation after move t; predicted at input position 2t), with s = the last earlier visit to the
wrapped cell:
  stratum   wrap-only / short (gap < 128) / long (gap >= 128), as eval_rank_strata
  retrace   the current run reverses the previous run and has not yet overshot it (then s = t - 2 cur)
  nruns     number of action runs spanned by moves s+1..t (retrace: 2)
  gap bins  moves since s
Attention (per pass, per head, at query 2t): mass on earlier observation tokens of the same wrapped cell ('same'), mass
on the last visit's observation + arrival-action tokens ('last'), chance level of 'same' under uniform attention,
normalised entropy H / log(n_keys), and mean attention distance in moves.
Probes (walks 0-39, action positions 2t, t >= 16): ridge R^2 from each pass's residual stream for the unwrapped
displacement P_t (x, y), its torus phase (cos, sin of 2 pi P / 64 per axis), and the displacement over the last w moves
d_w(t) for w in 4, 16, 64, 256 (train walks 0-29, test 30-39).

  cd /home/prashr && python3 mapformer/docs/paper/expansion/scripts/depth/torus_depth.py ARM [n_walks]
writes out/torus_depth_ARM.json (checks, probes) and $DEPTH_SCRATCH/torus_depth_ARM.npz (per-revisit rows: correct,
labels, attention stats st[n, pass, head, (same, last, entropy, mean distance)])"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/depth")
from depthlib import forward_capture, check_logits, runs_info, ridge_r2, REPO, OUTDIR  # noqa: E402
from mapformer.environment import GridWorld  # noqa: E402
from mapformer.eval_rank_strata import kinds_vec  # noqa: E402
from mapformer.train_variant import VARIANT_MAP  # noqa: E402

SCRATCH = os.environ.get("DEPTH_SCRATCH", OUTDIR)
torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
ARMS = {"L1": ("L1", "RoPE", "FACTORIAL_L1_STRATA.json"), "L4": ("L4", "RoPE", "FACTORIAL_L4_STRATA.json"),
        "LOOP": ("LOOP", "RoPELooped", "FACTORIAL_LOOP_STRATA.json"),
        "PATH": ("L1", "Vanilla_r4", "FACTORIAL_L1_STRATA.json")}
SEEDS = (500, 501, 502, 503)
DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)
GAP_BINS = [(2, 8), (8, 32), (32, 128), (128, 512), (512, 1025)]
RUN_BINS = [(2, 3), (3, 5), (5, 9), (9, 17), (17, 10 ** 6)]
WINDOWS = (4, 16, 64, 256)
N_PROBE_TR, N_PROBE_TE = 30, 10


def load(arm, s):
    sub, v, _ = ARMS[arm]
    b = torch.load(f"{REPO}/runs/factorial/{sub}/p0/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
    c = b["config"]
    m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"],
                       n_layers=c["n_layers"], grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); m.eval()
    return m


def labels(tok, size=64):
    a = tok[0::2].numpy(); T = len(a)
    u = np.cumsum(DELTA[a], 0)
    new_u, lag = kinds_vec(tok, size)
    cur, prev, rid = runs_info(a, lambda x, y: x == (y ^ 1))
    retr = (prev > 0) & (cur <= prev)
    last = np.arange(T) - lag                                  # last visit (invalid where never visited)
    cell = (u[:, 0] % size) * size + (u[:, 1] % size)
    nruns = np.zeros(T, int)
    ok = lag < T
    s1 = np.clip(last + 1, 0, T - 1)
    nruns[ok] = rid[ok] - rid[s1[ok]] + 1
    return dict(u=u, wrap=new_u, lag=lag, retr=retr, last=last, cell=cell, nruns=nruns, cur=cur)


def main():
    arm = sys.argv[1]; n_walks = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ref = json.load(open(f"{REPO}/{ARMS[arm][2]}"))
    v = ARMS[arm][1]
    rows = []                         # per revisit: seed, correct, labels, attention stats per pass/head
    probe = {}
    checks = {}
    for s in SEEDS:
        t0 = time.time()
        m = load(arm, s)
        env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=10000)
        np.random.seed(1234 + s)
        hit = n = 0
        PX = []; PY = []
        for w in range(n_walks):
            tok, _o, rev = env.generate_trajectory(1024)
            x = tok[None, :-1]
            if w == 0:
                checks[s] = check_logits(m, x)
            logits, res, att = forward_capture(m, x, keep_attn=True)
            pred = logits[0].argmax(-1)
            lb = labels(tok)
            T = 1024; L = 2047
            r = rev[1::2].numpy()                                  # revisit flag of move t
            ts = np.nonzero(r)[0]
            corr = (pred[2 * ts] == tok[2 * ts + 1]).numpy()
            hit += int(corr.sum()); n += len(ts)
            # token -> cell (action 2t and obs 2t+1 both at cell(t))
            tcell = np.repeat(lb["cell"], 2)[:L]
            q = torch.as_tensor(2 * ts)
            kidx = torch.arange(L)
            same_obs = (torch.as_tensor(tcell)[None, :] == torch.as_tensor(lb["cell"][ts])[:, None]) & \
                       (kidx[None, :] < q[:, None]) & (kidx[None, :] % 2 == 1)
            lastv = torch.as_tensor(lb["last"][ts])
            n_keys = (q + 1).float()
            dist_moves = ((q[:, None] - kidx[None, :]).clamp_min(0).float() / 2)
            stats = []
            for A in att:                                          # (1, H, L, L)
                Aq = A[0][:, q, :]                                 # (H, nq, L)
                same = (Aq * same_obs[None]).sum(-1)
                lastm = Aq.gather(-1, (2 * lastv + 1).clamp(0, L - 1)[None, :, None].expand(Aq.shape[0], -1, 1))[..., 0] + \
                    Aq.gather(-1, (2 * lastv).clamp(0, L - 1)[None, :, None].expand(Aq.shape[0], -1, 1))[..., 0]
                ent = -(Aq.clamp_min(1e-30) * Aq.clamp_min(1e-30).log()).sum(-1) / n_keys.log()[None]
                md = (Aq * dist_moves[None]).sum(-1)
                stats.append(torch.stack([same, lastm, ent, md], -1).numpy())   # (H, nq, 4)
            chance = (same_obs.sum(-1).float() / n_keys).numpy()
            st = np.stack(stats, 0)                                # (P, H, nq, 4)
            rows.append(dict(seed=np.full(len(ts), s), c=corr, wrap=lb["wrap"][ts], lag=lb["lag"][ts],
                             retr=lb["retr"][ts], nruns=lb["nruns"][ts], chance=chance.astype(np.float32),
                             st=np.transpose(st, (2, 0, 1, 3)).astype(np.float32)))
            if w < N_PROBE_TR + N_PROBE_TE:
                tt = np.arange(16, T)
                u = lb["u"]
                Y = [u[tt, 0], u[tt, 1], np.cos(2 * np.pi * u[tt, 0] / 64), np.sin(2 * np.pi * u[tt, 0] / 64),
                     np.cos(2 * np.pi * u[tt, 1] / 64), np.sin(2 * np.pi * u[tt, 1] / 64)]
                for wd in WINDOWS:
                    prev_u = np.where((tt - wd)[:, None] >= 0, u[np.clip(tt - wd, 0, None)], 0)
                    Y += [u[tt, 0] - prev_u[:, 0], u[tt, 1] - prev_u[:, 1]]
                Y = np.stack(Y, 1)
                Xs = [rr[0, 2 * tt].numpy() for rr in res]
                (PX if w < N_PROBE_TR else PY).append((Xs, Y))
        acc = hit / n
        want = ref[f"{v}|{s}|1024"]["all"]["acc"]
        if n_walks == 100: assert abs(acc - want) < 2e-3, (arm, s, acc, want)
        checks[f"acc_{s}"] = (acc, want)
        # probes
        P = len(PX[0][0]) if PY else 0
        Ytr = Yte = None
        if PY: Ytr = np.concatenate([y for _, y in PX]); Yte = np.concatenate([y for _, y in PY])
        r2 = []
        for p in range(P):
            Xtr = np.concatenate([xs[p] for xs, _ in PX]); Xte = np.concatenate([xs[p] for xs, _ in PY])
            r2.append(ridge_r2(Xtr, Ytr, Xte, Yte).tolist())
        probe[s] = r2
        print(arm, s, f"acc {acc:.4f} (stored {want:.4f})", f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(dict(arm=arm, n_walks=n_walks, checks={str(k): v for k, v in checks.items()}, probe=probe,
                   probe_cols=["Px", "Py", "cosx", "sinx", "cosy", "siny"] + [f"d{w}{a}" for w in WINDOWS for a in "xy"]),
              open(f"{OUTDIR}/torus_depth_{arm}.json", "w"), indent=1)
    cat = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(f"{SCRATCH}/torus_depth_{arm}.npz", **cat)


if __name__ == "__main__":
    main()
