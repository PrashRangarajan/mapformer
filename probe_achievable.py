"""T1's decisive test: for each query token, was a good rewind AVAILABLE in the model's own subspace?

`THEORY_SEARCH_AND_LENGTH.md` T1. A rewind for token `q_k` is a choice of `z in R^r` (r=4) with

    (W_out z)_i  ≡  -b_i   (mod 2*pi/omega_i)   for every block with amplitude a_i > 0,

`b` being everything else the accumulator collects between the answer and the query. That is up
to 64 congruences in 4 unknowns, so what matters is the BEST ACHIEVABLE weighted kernel value

    Q_k = max_z  E_episodes[ sum_i a_i cos(omega_i (b_i + (W_out z)_i)) ] / sum_i a_i

computed with the model's own trained `omega`, `W_out` and amplitudes, and with `b` sampled from
real episodes so filler noise is included (a rewind must work across episodes, not for one).
`A_k` is what the model actually achieves with its learned `Delta(q_k)`.

    Q_k high, A_k low  on a FAILED token -> the solution existed and was not found: SEARCH.
    Q_k low            on a FAILED token -> unsolvable in this subspace: EXISTENCE, per token.

Rule 29 at per-token grain. Also reports the dead-block fraction, which T1 claims is the search
strategy rather than an oddity.

    python3 -m mapformer.probe_achievable
"""
from __future__ import annotations

import json

import numpy as np
import torch

from mapformer.ckpt_guard import REPO, require_checkpoints
from mapformer.probe_rewind import delta_table, load_model, position_origins

N_EP, T, N_SAMP, N_START, N_STEP = 24, 1024, 128, 48, 250
DEAD = 1e-6                     # amplitude below this fraction of the max is a dead block


@torch.no_grad()
def collect_b(m, env, seed, n_ep=N_EP):
    """Per offset k, samples of b = S_query - S_answer - Delta(q_k) (the rest of the path)."""
    D = delta_table(m)                                   # (V, H*nb), float64
    rng = np.random.RandomState(9000 + seed)
    toks, sps, ans, infos = env.generate_batch(n_ep, T, rng)
    x = toks[:, :-1]
    out = {}
    for b_i in range(x.shape[0]):
        S = torch.cumsum(D[x[b_i]], dim=0)               # (L, H*nb)
        sym = np.asarray(infos[b_i]["sym_positions"])
        for p, k in zip(sps[b_i], infos[b_i]["offsets"]):
            if p >= x.shape[1]:
                continue
            a = sym[sym < p][-k]
            out.setdefault(k, []).append(S[p] - S[a] - D[x[b_i, p]])
    return {k: torch.stack(v) for k, v in out.items()}


def achievable(b, a, om, Wout, z0):
    """max over z of mean_samples sum_i a_i cos(om_i (b_i + (Wout z)_i)) / sum a_i."""
    dev = b.device
    z = torch.randn(N_START, Wout.shape[1], dtype=torch.float64, device=dev) * 2.0
    z[0] = z0                                            # the linear rewind, as one start
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    aw = a / a.sum()
    for _ in range(N_STEP):
        d = z @ Wout.T                                   # (starts, blocks)
        val = (aw * torch.cos(om * (b[:, None] + d[None]))).sum(-1).mean(0)
        loss = -val.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        d = z @ Wout.T
        val = (aw * torch.cos(om * (b[:, None] + d[None]))).sum(-1).mean(0)
    return float(val.max())


def run(arm="VanillaEM_P0_r4", runs_dir="runs/dof/recency", seeds=range(8), device="cpu"):
    f = require_checkpoints(runs_dir, "recency", [arm], list(seeds))
    rows = []
    for (a_, s), pt in sorted(f.found.items()):
        m, ck, env = load_model(pt, arm)
        p0, _, _ = position_origins(m)
        H, dh = p0.shape
        amp = p0.view(H, dh // 2, 2).double().pow(2).sum(-1).reshape(-1)      # (blocks,)
        om = m.path_integrator.omega.detach().double().reshape(-1)
        Wout = m.action_to_lie.w_out.weight.detach().double()                 # (blocks, r)
        D = delta_table(m)
        s_sym = D[env.sym_offset:env.sym_offset + env.n_symbols].mean(0)
        live = amp > DEAD * amp.max()
        bs = collect_b(m, env, s)
        # least-squares z for the linear rewind, as an optimiser start
        Q, A, ks = {}, {}, []
        for k in sorted(bs):
            samp = bs[k][:N_SAMP]
            tgt = -(k - 1) * s_sym
            z0 = torch.linalg.lstsq(Wout, tgt).solution
            Q[k] = achievable(samp[:, live], amp[live], om[live], Wout[live], z0)
            dq = D[env.query_offset + k - 1]
            aw = amp[live] / amp[live].sum()
            A[k] = float((aw * torch.cos(om[live] * (samp[:, live] + dq[live]))).sum(-1).mean())
            ks.append(k)
        rows.append(dict(arm=a_, seed=s, ks=ks, Q=[Q[k] for k in ks], A=[A[k] for k in ks],
                         dead_frac=float((~live).double().mean()), n_live=int(live.sum())))
        print(f"{a_} s{s}: dead {rows[-1]['dead_frac']:.2f} "
              f"meanQ {np.mean(rows[-1]['Q']):.3f} meanA {np.mean(rows[-1]['A']):.3f}", flush=True)
    return rows


def main():
    rows = run()
    acc = {r["seed"]: r["acc"] for r in json.load(open(REPO / "_ANATOMY_EXT.json"))
           if r["arm"] == "VanillaEM_P0_r4"}
    sol_Q, sol_A, fail_Q, fail_A, per_seed = [], [], [], [], []
    for r in rows:
        a_k = acc[r["seed"]]
        nsol = 0
        for k, q, av in zip(r["ks"], r["Q"], r["A"]):
            if k < 8 or k > len(a_k):
                continue
            if a_k[k - 1] >= 0.9:
                sol_Q.append(q); sol_A.append(av); nsol += 1
            elif a_k[k - 1] <= 0.3:
                fail_Q.append(q); fail_A.append(av)
        per_seed.append((r["dead_frac"], nsol))
    d, n = np.array(per_seed).T
    L = ["## T1 -- was a good rewind AVAILABLE? (single-p0 EM, k >= 8)\n",
         "`Q` = best achievable weighted kernel at the answer, optimising z in the model's own "
         "rank-4 subspace over real episode samples. `A` = what the model achieves. 1.0 is a "
         "perfect wrapped rewind.\n",
         "| cells | n | mean Q (available) | mean A (achieved) | gap |", "|---|---|---|---|---|",
         f"| SOLVED (acc >= 0.9) | {len(sol_Q)} | {np.mean(sol_Q):.3f} | {np.mean(sol_A):.3f} | "
         f"{np.mean(sol_Q) - np.mean(sol_A):+.3f} |",
         f"| FAILED (acc <= 0.3) | {len(fail_Q)} | {np.mean(fail_Q):.3f} | {np.mean(fail_A):.3f} | "
         f"{np.mean(fail_Q) - np.mean(fail_A):+.3f} |", ""]
    verdict = ("SEARCH -- the solution was available and not found"
               if np.mean(fail_Q) > 0.5 and np.mean(fail_A) < 0.3 else
               "EXISTENCE -- failed tokens had no good solution in this subspace"
               if np.mean(fail_Q) < 0.3 else "MIXED -- report both")
    L.append(f"**Verdict: {verdict}.** Failed tokens had Q = {np.mean(fail_Q):.3f} available and "
             f"achieved A = {np.mean(fail_A):.3f}.")
    r_ds = float(np.corrcoef(d, n)[0, 1]) if len(set(d)) > 1 else float("nan")
    L += ["", f"**P1a (pruning is the strategy)**: r(dead-block fraction, tokens solved) = "
              f"{r_ds:+.3f} over {len(d)} seeds; dead fraction {d.mean():.2f} +/- {d.std():.2f}, "
              f"solved {n.mean():.1f} +/- {n.std():.1f} of 57."]
    txt = "\n".join(L)
    print("\n" + txt)
    (REPO / "runs/search").mkdir(parents=True, exist_ok=True)
    (REPO / "runs/search/T1_ACHIEVABLE.md").write_text(txt + "\n")
    json.dump(rows, open(REPO / "_ACHIEVABLE.json", "w"))


if __name__ == "__main__":
    main()
