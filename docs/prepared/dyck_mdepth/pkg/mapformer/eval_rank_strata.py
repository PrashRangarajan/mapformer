"""Revisit accuracy and NLL by KIND of revisit, for the rank matched-length test.

The +0.085 that RANK_SWEEP reports for r=4 over r=2 at T=1024 was measured after
training at T=128. A pre-launch audit (2026-09-23) split it three ways on the old
checkpoints and found 94% of it in ONE stratum:

  plain_lag<128   a revisit whose gap (steps since the last visit to the cell) was
                  possible at T=128, but which happens later in the sequence than
                  training ever reached. r=2 0.903 vs r=4 0.999.
  plain_lag>=128  a revisit with a gap longer than any T=128 training walk.
  wrap            a revisit reachable only by going round the torus (the unwrapped
                  position is new, the wrapped cell is not). Never occurs at T=128;
                  both arms sit below the 0.512 always-blank floor.

Training at T=1024 puts all three in distribution, so this is the readout that says
WHICH part of the old effect survives. Scoring, trajectories and the held-out map
are identical to eval_noise_refine (seed 1234+s, env seed 10000), and the 'all'
column reproduces it exactly -- the script asserts that when --check-json is given.

Checkpoints at <runs-dir>/p0/<V>_s<S>/<V>.pt (the ckpt_guard layout).
"""
import argparse, json, os

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP

STRATA = ("all", "plain_lag<128", "plain_lag>=128", "wrap")


def kinds(tok, deltas, size):
    """Per step: (unwrapped position is new, gap since last visit to the wrapped cell)."""
    a = tok[0::2].numpy()
    u = np.cumsum(np.array([deltas[int(x)] for x in a]), 0)
    su, last, out = set(), {}, []
    for t in range(len(a)):
        ku, kw = tuple(u[t]), tuple(u[t] % size)
        out.append((ku not in su, t - last.get(kw, -10**9)))
        su.add(ku); last[kw] = t
    return out


_DELTA_ARR = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)  # == ACTION_DELTAS


def kinds_vec(tok, size):
    """Vectorised kinds(): (unwrapped position is new, lag since last visit to the wrapped
    cell, or t + 10**9 when never visited) for every step, as two arrays."""
    a = tok[0::2].numpy()
    u = np.cumsum(_DELTA_ARR[a], 0)                           # unwrapped position after step t
    n = len(a)
    off = n + 1                                               # |u| <= n: make keys non-negative
    ku = (u[:, 0] + off) * (2 * off + 1) + (u[:, 1] + off)
    new_u = np.zeros(n, dtype=bool)
    new_u[np.unique(ku, return_index=True)[1]] = True         # first occurrence = not seen before
    kw = (u[:, 0] % size) * size + (u[:, 1] % size)
    order = np.argsort(kw, kind="stable")                     # time order within each cell
    prev = np.full(n, -10**9, dtype=np.int64)
    same = kw[order[1:]] == kw[order[:-1]]
    prev[order[1:][same]] = order[:-1][same]
    return new_u, np.arange(n) - prev


@torch.no_grad()
def evaluate(m, env, T, n_trials, seed, dev):
    """Same trajectories, same B=1 forward and same outputs as the per-target loop it
    replaces (audit 2026-09-24): hits and counts are integers; NLL is summed in float64
    in the SAME ORDER (np.cumsum is a strict left-to-right accumulate), so the JSON is
    byte-identical. ~20-40 us of Python per scored target becomes a few array ops."""
    np.random.seed(seed)
    hits = {k: 0 for k in STRATA}; cnt = {k: 0 for k in STRATA}
    nls = {k: [] for k in STRATA}; tg = {k: [] for k in STRATA}
    for _ in range(n_trials):
        tok, _o, rev = env.generate_trajectory(T)
        lp = F.log_softmax(m(tok[None, :-1].to(dev)).float(), -1)[0].cpu()
        tgt = tok[1:]; msk = rev[1:]
        idx = torch.nonzero(msk).flatten()
        if idx.numel() == 0:
            continue
        t_i = tgt[idx]
        hit = (lp.argmax(-1)[idx] == t_i).numpy()
        nl = (-lp[idx, t_i]).double().numpy()
        new_u, lag = kinds_vec(tok, env.size)
        st = (idx // 2).numpy()                               # target i is the obs of step i//2
        wrap = new_u[st]; short = lag[st] < 128
        t_np = t_i.numpy()
        for k, sel in (("all", np.ones(len(st), dtype=bool)), ("wrap", wrap),
                       ("plain_lag<128", ~wrap & short), ("plain_lag>=128", ~wrap & ~short)):
            hits[k] += int(hit[sel].sum()); cnt[k] += int(sel.sum())
            nls[k].append(nl[sel]); tg[k].append(t_np[sel])
    out = {}
    for k in STRATA:
        n = cnt[k]
        # a leading 0.0 reproduces Python's `0.0 + x` exactly, signed zeros included
        s = float(np.cumsum(np.concatenate([np.zeros(1)] + nls[k]))[-1])
        fl = int(np.bincount(np.concatenate(tg[k])).max()) if n else 0
        # floor = best CONSTANT prediction within the stratum (model-independent;
        # it is the always-blank rate wherever blank is the commonest target)
        out[k] = {"acc": hits[k] / n if n else None, "nll": s / n if n else None, "n": n,
                  "floor": fl / n if n else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--variants", nargs="+", default=["Vanilla", "Vanilla_r4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--lengths", nargs="+", type=int, default=[1024])
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--env-seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="JSON path")
    ap.add_argument("--check-json", default=None,
                    help="an eval_noise_refine JSON for the same runs; 'all' must match it")
    a = ap.parse_args()
    dev = torch.device(a.device)
    ref = json.load(open(a.check_json)) if a.check_json else None
    res, missing = {}, []
    for v in a.variants:
        for s in a.seeds:
            ck = os.path.join(a.runs_dir, "p0", f"{v}_s{s}", f"{v}.pt")
            if not os.path.exists(ck):
                missing.append(ck); continue
            b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]
            m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"],
                               n_heads=c["n_heads"], n_layers=c["n_layers"],
                               grid_size=c["grid_size"])
            m.load_state_dict(b["model_state_dict"]); m = m.to(dev).eval()
            env = GridWorld(size=c["grid_size"], n_obs_types=c.get("n_obs_types", 16),
                            p_empty=c.get("p_empty", 0.5), seed=a.env_seed)
            for T in a.lengths:
                r = evaluate(m, env, T, a.n_trials, 1234 + s, dev)
                res[f"{v}|{s}|{T}"] = r
                if ref is not None and f"0.0|{v}|{T}" in ref:
                    want = {x[0]: x[1] for x in ref[f"0.0|{v}|{T}"]}[s]
                    assert abs(want - r["all"]["acc"]) < 1e-3, (v, s, T, want, r["all"]["acc"])
                print(v, s, T, {k: (round(x["acc"], 4) if x["acc"] is not None else None, x["n"])
                                for k, x in r.items()}, flush=True)
            del m; torch.cuda.empty_cache()
    if missing:
        print("MISSING", len(missing), missing[:4], flush=True)
        if not res:
            raise SystemExit("no checkpoints found under " + a.runs_dir)
    json.dump(res, open(a.out, "w"), indent=1)
    for T in a.lengths:
        for k in STRATA:
            line = [f"T={T} {k:15s}"]
            for v in a.variants:
                x = [res[f"{v}|{s}|{T}"][k]["acc"] for s in a.seeds if f"{v}|{s}|{T}" in res]
                x = [y for y in x if y is not None]
                if x:
                    line.append(f"{v} {np.mean(x):.3f} sd {np.std(x, ddof=1) if len(x) > 1 else 0:.3f}")
            fl = [r[k]["floor"] for kk, r in res.items() if kk.endswith(f"|{T}") and r[k]["floor"] is not None]
            if fl:
                line.append(f"floor {np.mean(fl):.3f}")
            print("  ".join(line), flush=True)


if __name__ == "__main__":
    main()
