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


@torch.no_grad()
def evaluate(m, env, T, n_trials, seed, dev):
    np.random.seed(seed)
    acc = {k: [0, 0, 0.0] for k in STRATA}          # hits, n, summed nll
    freq = {k: {} for k in STRATA}                    # target counts -> constant floor
    for _ in range(n_trials):
        tok, _o, rev = env.generate_trajectory(T)
        lp = F.log_softmax(m(tok[None, :-1].to(dev)).float(), -1)[0].cpu()
        tgt = tok[1:]; msk = rev[1:]
        if msk.sum() == 0:
            continue
        pred = lp.argmax(-1); c = kinds(tok, env.ACTION_DELTAS, env.size)
        for i in torch.nonzero(msk).flatten().tolist():
            wrapped, lag = c[i // 2]                 # target i is the obs of step i//2
            k = "wrap" if wrapped else ("plain_lag<128" if lag < 128 else "plain_lag>=128")
            hit = int(pred[i] == tgt[i]); nl = float(-lp[i, tgt[i]])
            for kk in ("all", k):
                acc[kk][0] += hit; acc[kk][1] += 1; acc[kk][2] += nl
                y = int(tgt[i]); freq[kk][y] = freq[kk].get(y, 0) + 1
    # floor = best CONSTANT prediction within the stratum (model-independent;
    # it is the always-blank rate wherever blank is the commonest target)
    return {k: {"acc": h / n if n else None, "nll": s / n if n else None, "n": n,
                "floor": max(freq[k].values()) / n if n else None}
            for k, (h, n, s) in acc.items()}


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
