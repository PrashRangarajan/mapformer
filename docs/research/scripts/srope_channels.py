"""CPU, post hoc, eval-only, on the stored SROPE_TORUS checkpoints (runs/srope_torus, seeds 1700-1707). Per rotary channel
(head x block, 64 per model) of the layer-1 angle, on 4 held-out-map eval walks (env seed 10000, np seed 1234 + seed, T = 1024):
  res_c     resolution: rms of wrap(theta_t - theta_s) over random observation pairs (a channel that barely moves is 'clean'
            for free, so cleanliness is read beside it);
  err_c[st] same-cell phase error: rms of wrap(theta_t - theta_s) over revisit pairs (t, s) = (observation at a revisit, the
            last observation at the same wrapped cell), by stratum short-gap (< 128 moves) / long-gap / wrap-only (the cell was
            seen before only from another unwrapped position; kinds_vec's definition). A map channel has err ~ 0 in every
            stratum; a clock channel's err grows with the gap; a non-periodic channel is clean on plain revisits and fails
            on wrap-only ones;
  per_c     the 64-move periodicity residual of the tick rule (Prop. 2b with g = N = 64 for a lap): |wrap(64 * a_c(move))|,
            mean over the four moves, from the (action k, observation k) pair increments as analyze_srope_torus does.
Summary per arm (mean over seeds): channels with res > 1 rad ('resolving'); among them the share clean (err / res < 0.25) on
short-gap, long-gap and wrap-only revisits; and the periodicity residual of the resolving channels. Written 2026-10-29 for
docs/research/SCOPE_EXPANSION_2026-10-29.md; not registered; reads checkpoints only."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "/home/prashr")
from mapformer.train_variant import VARIANT_MAP
from mapformer.environment import GridWorld
from mapformer.eval_rank_strata import kinds_vec
from mapformer.analyze_srope_torus import layer1_angles, wrap

REPO = "/home/prashr/mapformer"; OUT = os.path.join(REPO, "docs/research/scripts/out")
ARMS = {"SRoPE": "SRoPE", "SRoPE_NoSiLU": "SRoPE_NoSiLU", "SRoPE_ST": "SRoPE_ST", "SRoPEGen": "SRoPEGen", "MapWM": "Signed_r4"}
SEEDS = list(range(1700, 1708)); N = 64; T = 1024; WALKS = 4


def load(arm, s):
    V = ARMS[arm]
    b = torch.load(f"{REPO}/runs/srope_torus/{arm}/p0/{V}_s{s}/{V}.pt", map_location="cpu", weights_only=False)
    c = b["config"]
    m = VARIANT_MAP[V](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                       grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); m.eval()
    return m


@torch.no_grad()
def readout(m, s):
    env = GridWorld(size=N, n_obs_types=16, p_empty=0.5, seed=10000)
    np.random.seed(1234 + s)
    toks = [env.generate_trajectory(T)[0] for _ in range(WALKS)]
    th = layer1_angles(m, torch.stack(toks))                              # (B, 2T, C)
    C = th.shape[-1]
    same = {"short": [], "long": [], "wrap": []}; rnd = []; pinc = []
    rng = np.random.default_rng(s)
    for b, tok in enumerate(toks):
        new_u, lag = kinds_vec(tok, N)
        a = tok[0::2].numpy()
        ob = th[b, 1::2]                                                   # theta at observation k (T, C)
        seen_w = {}
        for t in range(T):
            kw = int(tok[1::2][t]) * 0  # placeholder (unused)
        # last visit index per step from lag (lag = t - prev, prev = -1e9 when first)
        prev = np.arange(T) - lag
        rev = prev >= 0
        for t in np.nonzero(rev)[0]:
            st = "wrap" if new_u[t] else ("short" if lag[t] < 128 else "long")
            same[st].append(wrap(ob[t] - ob[prev[t]]))
        ii = rng.integers(0, T, 400); jj = rng.integers(0, T, 400)
        rnd.append(wrap(ob[ii] - ob[jj]))
        pi = torch.cat([ob[:1], ob[1:] - ob[:-1]], 0)                     # pair increment keyed by action k
        pinc.append(torch.stack([pi[torch.as_tensor(a == k)].mean(0) for k in range(4)]))
    res = torch.cat(rnd).pow(2).mean(0).sqrt()                            # (C,)
    err = {k: torch.stack(v).pow(2).mean(0).sqrt() if v else torch.full((C,), float("nan")) for k, v in same.items()}
    P = torch.stack(pinc).mean(0)                                         # (4, C) mean pair increment per move
    per = wrap(64.0 * P).abs().mean(0)                                    # (C,) lap-periodicity residual
    amove = wrap(P).abs().mean(0)
    return dict(res=res.numpy(), err={k: v.numpy() for k, v in err.items()}, per=per.numpy(), amove=amove.numpy(), C=C)


def main():
    rows = {}
    print("arm | seed | resolving ch (res>1) | clean short / long / wrap (share of resolving) | periodicity residual (resolving, mean rad) | "
          "clean-wrap & resolving count | mean |a_move| of clean-wrap resolving ch")
    for arm in ARMS:
        agg = []
        for s in SEEDS:
            r = readout(load(arm, s), s)
            res, per = r["res"], r["per"]; sel = res > 1.0
            clean = {k: (r["err"][k] / np.maximum(res, 1e-6) < 0.25) for k in ("short", "long", "wrap")}
            row = dict(seed=s, n_res=int(sel.sum()),
                       clean_short=float(clean["short"][sel].mean()) if sel.any() else float("nan"),
                       clean_long=float(clean["long"][sel].mean()) if sel.any() else float("nan"),
                       clean_wrap=float(clean["wrap"][sel].mean()) if sel.any() else float("nan"),
                       per_res=float(per[sel].mean()) if sel.any() else float("nan"),
                       n_clean_wrap_res=int((clean["wrap"] & sel).sum()),
                       amove_cw=float(r["amove"][clean["wrap"] & sel].mean()) if (clean["wrap"] & sel).any() else float("nan"),
                       per_all=float(per.mean()), res_median=float(np.median(res)))
            # which channel indices (within a head of 32) are clean on wrap: index = temperature rung for SRoPE
            idx = np.nonzero(clean["wrap"] & sel)[0] % 32
            row["clean_wrap_rungs"] = sorted(set(int(i) for i in idx))
            agg.append(row)
            print(f"{arm:12s} | {s} | {row['n_res']:2d} | {row['clean_short']:.2f} / {row['clean_long']:.2f} / {row['clean_wrap']:.2f} | "
                  f"{row['per_res']:.2f} | {row['n_clean_wrap_res']:2d} | {row['amove_cw']:.3f} | rungs {row['clean_wrap_rungs'][:12]}")
        rows[arm] = agg
        mean = {k: float(np.nanmean([a[k] for a in agg])) for k in ("n_res", "clean_short", "clean_long", "clean_wrap", "per_res",
                                                                     "n_clean_wrap_res", "amove_cw", "res_median")}
        print(f"{arm:12s} | MEAN | {mean['n_res']:.1f} | {mean['clean_short']:.2f} / {mean['clean_long']:.2f} / {mean['clean_wrap']:.2f} | "
              f"{mean['per_res']:.2f} | {mean['n_clean_wrap_res']:.1f} | {mean['amove_cw']:.3f} | res median {mean['res_median']:.2f}")
    json.dump(rows, open(os.path.join(OUT, "srope_channels.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
