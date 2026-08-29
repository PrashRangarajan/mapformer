"""Pre-flight task-validity gates for the MiniWorld cross-cell-revisit task.

CPU only, no training, no GPU. Run BEFORE spending compute (project rule:
validate the task first). MiniWorld is headless via pyglet EGL and slow, so we
use few episodes (~30-50) at the OOD length.

The task under test (miniworld_env.py, allocentric=False by default):
  An agent takes macro-actions in continuous 3D MiniWorld -- turn 15 deg, or
  forward-until-the-discretized-(x,z)-grid-cell-changes. Each grid cell has a
  FIXED random observation token (obs_map, ~half blank at p_empty=0.5). The
  interleaved token stream is [a0, o0, a1, o1, ...]. The SCORED target is the
  observation token at CROSS-CELL REVISITS: an obs position whose cell was seen
  before AND is not the immediately-previous cell (rev_mask in the env).

Gates (a gate that sits at chance / adds no info is a PASS):
  G1 chance         1 / n_obs_types (= 1/16 = 0.0625). Reference floor.
  G2 marginal       always predict the most-common obs class on the scored
                    subset. Elevated by the ~41% blank mass -- reported with and
                    without blank.
  G3 copy-last-obs  predict the PREVIOUS step's obs at each scored position.
  G4 action n-gram  orders 1..5 over the ACTION tokens alone predicting the
                    scored obs. THE LOAD-BEARING GATE: obs is random per seed and
                    location-determined, so recent actions must not predict it.
                    A PASS means n-gram does NOT beat the marginal.
  G5 label mass     fraction of steps scored, and scored-count per trajectory.
  G6 revisit lag    steps back to the matching prior visit at each scored pos;
                    is it long-range, or solvable in a bounded 8/16/32 window?
  G7 oracle         obs recomputed from the true (x,z)-cell via obs_map must
                    equal the stored scored token on 100% of scored positions
                    (target is deterministic and answerable from position).
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

import pyglet
pyglet.options["headless"] = True          # EGL; must precede miniworld import
from mapformer.miniworld_env import MiniWorldWorld


def ngram_eval(contexts_answers, orders, fallback):
    """contexts_answers: list of (context_tuple, answer). Train on first half,
    test on second half; unseen context -> fallback (global majority)."""
    n = len(contexts_answers)
    half = n // 2
    out = {}
    for K in orders:
        tab = defaultdict(Counter)
        for i in range(half):
            ctx, ans = contexts_answers[i]
            tab[ctx[-K:] if K <= len(ctx) else None][ans] += 1
        pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
        ok = tot = 0
        for i in range(half, n):
            ctx, ans = contexts_answers[i]
            key = ctx[-K:] if K <= len(ctx) else None
            ok += int(pred.get(key, fallback) == ans)
            tot += 1
        out[K] = ok / max(tot, 1)
    return out


def run_config(grid_size, T, n_episodes, n_obs, seed, env_name, fixed_map=False,
               allocentric=False, oracle=False):
    w = MiniWorldWorld(env_name=env_name, grid_size=grid_size,
                       n_obs_types=n_obs, seed=seed, allocentric=allocentric,
                       fixed_map=fixed_map, oracle=oracle)
    rng = np.random.RandomState(seed)
    blank = w.blank_token
    obs_off = w.obs_offset
    # G8 support: the widest token id the env ever emits, checked against the
    # vocab the trainer will size the embedding to. An out-of-range id surfaces
    # as CUBLAS_STATUS_ALLOC_FAILED, which reads exactly like CUDA OOM -- three
    # conditions once passed every answer-stream gate while being untrainable.
    tok_min, tok_max = 10**9, -10**9

    answers = []               # obs class at scored positions (may be blank)
    prev_obs = []              # obs class at step i-1 (copy-last prediction)
    action_ctx = []            # (action-context tuple, answer) for n-gram
    lags = []                  # steps back to matching prior visit
    scored_counts = []         # scored positions per episode
    oracle_hits = oracle_tot = 0

    for _ in range(n_episodes):
        tok, om_mask, rev = w.generate_trajectory(T, rng=rng)
        # read the map for THIS episode AFTER generation: fresh_map redraws
        # self.obs_map every episode, so a map cached before the loop would be
        # stale and the oracle would (wrongly) fail on fresh-map.
        obs_map = w.obs_map
        tok = tok.numpy()
        tok_min = min(tok_min, int(tok.min())); tok_max = max(tok_max, int(tok.max()))
        rev = rev.numpy()
        cells = w.visited_locations            # T cells, one per step
        actions = tok[0::2]                    # T action tokens
        obs_tok = tok[1::2]                    # T obs tokens
        obs_cls = obs_tok - obs_off            # obs class in [0, n_obs] (blank=n_obs)

        scored_steps = [i for i in range(T) if rev[2 * i + 1]]
        scored_counts.append(len(scored_steps))

        last_step = {}
        for i in range(T):
            c = cells[i]
            if rev[2 * i + 1]:                 # scored cross-cell revisit
                # oracle: recompute the answer purely from position + obs_map
                gt = int(obs_map[c[0], c[1]])
                oracle_hits += int(gt == int(obs_cls[i]))
                oracle_tot += 1
                answers.append(int(obs_cls[i]))
                prev_obs.append(int(obs_cls[i - 1]) if i > 0 else -1)
                ctx = tuple(int(a) for a in actions[max(0, i - 4):i + 1])
                action_ctx.append((ctx, int(obs_cls[i])))
                if c in last_step:             # prior visit (always true here)
                    lags.append(i - last_step[c])
            last_step[c] = i                   # update AFTER, so it holds prior visit

    answers = np.array(answers)
    n = len(answers)
    K = n_obs

    # G1
    g1 = 1.0 / K
    # G2 marginal (incl blank) + on non-blank subset
    cnt = Counter(answers.tolist())
    maj_cls, maj_ct = cnt.most_common(1)[0]
    g2_all = maj_ct / max(n, 1)
    nonblank = answers[answers != blank]
    n_nb = len(nonblank)
    if n_nb:
        cnt_nb = Counter(nonblank.tolist())
        g2_nb = cnt_nb.most_common(1)[0][1] / n_nb
    else:
        g2_nb = float("nan")
    blank_frac = float((answers == blank).mean()) if n else float("nan")
    # G3 copy-last-obs
    po = np.array(prev_obs)
    valid = po >= 0
    g3 = float((po[valid] == answers[valid]).mean()) if valid.any() else float("nan")
    # G4 action n-gram (full scored set, incl blank) and non-blank
    g4 = ngram_eval(action_ctx, (1, 2, 3, 4, 5), fallback=maj_cls)
    ac_nb = [(c, a) for (c, a) in action_ctx if a != blank]
    fb_nb = Counter([a for _, a in ac_nb]).most_common(1)[0][0] if ac_nb else 0
    g4_nb = ngram_eval(ac_nb, (1, 3, 5), fallback=fb_nb)
    # G5 label mass
    g5 = n / max(n_episodes * T, 1)
    scored_counts = np.array(scored_counts)
    # G6 lag
    lags = np.array(lags) if lags else np.array([0])
    within = {win: float((lags <= win).mean()) for win in (8, 16, 32, 64, 128)}
    # G7 oracle
    g7 = oracle_hits / max(oracle_tot, 1)

    return dict(
        grid_size=grid_size, T=T, n_episodes=n_episodes, n_scored=int(n),
        n_obs=n_obs, vocab_size=int(w.unified_vocab_size),
        tok_min=int(tok_min), tok_max=int(tok_max),
        chance=g1, marginal_all=g2_all, marginal_nonblank=g2_nb,
        blank_frac=blank_frac, copy_last=g3, ngram=g4, ngram_nonblank=g4_nb,
        label_mass=g5, scored_per_traj_mean=float(scored_counts.mean()),
        scored_per_traj_min=int(scored_counts.min()),
        scored_per_traj_max=int(scored_counts.max()),
        lag_median=float(np.median(lags)), lag_mean=float(lags.mean()),
        lag_p90=float(np.percentile(lags, 90)), lag_max=int(lags.max()),
        lag_within=within, oracle=g7, oracle_tot=int(oracle_tot),
    )


def verdict_lines(r):
    """PASS/FAIL per gate for one config."""
    g = []
    ch = r["chance"]
    # G1: reference, always PASS
    g.append(("G1 chance (ref)", "PASS", f"{ch:.4f} = 1/{r['n_obs']}"))
    # G2: marginal is expected to be elevated by blank; report, not a fail on its
    # own, but flag if non-blank marginal is far above chance (skew).
    g2nb = r["marginal_nonblank"]
    g2v = "PASS" if (np.isnan(g2nb) or g2nb < 3 * ch) else "WARN"
    g.append(("G2 marginal", g2v,
              f"all={r['marginal_all']:.3f} (blank {r['blank_frac']:.2f}), "
              f"non-blank={g2nb:.3f} vs chance {ch:.3f}"))
    # G3 copy-last: informational; FAIL only if it alone solves the task (high)
    g3 = r["copy_last"]
    g3v = "PASS" if g3 < max(0.30, r["marginal_all"] + 0.03) else "WARN"
    g.append(("G3 copy-last-obs", g3v, f"{g3:.3f}"))
    # G4 action n-gram: LOAD-BEARING. PASS iff best order does not beat marginal.
    best = max(r["ngram"].values())
    best_nb = max(r["ngram_nonblank"].values())
    leak = best - r["marginal_all"]
    leak_nb = best_nb - (r["marginal_nonblank"] if not np.isnan(r["marginal_nonblank"]) else ch)
    g4v = "PASS" if (leak < 0.03 and leak_nb < 0.03) else "FAIL"
    g.append(("G4 action n-gram", g4v,
              "o1..5 " + "/".join(f"{r['ngram'][k]:.3f}" for k in (1, 2, 3, 4, 5))
              + f" (best {best:.3f} vs marg {r['marginal_all']:.3f}, "
              f"non-blank best {best_nb:.3f} vs {r['marginal_nonblank']:.3f})"))
    # G5 label mass: PASS if enough scored positions to train (>~6% and >=10/traj)
    lm = r["label_mass"]
    g5v = "PASS" if (lm > 0.06 and r["scored_per_traj_mean"] >= 10) else "WARN"
    g.append(("G5 label mass", g5v,
              f"{lm:.3f} of steps scored; {r['scored_per_traj_mean']:.1f}/traj "
              f"[{r['scored_per_traj_min']}..{r['scored_per_traj_max']}]"))
    # G6 revisit lag: PASS (long-range) if a meaningful fraction exceeds a 32-win
    w32 = r["lag_within"][32]
    g6v = "PASS" if w32 < 0.85 else "WARN"
    g.append(("G6 revisit lag", g6v,
              f"median={r['lag_median']:.0f} p90={r['lag_p90']:.0f} "
              f"max={r['lag_max']}; within8/16/32={r['lag_within'][8]:.2f}/"
              f"{r['lag_within'][16]:.2f}/{w32:.2f}"))
    # G7 oracle: MUST be 1.0
    g7 = r["oracle"]
    g7v = "PASS" if g7 >= 0.999 else "FAIL"
    g.append(("G7 oracle", g7v, f"{g7:.4f} over {r['oracle_tot']} scored"))
    g.append(_g8(r))
    return g


def _g8(r):
    """Token ids must lie inside the vocab the trainer sizes the embedding to.

    Added 2026-08-28. Three continuous conditions once passed EVERY answer-stream
    gate while being untrainable: an out-of-range embedding lookup raises
    CUBLAS_STATUS_ALLOC_FAILED, which is indistinguishable from CUDA OOM at the
    console. Cheap to check, and it fails loudly instead of at 3am on a GPU.
    """
    ok = 0 <= r["tok_min"] and r["tok_max"] < r["vocab_size"]
    return ("G8 vocab range", "PASS" if ok else "FAIL",
            f"tokens [{r['tok_min']}, {r['tok_max']}] vs vocab {r['vocab_size']}"
            + ("" if ok else "  <-- OUT OF RANGE: would crash as a fake CUDA OOM"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", type=int, default=[8, 6])
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-name", default="MiniWorld-OneRoom-v0")
    ap.add_argument("--fixed-map", action="store_true")
    ap.add_argument("--allocentric", action="store_true",
                    help="validate the ALLOCENTRIC (displacement-direction) action "
                         "stream -- the load-bearing G4 n-gram gate must PASS on "
                         "this stream too, since the last few allo tokens form a "
                         "truncated path that a shallow n-gram could localise from")
    ap.add_argument("--oracle", action="store_true",
                    help="validate the ORACLE exact-cell-transition stream (9 "
                         "classes) -- n-gram must stay at chance: exact relative "
                         "displacement gives only RELATIVE position, and the fresh "
                         "obs_map is uncorrelated with the action prefix")
    ap.add_argument("--out", default="MINIWORLD_GATES.md")
    args = ap.parse_args()

    results = []
    for gs in args.grids:
        print(f"\n=== grid_size={gs} T={args.T} n_episodes={args.n_episodes} "
              f"fixed_map={args.fixed_map} allocentric={args.allocentric} ===",
              flush=True)
        r = run_config(gs, args.T, args.n_episodes, args.n_obs, args.seed,
                       args.env_name, fixed_map=args.fixed_map,
                       allocentric=args.allocentric, oracle=args.oracle)
        results.append(r)
        for name, v, detail in verdict_lines(r):
            print(f"  [{v:4s}] {name:18s} {detail}", flush=True)

    lines = ["# MiniWorld cross-cell-revisit task -- pre-flight gates (CPU, no training)",
             "",
             f"Env `{args.env_name}`, allocentric=False (raw 3-action macros), "
             f"n_obs={args.n_obs} (chance 1/16 = {1.0/args.n_obs:.4f}), "
             f"p_empty=0.5. {args.n_episodes} episodes/config at T={args.T}.",
             "",
             "Scored target = obs token at CROSS-CELL revisits (cell seen before "
             "and != previous cell).",
             ""]
    for r in results:
        lines.append(f"## grid_size={r['grid_size']}  (T={r['T']}, "
                     f"{r['n_scored']} scored positions)")
        lines.append("")
        lines.append("| gate | verdict | measured |")
        lines.append("|---|---|---|")
        for name, v, detail in verdict_lines(r):
            lines.append(f"| {name} | {v} | {detail} |")
        lines.append("")
    open(args.out, "w").write("\n".join(lines) + "\n")
    json.dump(results, open(args.out.replace(".md", ".json"), "w"),
              indent=2, default=str)
    print("\n" + "\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
