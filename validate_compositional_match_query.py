"""Pre-flight gates for Compositional Match-Query. CPU only. Run BEFORE training.

Same discipline as validate_match_query.py, but every baseline is split by the
two scored categories, because the whole point is that they dissociate:

  exact  answerable by path-integration matching alone (ordinary Match-Query).
  cross  answerable ONLY by path-integration AND motif abstraction together.

Gates (a gate at chance is a PASS):
  G1 chance         1/n_obs_types.
  G2 marginal       always predict the most common answer (per category).
  G3 answer n-gram  order 1..5 over the answer stream (per category). HIGH RISK:
                    a repeated cell would make the answer repeat. Dedup should
                    hold both categories at chance.
  G4 never-moved    predict the end-of-explore observation every time.
  G5 label mass     answerable rate + per-category counts. Need enough `cross`.
  G6 consistency    cross answers must equal the remembered motif value (0 fails).
  G7 oracle         true position + true map => 1.0 for BOTH categories (by
                    construction; asserted, not learned).
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_compositional_match_query import CompositionalMatchQueryGridWorld


def _ngram(ans, orders=(1, 2, 3, 5)):
    n = len(ans)
    out = {}
    if n < 8:
        return {k: float("nan") for k in orders}
    cnt = Counter(ans)
    fb = cnt.most_common(1)[0][0]
    half = n // 2
    for K in orders:
        tab = defaultdict(Counter)
        for i in range(K, half):
            tab[tuple(ans[i - K:i])][ans[i]] += 1
        pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
        ok = sum(int(pred.get(tuple(ans[i - K:i]), fb) == ans[i])
                 for i in range(half + K, n))
        out[K] = ok / max(n - half - K, 1)
    return out


def run(TE, TQ, n_episodes, size, room_size, n_obs, n_templates, seed):
    env = CompositionalMatchQueryGridWorld(size=size, room_size=room_size,
                                           n_obs_types=n_obs, n_templates=n_templates,
                                           seed=10000)
    rng = np.random.RandomState(seed)
    by = {"exact": {"ans": [], "nm": [], "nm_g": []},
          "cross": {"ans": [], "nm": [], "nm_g": []}}
    n_steps_tot = n_scored_tot = consistency_fail = 0
    for _ in range(n_episodes):
        _t, _rv, sp, ans, cats, info = env.generate_cmq_episode(TE, TQ, rng)
        n_steps_tot += TQ
        n_scored_tot += len(ans)
        consistency_fail += info["consistency_fail"]
        ex, ey = info["end_explore_pos"]
        g = int(info["obs_map"][ex, ey])                 # never-moved guess
        g_tok = g + env.obs_offset
        g_ok = g != env.blank_token
        for a, c in zip(ans, cats):
            by[c]["ans"].append(a)
            if g_ok:
                by[c]["nm"].append(g_tok); by[c]["nm_g"].append(a)

    K = env.n_obs_types
    rows = {}
    for c in ("exact", "cross"):
        ans = by[c]["ans"]
        n = len(ans)
        if n == 0:
            rows[c] = dict(n=0)
            continue
        cnt = Counter(ans)
        marginal = cnt.most_common(1)[0][1] / n
        ng = _ngram(ans)
        nm = (sum(int(a == b) for a, b in zip(by[c]["nm_g"], by[c]["nm"]))
              / max(len(by[c]["nm"]), 1)) if by[c]["nm"] else float("nan")
        rows[c] = dict(n=n, chance=1.0 / K, marginal=marginal, ngram=ng,
                       never_moved=nm)
    rows["_meta"] = dict(TE=TE, TQ=TQ, answerable=n_scored_tot / max(n_steps_tot, 1),
                         n_scored=n_scored_tot, consistency_fail=consistency_fail,
                         chance=1.0 / K)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-explore", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--t-query", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--room-size", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--n-templates", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="COMPOSITIONAL_MATCH_QUERY_GATES.md")
    args = ap.parse_args()

    all_rows = []
    for TE in args.t_explore:
        for TQ in args.t_query:
            r = run(TE, TQ, args.n_episodes, args.size, args.room_size,
                    args.n_obs, args.n_templates, args.seed)
            all_rows.append(r)
            m = r["_meta"]
            for c in ("exact", "cross"):
                rc = r[c]
                if rc["n"] == 0:
                    print(f"TE={TE} TQ={TQ} {c}: NO LABELS"); continue
                print(f"TE={TE:4d} TQ={TQ:3d} {c:5s} n={rc['n']:6d} "
                      f"chance={rc['chance']:.4f} marginal={rc['marginal']:.4f} "
                      f"ng1={rc['ngram'][1]:.4f} ng3={rc['ngram'][3]:.4f} "
                      f"never_moved={rc['never_moved']:.4f}", flush=True)
            print(f"          answerable={m['answerable']:.3f} "
                  f"consistency_fail={m['consistency_fail']} "
                  f"(oracle=1.000 both by construction)")

    lines = ["# Compositional Match-Query -- pre-flight gates (CPU, no training)", "",
             "Blind continuation in a repeated-motif world. `exact` = exact cell "
             "seen in explore (path-integration matching). `cross` = exact cell "
             "NOT seen, room explored + motif seen in another copy "
             "(path-integration AND motif abstraction). "
             f"Non-blank only; chance = 1/{args.n_obs} = {1.0/args.n_obs:.4f}. "
             "Oracle = 1.000 for both by construction.", "",
             "| TE | TQ | cat | chance | marginal | ng1 | ng3 | ng5 | never-moved | n |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in all_rows:
        m = r["_meta"]
        for c in ("exact", "cross"):
            rc = r[c]
            if rc["n"] == 0:
                continue
            lines.append(f"| {m['TE']} | {m['TQ']} | {c} | {rc['chance']:.4f} | "
                         f"{rc['marginal']:.4f} | {rc['ngram'][1]:.4f} | "
                         f"{rc['ngram'][3]:.4f} | {rc['ngram'][5]:.4f} | "
                         f"{rc['never_moved']:.4f} | {rc['n']} |")
    lines += ["",
              "| TE | TQ | answerable rate | consistency fails |",
              "|---|---|---|---|"]
    for r in all_rows:
        m = r["_meta"]
        lines.append(f"| {m['TE']} | {m['TQ']} | {m['answerable']:.3f} | "
                     f"{m['consistency_fail']} |")
    lines += ["", "**Reading it.** Every baseline column should sit at chance "
              "(0.0625) for BOTH categories. `cross` marginal/n-gram above chance "
              "would mean the compositional answer is guessable without the map. "
              "`consistency fails` must be 0 (else the cross target is ill-defined). "
              "`answerable rate` sets the effective sample size."]
    open(args.out, "w").write("\n".join(lines) + "\n")
    json.dump(all_rows, open(args.out.replace(".md", ".json"), "w"),
              indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
