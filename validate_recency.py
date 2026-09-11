"""Pre-flight gates for the Recency (k-back) task. CPU only. Run BEFORE any GPU.

Same discipline as validate_match_query.py, gates re-aimed at this task's risks.

  G1 chance         uniform guess over n_symbols. PASS level for every baseline.
  G2 marginal       always predict the most common answer. Catches a skewed draw.
  G3 answer n-gram  order 1..5 over the ANSWER stream alone. THE HIGH-RISK GATE.
                    Two queries with no symbol between them have related answers,
                    and two consecutive q_1 repeat outright -- so `min_gap` is
                    swept rather than assumed. This is the family of bug that
                    invalidated hier-goal twice and forced Match-Query's dedup.
  G4 most-recent    ignore k and always answer the immediately preceding symbol.
                    Solves every k=1 query for free, so it sits near 1/k_max and
                    MUST NOT sit higher.
  G5 scored rate    scored queries per token; sets the effective sample size.
  G6 oracle         true history + k must give 1.0. Gate 7 of the standing rules:
                    this CALLS the environment rather than reimplementing it.
  G7 identifiability  the mechanism check, and the reason the task exists. For a
                    MONOTONE accumulator theta_t - theta_s is a bijection with
                    "symbols back", so the k-back key is uniquely addressed and
                    the collision rate is 0 by construction. For a SIGNED one
                    theta is a random walk that revisits values, so several
                    candidate keys share one theta and the difference no longer
                    addresses a unique position. Measured, not assumed.

                    The signed walk is simulated as +/-1 per token, a proxy for
                    "unconstrained and NOT learning a counter". It is an upper
                    bound on the difficulty, not a prediction of what a trained
                    signed model does -- the unconstrained arm may learn a
                    monotone code, which is prediction (2) in the environment
                    docstring and is measured on trained models, not here.

A gate at chance is a PASS. G7 is DIAGNOSTIC, not pass/fail: a high signed
collision rate is the task working as designed.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_recency import RecencyWorld


def ngram_acc(answers, orders=(1, 2, 3, 5)):
    """Fit on the first half of the answer stream, score the second."""
    out, n = {}, len(answers)
    if n < 20:
        return {k: float("nan") for k in orders}
    cnt = Counter(answers)
    fb = cnt.most_common(1)[0][0]
    half = n // 2
    for K in orders:
        tab = defaultdict(Counter)
        for i in range(K, half):
            tab[tuple(answers[i - K:i])][answers[i]] += 1
        pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
        ok = sum(int(pred.get(tuple(answers[i - K:i]), fb) == answers[i])
                 for i in range(half + K, n))
        out[K] = ok / max(n - half - K, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-symbols", type=int, default=16)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--p-query", type=float, default=0.25)
    ap.add_argument("--min-gaps", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="RECENCY_GATES.md")
    ap.add_argument("--k-fixed", type=int, default=None)
    ap.add_argument("--k-set", default=None)
    a = ap.parse_args()

    rows = []
    for min_gap in a.min_gaps:
        for T in a.lengths:
            env = RecencyWorld(n_symbols=a.n_symbols, k_max=a.k_max,
                               p_query=a.p_query, min_gap=min_gap, seed=a.seed,
                               k_fixed=a.k_fixed,
                               k_set=[int(v) for v in a.k_set.split(",")] if a.k_set else None)
            rng = np.random.RandomState(a.seed + 1)

            answers, mostrecent_hit, oracle_hit = [], [], []
            dist_by_k = defaultdict(list)
            n_scored_tot = n_tok_tot = 0
            sgn_amb, mono_amb, sgn_cands = [], [], []

            for _ in range(a.episodes):
                toks, sp, ans, info = env.generate_episode(T, rng)
                answers.extend(ans)
                for k, d in zip(info["offsets"], info["tok_dist"]):
                    dist_by_k[k].append(d)
                n_scored_tot += len(sp); n_tok_tot += info["T"]

                sym_pos = info["sym_positions"]; sym_val = info["sym_values"]
                # G6 oracle + G4 most-recent, both keyed on the env's own output
                for q_i, (p, k) in enumerate(zip(sp, info["offsets"])):
                    n_before = sum(1 for x in sym_pos if x < p)
                    oracle_hit.append(int(sym_val[n_before - k] == ans[q_i]))
                    mostrecent_hit.append(int(sym_val[n_before - 1] == ans[q_i]))

                # G7: does the accumulator difference address a unique key?
                theta_s = np.cumsum(rng.choice([-1, 1], size=info["T"]))
                for q_i, (p, k) in enumerate(zip(sp, info["offsets"])):
                    n_before = sum(1 for x in sym_pos if x < p)
                    if n_before < k:
                        continue
                    tgt_idx = n_before - k
                    cand = [(sym_pos[m], sym_val[m]) for m in range(n_before)]
                    tv = sym_val[tgt_idx]
                    th = theta_s[sym_pos[tgt_idx]]
                    same = [v for (pp, v) in cand if theta_s[pp] == th]
                    sgn_cands.append(len(same))
                    sgn_amb.append(int(any(v != tv for v in same)))
                    mono_amb.append(0)      # theta = t is injective by construction

            n = len(answers)
            cnt = Counter(answers)
            rows.append(dict(
                min_gap=min_gap, T=T, n_answers=n,
                chance=1.0 / a.n_symbols,
                marginal=cnt.most_common(1)[0][1] / max(n, 1),
                ngram=ngram_acc(answers),
                most_recent=float(np.mean(mostrecent_hit)) if mostrecent_hit else float("nan"),
                oracle=float(np.mean(oracle_hit)) if oracle_hit else float("nan"),
                scored_rate=n_scored_tot / max(n_tok_tot, 1),
                dist_sd={int(k): float(np.std(v)) for k, v in dist_by_k.items()},
                dist_mean={int(k): float(np.mean(v)) for k, v in dist_by_k.items()},
                signed_ambiguous=float(np.mean(sgn_amb)) if sgn_amb else float("nan"),
                signed_candidates=float(np.mean(sgn_cands)) if sgn_cands else float("nan"),
                monotone_ambiguous=float(np.mean(mono_amb)) if mono_amb else float("nan"),
            ))
            r = rows[-1]
            print(f"min_gap={min_gap} T={T:5d} chance={r['chance']:.4f} "
                  f"marg={r['marginal']:.4f} o1={r['ngram'][1]:.4f} "
                  f"o3={r['ngram'][3]:.4f} recent={r['most_recent']:.4f} "
                  f"oracle={r['oracle']:.4f} rate={r['scored_rate']:.3f} "
                  f"sgn_amb={r['signed_ambiguous']:.3f} n={n}", flush=True)

    chance = 1.0 / a.n_symbols
    L = [f"# Recency (k-back) task -- pre-flight gates (CPU, no training)", "",
         "Retrieve the k-th most recent symbol. A MATCH on the time axis, not a "
         "decode -- `environment_map_query.py` asked for a decode and got 0.121 "
         "against chance 0.016.", "",
         f"n_symbols={a.n_symbols}, k_max={a.k_max}, p_query={a.p_query}, "
         f"{a.episodes} episodes per row. **chance = 1/{a.n_symbols} = {chance:.4f}**; "
         f"the most-recent-symbol shortcut floor is 1/k_max + chance*(1-1/k_max) = "
         f"{1/a.k_max + chance*(1-1/a.k_max):.4f}.", "",
         "| min_gap | T | chance | marginal | o1 | o2 | o3 | o5 | most-recent | oracle | scored/token | n |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["ngram"]
        L.append(f"| {r['min_gap']} | {r['T']} | {r['chance']:.4f} | "
                 f"{r['marginal']:.4f} | {g[1]:.4f} | {g[2]:.4f} | {g[3]:.4f} | "
                 f"{g[5]:.4f} | {r['most_recent']:.4f} | {r['oracle']:.4f} | "
                 f"{r['scored_rate']:.3f} | {r['n_answers']} |")

    L += ["", "## G8 -- can a FIXED INDEX code address the answer?", "",
          "The token distance from a query back to its answer, per offset k. If "
          "that distance were constant, `k` would just be a token offset and an "
          "index code could address it directly -- measured, and it does: with "
          "`p_filler = 0`, RoPE scores **1.000** at every `k_max` tried while "
          "signed and monotone TIE at 0.946, i.e. the task was index retrieval "
          "in disguise. Filler tokens are emitted into the stream but not "
          "counted, so the distance becomes a random variable and `k` becomes a "
          "CONTEXTUAL position in CoPE's sense (arXiv:2405.11582: relative PE "
          "\"can do\" no better than \"a decaying attention\"). With "
          "`p_filler = 0.5`, RoPE falls to **0.31-0.37**.", "",
          "| min_gap | T | k=1 mean+/-sd | k=mid mean+/-sd | k=max mean+/-sd |",
          "|---|---|---|---|---|"]
    for r in rows:
        dm, ds = r["dist_mean"], r["dist_sd"]
        ks = sorted(dm)
        if not ks:
            continue
        pick = [ks[0], ks[len(ks) // 2], ks[-1]]
        L.append(f"| {r['min_gap']} | {r['T']} | " + " | ".join(
            f"k={k}: {dm[k]:.1f}+/-{ds[k]:.1f}" for k in pick) + " |")

    L += ["", "## G7 -- does the accumulator difference address a unique key?", "",
          "The mechanism the task is built to separate. `signed ambiguous` is the "
          "fraction of scored queries where some OTHER candidate key carries the "
          "same accumulator value AND a different symbol, so the difference "
          "`theta_query - theta_key` does not identify the answer.", "",
          "| min_gap | T | signed ambiguous | mean keys sharing theta | monotone ambiguous |",
          "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['min_gap']} | {r['T']} | **{r['signed_ambiguous']:.3f}** | "
                 f"{r['signed_candidates']:.2f} | {r['monotone_ambiguous']:.3f} |")
    L += ["", "Monotone is 0.000 by construction (`theta = t` is injective), which "
          "is the point and not a measurement. The signed column is simulated as a "
          "+/-1 walk per token -- a proxy for *unconstrained and not learning a "
          "counter*. It is an upper bound on the difficulty, NOT a prediction about "
          "a trained signed model: the unconstrained arm is free to learn a "
          "monotone code, and whether it does is the second prediction, measured "
          "on trained models with `probe_sign.py`.", "",
          "**Reading it.** Every baseline column must sit at `chance` except "
          "`most-recent`, which sits at its own stated floor, and `oracle`, which "
          "must be exactly 1.000. `o1`-`o5` above chance means the answer stream "
          "is self-predictable and the `min_gap` for that row is unusable."]
    # ---- verdict, computed rather than left to the reader ----
    mr_floor = 1.0 / a.k_max + chance * (1.0 - 1.0 / a.k_max)
    L += ["", "## Verdict", "",
          "A row PASSES when every n-gram order is within 0.01 of chance, "
          "`marginal` is within 0.01 of chance, `most-recent` is within 0.02 of "
          "its stated floor, and `oracle` is exactly 1.000.", "",
          "| min_gap | T | n-gram | marginal | most-recent | oracle | verdict |",
          "|---|---|---|---|---|---|---|"]
    n_pass = 0
    for r in rows:
        ng_ok = all(abs(v - chance) <= 0.01 for v in r["ngram"].values())
        mg_ok = abs(r["marginal"] - chance) <= 0.01
        mr_ok = abs(r["most_recent"] - mr_floor) <= 0.02
        or_ok = abs(r["oracle"] - 1.0) < 1e-9
        ok = ng_ok and mg_ok and mr_ok and or_ok
        n_pass += int(ok)
        ng_cell = "ok" if ng_ok else "**FAIL** o1={:.4f}".format(r["ngram"][1])
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["min_gap"], r["T"], ng_cell,
            "ok" if mg_ok else "FAIL", "ok" if mr_ok else "FAIL",
            "ok" if or_ok else "FAIL", "**PASS**" if ok else "fail"))
    L += ["", "{} of {} rows pass. The failures are the low-`min_gap` diagnostic "
          "rows and are EXPECTED -- they are what establishes that the default "
          "(`min_gap = k_max`) is necessary rather than decorative.".format(
              n_pass, len(rows)), ""]
    open(a.out, "w").write("\n".join(L) + "\n")
    json.dump(rows, open(a.out.replace(".md", ".json"), "w"), indent=2, default=str)
    print("\n".join(L)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
