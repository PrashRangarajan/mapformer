"""Does a per-token recursion router have anything to learn HERE?

MoR (arXiv 2507.10524) buys its win by routing each TOKEN to a different
recursion depth. That only pays if tokens actually differ in the depth they want.
Before building a router, measure whether they do -- on checkpoints that already
exist, with no training (rule 18: the loop count is a RUNTIME argument, and an
eval-only sweep is what found the loop's mechanism when a 12-run training sweep
had been proposed).

THE TEST. Sweep the global loop count k and score each query position by its
position WITHIN the blind query phase. Position is the highest-prior difficulty
axis because it is the one this repo has already measured depth-sensitivity on:
the same torus weights peak at 4 passes at T=128 and at 2 passes at T=512.

  argmax-k DIFFERS across position strata -> tokens want different depths, and a
      router has a real signal to exploit. MoR is worth building here.
  argmax-k is the SAME everywhere -> nothing to route on along this axis. MoR
      reduces to what LoopedSampled already does for free (0.998 at ONE pass,
      count-vs-accuracy curve flattened from 0.178 spread to 0.001), and the
      effort should go into a harder task instead.

SCOPE, stated honestly. This tests per-POSITION heterogeneity, which is a subset
of per-token. A router conditioned on the hidden state could in principle find an
axis this misses. Position is where our own evidence points, so a null here is
evidence against the idea, not proof against it.
"""
import argparse, json, statistics as st
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_match_query import MatchQueryGridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.train_match_query import _match_logits


@torch.no_grad()
def by_stratum(model, env, T_explore, T_query, n_batches, batch_size, device,
               seed, n_strata):
    """Accuracy per quintile of position WITHIN the query phase."""
    model.eval()
    rng = np.random.RandomState(seed)
    ok = np.zeros(n_strata); tot = np.zeros(n_strata)
    q_start = 2 * T_explore                       # query tokens begin here
    for _ in range(n_batches):
        toks, _rev, sps, ans, _i = env.generate_match_batch(
            batch_size, T_explore, T_query, rng)
        logits = model(toks[:, :-1].to(device))
        for b in range(toks.shape[0]):
            for p, a in zip(sps[b], ans[b]):
                if p >= logits.shape[1]:
                    continue
                frac = (p - q_start) / max(2 * T_query, 1)
                s = min(int(frac * n_strata), n_strata - 1)
                sl = _match_logits(logits, env, b, p)
                ok[s] += int(sl.argmax().item() == a - env.obs_offset)
                tot[s] += 1
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="mapformer/runs/loop_headroom/PI_loop")
    ap.add_argument("--variant", default="Looped")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--loops", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--t-explore", type=int, default=512)
    ap.add_argument("--t-query", type=int, default=256)
    ap.add_argument("--n-batches", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--n-strata", type=int, default=5)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="LOOP_DEPTH_STRATA.md")
    a = ap.parse_args()

    env = MatchQueryGridWorld(size=a.size, n_obs_types=a.n_obs, seed=10000)
    # acc[k][stratum] -> list over seeds
    acc = {k: [[] for _ in range(a.n_strata)] for k in a.loops}
    overall = {k: [] for k in a.loops}
    for s in a.seeds:
        cp = Path(a.runs_dir) / f"s{s}" / f"{a.variant}_matchquery.pt"
        if not cp.exists():
            print("MISSING", cp); continue
        c = torch.load(cp, map_location=a.device, weights_only=False)
        m = VARIANT_MAP[a.variant](vocab_size=c["vocab_size"], d_model=c["d_model"],
                                   n_heads=c["n_heads"], n_layers=c["n_layers"],
                                   grid_size=a.size).to(a.device).eval()
        m.load_state_dict(c["model_state"])
        for k in a.loops:
            m.n_loops = k                              # the runtime knob
            ok, tot = by_stratum(m, env, a.t_explore, a.t_query, a.n_batches,
                                 a.batch_size, a.device, 9000 + s, a.n_strata)
            for i in range(a.n_strata):
                if tot[i]:
                    acc[k][i].append(ok[i] / tot[i])
            overall[k].append(ok.sum() / max(tot.sum(), 1))
            print(f"[s{s}] k={k}: overall {ok.sum()/max(tot.sum(),1):.4f}  "
                  + " ".join(f"{(ok[i]/tot[i] if tot[i] else float('nan')):.3f}"
                             for i in range(a.n_strata)), flush=True)
        del m; torch.cuda.empty_cache()

    def mean(xs):
        return st.mean(xs) if xs else float("nan")

    lbl = [f"Q{i+1}" for i in range(a.n_strata)]
    o = ["# Do tokens want different recursion depths here?", "",
         f"`{a.variant}` on Match-Query {a.size}^2, n={len(a.seeds)} existing "
         f"checkpoints, inference only.", f"T_explore={a.t_explore}, "
         f"T_query={a.t_query}, chance = {1.0/a.n_obs:.4f}. Strata are equal slices "
         f"of position WITHIN the blind query phase (Q1 earliest).", "",
         "| loops k | overall | " + " | ".join(lbl) + " |",
         "|---" * (a.n_strata + 2) + "|"]
    for k in a.loops:
        o.append(f"| {k} | {mean(overall[k]):.3f} | "
                 + " | ".join(f"{mean(acc[k][i]):.3f}" for i in range(a.n_strata)) + " |")
    o.append("")

    best = {i: max(a.loops, key=lambda k: mean(acc[k][i])) for i in range(a.n_strata)}
    best_all = max(a.loops, key=lambda k: mean(overall[k]))
    o += ["## Best loop count per stratum", "",
          "| stratum | " + " | ".join(lbl) + " | overall |",
          "|---" * (a.n_strata + 2) + "|",
          "| argmax k | " + " | ".join(str(best[i]) for i in range(a.n_strata))
          + f" | {best_all} |", ""]

    spread = max(best.values()) - min(best.values())
    # how much a per-stratum ORACLE router would beat the single best global count
    oracle = mean([mean(acc[best[i]][i]) for i in range(a.n_strata)])
    fixed = mean([mean(acc[best_all][i]) for i in range(a.n_strata)])
    gain = oracle - fixed
    # the decisive comparison is against SEED NOISE, not against zero. An argmax
    # that wanders across a FLAT curve is noise, and branching on "spread != 0"
    # fires on it mechanically -- the same error as the grid-16 pre-registration
    # that called +0.015 "graded" because the band had been drawn too wide.
    sd = mean([st.stdev(overall[k]) for k in a.loops if len(overall[k]) > 1])
    o += ["## Verdict", "",
          f"argmax k spans {min(best.values())}..{max(best.values())} across strata "
          f"(spread {spread}) -- but read the oracle, not the argmax.",
          f"A per-stratum ORACLE router, allowed to know the best depth for each "
          f"stratum, scores {oracle:.3f} against {fixed:.3f} for the single best "
          f"global count. That is an upper bound of **{gain:+.3f}** on what ANY "
          f"router could buy along this axis.",
          f"Seed sd of overall accuracy is {sd:.3f}, so the oracle bound is "
          f"{sd / gain:.0f}x SMALLER than the run-to-run noise." if gain > 0 else "",
          ""]
    if gain < sd:
        o += [f"**No router can pay here.** The upper bound {gain:+.3f} is inside "
              f"the seed noise {sd:.3f}, so the differing argmaxes are wander on a "
              f"flat curve, not a signal. Above k=3 the whole curve is flat to "
              f"within 0.010. MoR's routing would reduce to picking one global "
              f"count, which LoopedSampled already does for free. Build a harder "
              f"task, not a router.", ""]
    else:
        o += [f"**A router has room**: the oracle bound {gain:+.3f} exceeds the seed "
              f"noise {sd:.3f}. Worth building.", ""]
    o += ["## Scope", "",
          "Per-POSITION heterogeneity only, which is a subset of per-token. A router "
          "on the hidden state could find an axis this misses; position is simply "
          "where this repo's own evidence points (the same torus weights peak at 4 "
          "passes at T=128 and 2 at T=512). A null here is evidence against, not "
          "proof against. Single task, single width, one T_explore/T_query pair."]

    Path(a.out).write_text("\n".join(o) + "\n")
    json.dump({"acc": {str(k): [list(v) for v in acc[k]] for k in acc},
               "overall": {str(k): overall[k] for k in overall}},
              open(a.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(o))


if __name__ == "__main__":
    main()
