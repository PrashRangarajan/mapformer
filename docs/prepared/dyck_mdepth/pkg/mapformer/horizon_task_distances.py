"""How far back must each task reach? (CPU only, no models.)

Part 2 of the horizon measurement. `REVISIT_DISTANCE.md` showed index-position
models beat the blank floor ONLY at recurrence interval 1-2 (+0.05 to +0.07) and
sit at or below it in every other bucket -- attention path-integrates over a
horizon of about two steps and then stops.

If that horizon is the whole story, then how well an index model does on a task
should be predicted by ONE number: the fraction of that task's scored events
whose answer lies within the horizon. This computes that fraction for each task,
from the task's own generator, with no model involved.

Recurrence interval = steps since the scored cell/node was last visited. For a
blind-continuation task it is measured from the last time the cell was OBSERVED,
which is what the model would have to reach back to.

Compositional is deliberately excluded and the reason is the interesting part:
its `cross_nb` metric scores prediction in a room-instance the agent has not
visited, using a motif seen in a DIFFERENT instance. That is template matching,
not reaching back to a previous visit, so recurrence interval is not defined for
it -- and that is exactly why index models do well there (0.216 vs MapFormer's
0.270) while sitting on the floor on the paper task. The dissociation is not that
attention sometimes path-integrates further; it is that some tasks never ask it
to.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
BUCKETS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 10**9)]


def _label(lo, hi):
    return f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"


def _bucketise(intervals):
    c = Counter()
    for d in intervals:
        for lo, hi in BUCKETS:
            if lo <= d <= hi:
                c[_label(lo, hi)] += 1
                break
    n = max(sum(c.values()), 1)
    return {_label(lo, hi): c[_label(lo, hi)] / n for lo, hi in BUCKETS}, n


def paper_task(n_ep, n_steps, seed):
    from mapformer.environment import GridWorld
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=seed)
    np.random.seed(seed)
    out = []
    for _ in range(n_ep):
        _t, _o, rev, locs = env.generate_batch(1, n_steps)
        last = {}
        for s, c in enumerate(locs[0]):
            c = tuple(c)
            if c in last and bool(rev[0, 2 * s + 1]):
                out.append(s - last[c])
            last[c] = s
    return out


def match_query(n_ep, TE, TQ, seed):
    from mapformer.environment_match_query import MatchQueryGridWorld
    env = MatchQueryGridWorld(size=64, n_obs_types=16, seed=seed)
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_ep):
        _t, _r, _sp, ans, info = env.generate_match_episode(TE, TQ, rng)
        # every scored cell was last OBSERVED during explore; the model must
        # reach back across the entire blind phase to it. Lower-bound the
        # interval by the number of query steps taken before scoring.
        for i, _a in enumerate(ans):
            out.append(TE + i)          # conservative: >= T_explore steps back
    return out


def family_tree(n_ep, n_steps, depth, n_obs, seed):
    from mapformer.environment_family_tree import FamilyTreeWorld
    env = FamilyTreeWorld(depth=depth, n_obs_types=n_obs, seed=seed)
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_ep):
        s_ = ""
        seen = {s_: 0}
        scored = set()
        for step in range(1, n_steps + 1):
            valid = env._valid(s_)
            a = int(valid[rng.randint(len(valid))])
            s_ = env._apply(s_, a)
            if s_ in seen and s_ not in scored:
                out.append(step - seen[s_])
                scored.add(s_)
            seen[s_] = step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--out", default=str(_REPO / "HORIZON_TASK_DISTANCES.md"))
    args = ap.parse_args()

    tasks = {
        "paper task (T=128)": paper_task(args.n_episodes, 128, args.seed),
        "paper task (T=512)": paper_task(max(args.n_episodes // 4, 25), 512, args.seed),
        "Match-Query (TE=512 TQ=256)": match_query(max(args.n_episodes // 4, 25),
                                                   512, 256, args.seed),
        "family tree (depth 5, T=64)": family_tree(args.n_episodes, 64, 5, 8, args.seed),
    }
    keys = [_label(lo, hi) for lo, hi in BUCKETS]
    rows, summary = [], {}
    for name, iv in tasks.items():
        d, n = _bucketise(iv)
        within = d[keys[0]]
        summary[name] = {"buckets": d, "n": n, "within_horizon": within,
                         "median_interval": float(np.median(iv)) if iv else None}
        rows.append((name, d, n, within, float(np.median(iv)) if iv else float("nan")))

    lines = ["# How far back must each task reach?", "",
             "`REVISIT_DISTANCE.md` measured that index-position models beat the "
             "blank floor ONLY at recurrence interval 1-2 (+0.05 to +0.07), and "
             "sit at or below it in every other bucket. **Attention "
             "path-integrates over a horizon of roughly two steps.**", "",
             "If that is the whole story, an index model's performance on a task "
             "should be predicted by the share of that task's scored events whose "
             "answer lies inside the horizon. That share is computed here from "
             "each task's own generator, with no model involved.", "",
             "| task | " + " | ".join(keys) + " | median | **within horizon (1-2)** |",
             "|---" * (len(keys) + 3) + "|"]
    for name, d, n, within, med in rows:
        lines.append(f"| {name} | " + " | ".join(f"{d[k]:.3f}" for k in keys) +
                     f" | {med:.0f} | **{within:.3f}** |")
    lines += ["", "## Reading it against measured index-model performance", "",
              "| task | within horizon | index model recovers |", "|---|---|---|",
              "| paper task | see above | +0.007-0.010 over a 0.506 floor, i.e. "
              "~2% of the 0.48 headroom |",
              "| Match-Query | see above | 0.154 vs a 0.089 floor — ~8% of the "
              "headroom |",
              "| family tree | see above | 0.612 vs a 0.163 hub floor — but this "
              "task's floor is itself visit-frequency, so read with care |",
              "| **compositional** | **not defined** | **0.216 vs MapFormer's "
              "0.270 — ~80% of the score** |", "",
              "## Why compositional is excluded, and why that is the point", "",
              "`cross_nb` scores prediction in a room-instance the agent has NOT "
              "visited, using a motif seen in a different instance. That is "
              "template matching, not reaching back to a previous visit, so "
              "recurrence interval is undefined for it. **That is exactly why "
              "index models do well there and fail on the paper task.** The "
              "dissociation is not that attention sometimes path-integrates "
              "further — it is that some tasks never ask it to.", "",
              "## What would falsify the horizon account", "",
              "A task with a LARGE within-horizon share on which index models "
              "still fail, or a small share on which they succeed. Either would "
              "mean the horizon is not the operative variable.", ""]
    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump(summary, open(str(args.out).replace(".md", ".json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
