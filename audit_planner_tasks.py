"""Action-only n-gram audit of every planner-demonstration task. CPU only.

WHY THIS EXISTS. Two tasks were invalidated on 2026-08-09 by the same flaw:
scoring next-action prediction on optimal-planner demonstrations, where the
planner's output is predictable from ITSELF. Measured on hier-goal, with no model
involved:

    raw BFS path        order-1  0.969
    interleaved path    order-1  0.320   but order-3  0.971

The "fix" moved the shortcut from order 1 to order 3 and made it stronger, and it
passed the only check that was run (a copy-previous baseline, which tests order 1
alone). Closed-loop success for every trained variant was 0.013-0.037 against a
0.010 random floor -- nothing navigational had been learned.

Every remaining task in this repo that scores planner actions carries the same
risk and has NOT been checked. This audits them all at orders 1..5.

READING IT. `chance` is 1/n_actions. If an n-gram fitted on the ACTION STREAM
ALONE -- no goal, no observations, no position -- scores far above chance, the
task is solvable without the capability it claims to test, and its results are
void. hier-goal is included as a positive control: it MUST come out ~0.97.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np


def ngram_scores(streams, n_actions, orders=(1, 2, 3, 4, 5)):
    """Fit order-k Markov predictors on half the streams, test on the other half."""
    flat, bounds = [], []
    for s in streams:
        bounds.append((len(flat), len(flat) + len(s)))
        flat.extend(s)
    half = len(streams) // 2
    train_end = bounds[half][0] if half else len(flat)
    out = {}
    for k in orders:
        tab = defaultdict(Counter)
        for i in range(k, train_end):
            tab[tuple(flat[i - k:i])][flat[i]] += 1
        pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
        glob = Counter(flat[:train_end]).most_common(1)[0][0] if train_end else 0
        ok = tot = 0
        for lo, hi in bounds[half:]:
            for i in range(max(lo + k, k), hi):
                ok += int(pred.get(tuple(flat[i - k:i]), glob) == flat[i])
                tot += 1
        out[k] = ok / max(tot, 1)
    return out, 1.0 / n_actions


def collect(name, n_ep, seed):
    """Return (list of scored-action streams, n_actions) for one task."""
    rng = np.random.RandomState(seed)
    streams = []
    if name == "hier_goal(control)":
        from mapformer.environment_hier_goal import HierGoalGridWorld
        env = HierGoalGridWorld(size=64, room_size=8, seed=10000)
        for _ in range(n_ep):
            t, _om, am, _i = env.generate_hier_episode(64, 64, rng=rng)
            streams.append([int(t[p + 1]) for p in am.nonzero().flatten().tolist()
                            if p + 1 < t.shape[0]])
        return streams, env.N_ACTIONS
    if name == "goal":
        from mapformer.environment_goal import GoalDirectedGridWorld as G
        env = G(size=64, seed=10000)
        for _ in range(n_ep):
            t, _om, am, _i = env.generate_goal_episode(T_explore=64, T_navigate=64, rng=rng)
            streams.append([int(t[p + 1]) for p in am.nonzero().flatten().tolist()
                            if p + 1 < t.shape[0]])
        return streams, env.N_ACTIONS
    if name == "rooms_goal":
        from mapformer.environment_rooms_goal import RoomsGoalWorld as G
        env = G(seed=10000)
        for _ in range(n_ep):
            t, am, _i = env.generate_goal_episode(T_explore=64, T_navigate=64, rng=rng)
            streams.append([int(t[p + 1]) for p in am.nonzero().flatten().tolist()
                            if p + 1 < t.shape[0]])
        return streams, env.N_ACTIONS
    if name == "rooms_maze":
        from mapformer.environment_rooms_maze import RoomsMazeWorld as G
        env = G(seed=10000)
        for _ in range(n_ep):
            t, am, _i = env.generate_goal_episode(T_explore=64, T_navigate=64, rng=rng)
            streams.append([int(t[p + 1]) for p in am.nonzero().flatten().tolist()
                            if p + 1 < t.shape[0]])
        return streams, env.N_ACTIONS
    if name == "maze_varying":
        from mapformer.environment_maze_varying import VaryingMazeWorld as G
        env = G(seed=10000)
        tries = 0
        while len(streams) < n_ep and tries < n_ep * 20:
            tries += 1
            ep = env.generate_episode(T_explore=256, T_navigate=48, rng=rng)
            if ep is None:            # the env resamples on unreachable goals
                continue
            t, am, _i = ep
            streams.append([int(t[p + 1]) for p in am.nonzero().flatten().tolist()
                            if p + 1 < t.shape[0]])
        return streams, env.N_ACTIONS
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["hier_goal(control)", "goal", "rooms_goal",
                             "rooms_maze", "maze_varying"])
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="PLANNER_TASK_AUDIT.md")
    args = ap.parse_args()

    rows = []
    for name in args.tasks:
        try:
            streams, nA = collect(name, args.n_episodes, args.seed)
            streams = [s for s in streams if len(s) > 6]
            if len(streams) < 10:
                rows.append(dict(task=name, error="too few scored actions")); continue
            ng, chance = ngram_scores(streams, nA)
            worst = max(ng.values())
            rows.append(dict(task=name, chance=chance, ngram=ng, worst=worst,
                             verdict="VOID" if worst > chance + 0.25 else
                                     "suspect" if worst > chance + 0.10 else "ok"))
            print(f"{name:22s} chance={chance:.3f}  "
                  f"o1={ng[1]:.3f} o2={ng[2]:.3f} o3={ng[3]:.3f} o5={ng[5]:.3f}  "
                  f"-> {rows[-1]['verdict']}", flush=True)
        except Exception as e:
            rows.append(dict(task=name, error=f"{type(e).__name__}: {e}"))
            print(f"{name:22s} ERROR {type(e).__name__}: {e}", flush=True)

    lines = ["# Planner-task audit: n-grams on the ACTION STREAM ALONE", "",
             "No model, no goal, no observations, no position -- just the scored action",
             "sequence predicting itself. Far above chance => the task is solvable without",
             "the capability it claims to test.", "",
             "`hier_goal(control)` is a POSITIVE CONTROL and must come out ~0.97.", "",
             "| task | chance | o1 | o2 | o3 | o4 | o5 | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['task']} | — | — | — | — | — | — | not audited ({r['error']}) |")
        else:
            g = r["ngram"]
            lines.append(f"| {r['task']} | {r['chance']:.3f} | {g[1]:.3f} | {g[2]:.3f} | "
                         f"{g[3]:.3f} | {g[4]:.3f} | {g[5]:.3f} | **{r['verdict']}** |")
    lines += ["", "Thresholds: **VOID** if any order exceeds chance+0.25, *suspect* if "
              "above chance+0.10."]
    open(args.out, "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(args.out.replace(".md", ".json"), "w"), indent=2, default=str)
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
