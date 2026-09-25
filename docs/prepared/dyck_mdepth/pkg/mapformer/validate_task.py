"""
Pre-flight task validation. Run this BEFORE spending GPU on a new task.

Three invalid setups in one session (2026-07-16) all shared a root cause: the
MODEL was validated carefully (causality, param-match, seeds) and the TASK /
COMPARISON was not validated at all.

  - cascade "win"      -> baseline was a stale, non-converged checkpoint
  - aggregate "win"    -> flat wasn't incapable, it just trained at a shorter length
  - rooms open-plan    -> "planning" task was 100% solvable by a greedy heuristic

Each was caught only after training. Each was detectable in minutes of CPU.
This module makes those checks cheap and routine.

Checks
------
1. trivial_baseline   -- can a hand-coded heuristic solve it? If yes, the task
                         does not test the capability you think it does.
2. label_stats        -- chance level, label entropy, majority-class frequency.
                         A task whose majority class is 0.9 is not a task.
3. demand_profile     -- WHERE the evidence lives (how far back). Tells you
                         whether the capability under test is even necessary.
4. confound_checklist -- the controls that must run IN THE SAME BATCH, not
                         after a positive result appears.

Usage
-----
    python3 -m mapformer.validate_task --task rooms_open
    python3 -m mapformer.validate_task --task rooms_maze
    python3 -m mapformer.validate_task --task retrieval --T 4096

Interpreting: a task is SUSPECT if a trivial baseline scores near the ceiling,
if label entropy is near zero, or if the demand profile shows the capability
under test is not exercised.
"""
from __future__ import annotations
import argparse
import collections

import numpy as np
import torch


GREEN, RED, YELL = "PASS", "FAIL", "WARN"


def _verdict(ok: bool, warn: bool = False) -> str:
    return YELL if warn else (GREEN if ok else RED)


# ----------------------------------------------------------------- check 1
def trivial_baseline_navigation(env, n=200, seed=0, maze=False):
    """Fraction of optimal actions a GREEDY (distance-reducing) policy gets.

    ~1.0 means the task needs no planning: a hierarchy has nothing to
    decompose, and any 'planning' result is vacuous.
    """
    S = env.size
    AD = env.ACTION_DELTAS
    rng = np.random.RandomState(seed)

    def tdist(p, g):
        return (min((g[0] - p[0]) % S, (p[0] - g[0]) % S)
                + min((g[1] - p[1]) % S, (p[1] - g[1]) % S))

    def step(p, a):
        if maze:
            return env.step(p[0], p[1], a)
        dx, dy = AD[a]
        return ((p[0] + dx) % S, (p[1] + dy) % S)

    def greedy(p, g):
        cd = tdist(p, g)
        out = set()
        for a in AD:
            if maze and not env.can_move(p[0], p[1], a):
                continue
            if tdist(step(p, a), g) < cd:
                out.add(a)
        return out

    match = tot = 0
    ratios = []
    for _ in range(n):
        g = env.goal_cells[rng.randint(0, len(env.goal_cells))]
        p = (rng.randint(0, S), rng.randint(0, S))
        path = env.bfs(p, g) if maze else _bfs_open(p, g, S)
        if not path:
            continue
        ratios.append(len(path) / max(tdist(p, g), 1))
        cur = p
        for a in path:
            if a in greedy(cur, g):
                match += 1
            tot += 1
            cur = step(cur, a)
    return (match / tot if tot else float("nan")), float(np.mean(ratios)) if ratios else float("nan")


def _bfs_open(p, g, S):
    from mapformer.environment_goal import bfs_torus
    return bfs_torus(p, g, S)


def trivial_baseline_labels(labels, name="majority-class"):
    """Score of always predicting the most common label."""
    vals, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


# ----------------------------------------------------------------- check 2
def label_stats(labels, n_classes=None):
    vals, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    max_ent = float(np.log(n_classes if n_classes else len(vals)))
    return {"n_classes_used": int(len(vals)), "entropy": ent, "max_entropy": max_ent,
            "majority_freq": float(p.max()), "chance": 1.0 / (n_classes or len(vals))}


# ----------------------------------------------------------------- check 3
def demand_profile_revisit(env, T, n=20, window_steps=64, seed=0):
    """How far back the evidence lives, for revisit-style tasks.

    If most revisits are recent, a bounded/recency mechanism suffices and
    long-range memory (hierarchical or otherwise) is not being tested.
    """
    lags = []
    for k in range(n):
        np.random.seed(seed + k)
        env.generate_trajectory(T)
        last = {}
        for s, (x, y) in enumerate(env.visited_locations):
            if (x, y) in last:
                lags.append(s - last[(x, y)])
            last[(x, y)] = s
    lags = np.array(lags)
    if len(lags) == 0:
        return {}
    return {"n": int(len(lags)), "median_lag": float(np.median(lags)),
            f"frac_within_{window_steps}": float(np.mean(lags <= window_steps)),
            "frac_beyond_256": float(np.mean(lags > 256))}


# ----------------------------------------------------------------- check 4
CONFOUND_CHECKLIST = [
    ("parameter count",  "param-match the arms, or add an inert-param control"),
    ("training length",  "train the baseline AT the eval length too"),
    ("stale baselines",  "retrain every arm in the same batch; never load old checkpoints"),
    ("RNG / init drift", "same-seed init check; multi-seed (n>=3) from the start"),
    ("capacity / budget", "equalise the read/memory budget across arms"),
    ("component attribution", "ablate each component before claiming a mechanism"),
]


def print_confound_checklist():
    print("\n## 4. Confound checklist (run these IN THE SAME BATCH)\n")
    for name, action in CONFOUND_CHECKLIST:
        print(f"  [ ] {name:22s} -> {action}")
    print("\n  Controls run only AFTER a positive result are how retractions happen.")


# ----------------------------------------------------------------- drivers
def validate_navigation(env, label, maze):
    print(f"\n## 1. Trivial baseline ({label})\n")
    g, ratio = trivial_baseline_navigation(env, maze=maze)
    ok = g < 0.85
    print(f"  greedy-optimal fraction : {g:.3f}   [{_verdict(ok, warn=0.85 <= g < 0.95)}]")
    print(f"  path / straight-line    : {ratio:.2f}x")
    if not ok:
        print("  -> A greedy heuristic already solves this. There is NO planning")
        print("     problem to decompose; any hierarchy result here is vacuous.")
    else:
        print("  -> Greedy fails often enough that real planning is required.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["rooms_open", "rooms_maze", "rooms_maze_full",
                             "retrieval", "aggregate"])
    ap.add_argument("--T", type=int, default=1024)
    args = ap.parse_args()

    print(f"# Task validation: {args.task}")

    if args.task in ("rooms_open", "rooms_maze", "rooms_maze_full"):
        if args.task == "rooms_open":
            from mapformer.environment_rooms_goal import RoomsGoalWorld
            env = RoomsGoalWorld(size=64, rooms_per_side=8, theme_size=3,
                                 p_empty=0.0, seed=0)
            env.bfs = lambda p, g: _bfs_open(p, g, env.size)   # open torus BFS
            validate_navigation(env, "open-plan rooms", maze=False)
        else:
            from mapformer.environment_rooms_maze import RoomsMazeWorld
            conn = "tree" if args.task == "rooms_maze" else "full"
            env = RoomsMazeWorld(size=64, rooms_per_side=8, theme_size=3,
                                 connectivity=conn, seed=0)
            validate_navigation(env, f"maze ({conn} doors)", maze=True)
        print("\n## 2. Label stats\n  action prediction: 4 classes, chance = 0.250")

    elif args.task == "retrieval":
        from mapformer.environment import GridWorld
        env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=0)
        print("\n## 1. Trivial baseline (retrieval)\n")
        toks = [env.generate_trajectory(args.T) for _ in range(20)]
        labels = torch.cat([t[0][1:][t[2][1:]] for t in toks]).numpy()
        mf = trivial_baseline_labels(labels)
        print(f"  majority-class accuracy : {mf:.3f}   [{_verdict(mf < 0.6)}]")
        st = label_stats(labels, n_classes=17)
        print(f"\n## 2. Label stats\n  {st}")
        print(f"\n## 3. Demand profile (T={args.T})\n")
        dp = demand_profile_revisit(env, args.T)
        print(f"  {dp}")
        within = dp.get("frac_within_64", 1.0)
        print(f"  -> {within:.0%} of revisits are within 64 steps.")
        if within > 0.9:
            print("     WARN: evidence is overwhelmingly RECENT; long-range memory")
            print("     (hierarchical or otherwise) is barely exercised at this T.")

    elif args.task == "aggregate":
        from mapformer.environment import GridWorld
        from mapformer.train_aggregate import agg_targets
        env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=0)
        tokens = torch.stack([env.generate_trajectory(256)[0] for _ in range(64)])
        tgt, mask = agg_targets(tokens, 128)
        labels = tgt[mask].numpy()
        mf = trivial_baseline_labels(labels)
        print(f"\n## 1. Trivial baseline\n  majority-class accuracy : {mf:.3f}   [{_verdict(mf < 0.4)}]")
        print(f"\n## 2. Label stats\n  {label_stats(labels, n_classes=16)}")

    print_confound_checklist()


if __name__ == "__main__":
    main()
