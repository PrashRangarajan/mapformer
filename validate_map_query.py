"""Pre-flight gates for the Map-Query task. CPU only. Run BEFORE any training.

Every gate here exists because a specific past failure was catchable this cheaply
and wasn't caught:

  G1 random policy      establishes CHANCE. Several optimal first actions can
                        coexist on a torus, so chance is NOT 0.25 and must be
                        measured, not assumed.
  G2 answer n-gram      order 1..5 over the ANSWER STREAM ALONE. This is the gate
                        hier-goal failed: its action stream was 0.969 at order 1
                        (raw BFS) and 0.971 at order 3 (interleaved). Must sit at
                        chance.
  G3 goal-only          predict from the goal token alone, ignoring the explore
                        phase. THE HIGH-RISK GATE: the start is fixed, so for
                        small T_explore the agent stays near the origin and the
                        goal alone determines the direction. Reported per
                        T_explore so the operating point is chosen from data.
  G4 explore-only       predict from position alone, ignoring the goal. Catches a
                        degenerate answer distribution.
  G5 positional entropy how spread the end-of-explore position actually is. This
                        is the quantity that drives G3; reported in bits against
                        the log2(size^2) ceiling.
  G6 oracle             a solver with true position + goal must score 1.0, else
                        the task or the scorer is broken.

A gate at chance is a PASS. A gate above chance means the task is solvable
without the capability under test.
"""
import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_map_query import MapQueryGridWorld, optimal_first_actions


def collect(env, n_ep, T_explore, n_queries, seed):
    """Episodes reduced to (goal, end_pos, optimal-action-set) triples."""
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n_ep):
        _t, _sp, answers, info = env.generate_query_episode(T_explore, n_queries, rng)
        ex, ey = info["end_pos"]
        for qi, ans in enumerate(answers):
            rows.append((ex, ey, ans))
        rows[-len(answers):] = [(ex, ey, a) for a in answers]
    return rows, [info["end_pos"] for _ in range(0)] or None


def episodes(env, n_ep, T_explore, n_queries, seed):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_ep):
        _t, _sp, answers, info = env.generate_query_episode(T_explore, n_queries, rng)
        goals = []
        # recover goals from answer construction: re-derive by scanning tokens is
        # unnecessary -- store what we need by re-running the same rng draw is
        # fragile, so instead record (end_pos, answer) and sample goals directly.
        out.append((info["end_pos"], answers))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[64])
    ap.add_argument("--t-explore", nargs="+", type=int, default=[64, 128, 256, 512, 1024])
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="MAP_QUERY_GATES.md")
    args = ap.parse_args()

    S = args.sizes[0]
    lines = ["# Map-Query task -- pre-flight gates (CPU, no training)", "",
             "A gate at chance is a PASS. Above chance means the task is solvable "
             "without the capability under test.", ""]
    report = {}

    for T in args.t_explore:
        env = MapQueryGridWorld(size=S, room_size=8, seed=10000)
        rng = np.random.RandomState(args.seed)

        ends, goals, answers = [], [], []
        for _ in range(args.n_episodes):
            obs_rng = rng
            _t, _sp, ans, info = env.generate_query_episode(T, args.n_queries, obs_rng)
            ex, ey = info["end_pos"]
            # re-draw goals identically to score goal-conditioned baselines: the
            # answer set already encodes (end_pos, goal), so recover goals by
            # sampling fresh ones against the SAME end position.
            for _q in range(args.n_queries):
                gr = int(rng.randint(0, env.n_rooms)); gl = int(rng.randint(0, env.n_local))
                gx, gy = env.room_local_to_cell(gr, gl)
                ends.append((ex, ey)); goals.append((gx, gy))
                answers.append(optimal_first_actions((ex, ey), (gx, gy), S))

        n = len(answers)
        # G1 chance / random policy
        g1 = sum(int(rng.randint(0, 4)) in a for a in answers) / n
        # G6 oracle
        g6 = sum((min(a) if a else 0) in a for a in answers if a) / max(
            sum(1 for a in answers if a), 1)

        # G2 n-gram over answer stream alone (canonical answer = min of set)
        canon = [min(a) if a else 0 for a in answers]
        ngram = {}
        for K in (1, 2, 3, 5):
            tab = defaultdict(Counter)
            half = n // 2
            for i in range(K, half):
                tab[tuple(canon[i - K:i])][canon[i]] += 1
            pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
            ok = tot = 0
            for i in range(half + K, n):
                c = tuple(canon[i - K:i])
                if c in pred:
                    ok += int(pred[c] in answers[i])
                tot += 1
            ngram[K] = ok / max(tot, 1)

        # G3 goal-only, with MAJORITY BACKOFF. Without backoff, unseen goal cells
        # score as wrong and the gate reads far below chance -- a coverage
        # artifact, not evidence of no signal.
        half = n // 2
        glob = Counter()
        for i in range(half):
            for a in answers[i]:
                glob[a] += 1
        fallback = glob.most_common(1)[0][0]
        tab = defaultdict(Counter)
        for i in range(half):
            for a in answers[i]:
                tab[goals[i]][a] += 1
        g3ok = 0
        for i in range(half, n):
            c = tab.get(goals[i])
            guess = c.most_common(1)[0][0] if c else fallback
            g3ok += int(guess in answers[i])
        g3 = g3ok / max(n - half, 1)

        # G3b THE REAL SHORTCUT: assume the agent never moved from the fixed
        # start. If the walk leaves position concentrated, this alone solves the
        # task without any path integration.
        # Predict ONE action (as chance does). Set-intersection scoring would be
        # incomparable to the single-action chance rate: two random 2-of-4 sets
        # overlap ~83% of the time regardless of any signal.
        def _one(st, g):
            o = optimal_first_actions(st, g, S)
            return min(o) if o else 0
        g3b = sum(int(_one(env.start, goals[i]) in answers[i])
                  for i in range(half, n)) / max(n - half, 1)

        # G4 explore-only: best action per end position, ignoring goal
        tab = defaultdict(Counter)
        _ = fallback
        for i in range(half):
            for a in answers[i]:
                tab[ends[i]][a] += 1
        g4ok = 0
        for i in range(half, n):
            c = tab.get(ends[i])
            guess = c.most_common(1)[0][0] if c else fallback
            g4ok += int(guess in answers[i])
        g4 = g4ok / max(n - half, 1)

        # G5 positional entropy of end-of-explore cell
        cnt = Counter(ends)
        tot = sum(cnt.values())
        ent = -sum((v / tot) * math.log2(v / tot) for v in cnt.values())
        ceiling = math.log2(S * S)
        # H(pos) is capped by log2(n_episodes), so it measures sample size, not the
        # walk. Mean torus distance from the start is sample-size independent; a
        # uniform position gives S/2 = mean of each axis marginal.
        mean_d = float(np.mean([min((ex - env.start[0]) % S, (env.start[0] - ex) % S)
                                + min((ey - env.start[1]) % S, (env.start[1] - ey) % S)
                                for (ex, ey) in ends]))
        uniform_d = S / 2.0

        # G7 ROOM QUERY: "which room are you in now?" -- one token, chance 1/n_rooms,
        # far higher resolution than the 0.5-chance direction query.
        room_of = [ (ex // env.room_size) * env.rooms_per_side + (ey // env.room_size)
                    for (ex, ey) in ends ]
        rc = Counter(room_of)
        room_majority = rc.most_common(1)[0][1] / len(room_of)   # best constant guess
        room_chance = 1.0 / env.n_rooms

        report[T] = dict(chance=g1, oracle=g6, ngram=ngram, goal_only=g3,
                         goal_only_assume_start=g3b, room_chance=room_chance,
                         room_majority=room_majority,
                         explore_only=g4, pos_entropy=ent, entropy_ceiling=ceiling, mean_dist=mean_d, uniform_dist=uniform_d,
                         distinct_end_cells=len(cnt))
        print(f"T_explore={T:5d} chance={g1:.3f} oracle={g6:.3f} "
              f"ngram1={ngram[1]:.3f} ngram3={ngram[3]:.3f} goal_only={g3:.3f} "
              f"assume_start={g3b:.3f} explore_only={g4:.3f} "
              f"roomMaj={room_majority:.3f}(chance {room_chance:.3f}) "
              f"meanD={mean_d:.1f}/{uniform_d:.1f}", flush=True)

    lines += ["| T_explore | chance | oracle | n-gram o1 | o3 | o5 | goal-only | assume-start | explore-only | room: best-constant (chance) | mean dist from start (uniform) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for T, r in report.items():
        lines.append(
            f"| {T} | {r['chance']:.3f} | {r['oracle']:.3f} | {r['ngram'][1]:.3f} | "
            f"{r['ngram'][3]:.3f} | {r['ngram'][5]:.3f} | {r['goal_only']:.3f} | "
            f"{r['goal_only_assume_start']:.3f} | {r['explore_only']:.3f} | "
            f"{r['room_majority']:.3f} ({r['room_chance']:.3f}) | "
            f"{r['mean_dist']:.1f} ({r['uniform_dist']:.0f}) |")
    lines += ["", "**Reading it.** `chance` is the random-policy rate and is the "
              "PASS level for every baseline column. `goal-only` above chance means "
              "the fixed start leaves position too concentrated -- raise T_explore. "
              "`oracle` must be 1.000."]
    open(args.out, "w").write("\n".join(lines) + "\n")
    json.dump(report, open(args.out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
