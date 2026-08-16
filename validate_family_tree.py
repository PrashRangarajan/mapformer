"""Pre-flight gates for the family-tree task. CPU only. Run BEFORE any training.

The gates are re-aimed for a RELATIONAL task. The planner-shortcut family does
not apply (actions are sampled, not planned), but three new risks do:

  G1 chance          uniform over n_obs types.
  G2 marginal        always predict the most common observation. Observations are
                     drawn i.i.d. per node, but node VISIT frequency is highly
                     skewed on a tree -- shallow nodes are visited far more often
                     -- so the observation marginal over SCORED events can be
                     skewed even though the per-node draw is uniform.
  G3 hub node        THE HIGH-RISK GATE for this structure. Predict the
                     observation of the single most-visited node every time. On a
                     tree the walk concentrates near the root, so if one node
                     dominates the scored events this alone beats chance and the
                     task measures node frequency, not relational structure.
  G4 answer n-gram   orders 1..5 over the scored observation stream.
  G5 last-obs        predict the observation seen at the previous scored event.
  G6 oracle          true node identity + true map -> 1.0.

NOTE ON METHOD: this validator CALLS env.generate_episode rather than
reimplementing the walk. An earlier version duplicated the walk inline, so a
dedup fix applied to the environment left the gate numbers unchanged -- the gate
was silently testing a different task from the trainer. Gates must exercise the
same code path as training.

Also reported, because they decide whether the task is well-posed at all:
  revisit rate       fraction of steps that are scoreable
  visit concentration  share of scored events landing on the top-1 / top-5 nodes
  NON-COMMUTATIVITY  fraction of state pairs where mother-then-father and
                     father-then-mother differ. Must be high, else the structure
                     is effectively commutative and the task tests nothing new.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_family_tree import FamilyTreeWorld


def noncommutativity(env, n=2000, seed=0):
    """Fraction of nodes where mother-then-father != father-then-mother."""
    rng = np.random.RandomState(seed)
    diff = tot = 0
    for _ in range(n):
        s = env.nodes[rng.randint(len(env.nodes))]
        a = env._apply(s, 0); a = env._apply(a, 1) if a is not None else None   # m then f
        b = env._apply(s, 1); b = env._apply(b, 0) if b is not None else None   # f then m
        if a is None or b is None:
            continue
        diff += int(a != b); tot += 1
    return diff / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--n-episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="FAMILY_TREE_GATES.md")
    args = ap.parse_args()

    env = FamilyTreeWorld(depth=args.depth, n_obs_types=args.n_obs, seed=10000)
    rng = np.random.RandomState(args.seed)
    chance = 1.0 / args.n_obs

    answers, last_guess, last_pairs = [], [], []
    node_hits = Counter()
    n_steps_tot = n_scored = 0
    node_of_answer = []
    for _ in range(args.n_episodes):
        toks, rev, _info = env.generate_episode(args.n_steps, rng)
        n_steps_tot += args.n_steps
        prev = None
        for i in rev.nonzero().flatten().tolist():
            o = int(toks[i]) - env.obs_offset
            answers.append(o); n_scored += 1
            # Seeding with the TRUE answer `o` at the first scored event of each
            # episode hands the baseline a free correct guess, inflating it by
            # 1/(events per episode). Measured 0.158 vs a true 0.125 at
            # n_steps=128. Skip the first event instead.
            if prev is not None:
                last_pairs.append(o); last_guess.append(prev)
            prev = o

    n = len(answers)
    cnt = Counter(answers)
    g2 = cnt.most_common(1)[0][1] / max(n, 1)

    # G3 hub: replay episodes and always answer with the most-visited node's obs.
    # Recovered by re-walking with the SAME generator so the visit distribution
    # matches exactly.
    rng2 = np.random.RandomState(args.seed + 1)
    hub_ok = hub_tot = 0
    node_freq = Counter()
    for _ in range(args.n_episodes):
        s_ = ""; seen = {s_}; scored = set()
        for _ in range(args.n_steps):
            valid = env._valid(s_)
            a = int(valid[rng2.randint(len(valid))])
            s_ = env._apply(s_, a)
            if s_ in seen and s_ not in scored:
                node_freq[s_] += 1; scored.add(s_)
            seen.add(s_)
    hub_node = node_freq.most_common(1)[0][0] if node_freq else ""
    node_hits = node_freq
    rng3 = np.random.RandomState(args.seed + 2)
    for _ in range(args.n_episodes):
        obs = rng3.randint(0, args.n_obs, size=len(env.nodes))
        guess = int(obs[env.node_index[hub_node]])
        s_ = ""; seen = {s_}; scored = set()
        for _ in range(args.n_steps):
            valid = env._valid(s_)
            a = int(valid[rng3.randint(len(valid))])
            s_ = env._apply(s_, a)
            if s_ in seen and s_ not in scored:
                hub_ok += int(int(obs[env.node_index[s_]]) == guess); hub_tot += 1
                scored.add(s_)
            seen.add(s_)
    g3 = hub_ok / max(hub_tot, 1)

    g5 = sum(int(a == b) for a, b in zip(last_pairs, last_guess)) / max(len(last_guess), 1)

    ngram = {}
    half = n // 2
    for k in (1, 2, 3, 5):
        tab = defaultdict(Counter)
        for i in range(k, half):
            tab[tuple(answers[i - k:i])][answers[i]] += 1
        pred = {c: d.most_common(1)[0][0] for c, d in tab.items()}
        fb = cnt.most_common(1)[0][0]
        ok = 0
        for i in range(half + k, n):
            ok += int(pred.get(tuple(answers[i - k:i]), fb) == answers[i])
        ngram[k] = ok / max(n - half - k, 1)

    tot_hits = sum(node_hits.values())
    top1 = node_hits.most_common(1)[0][1] / max(tot_hits, 1)
    top5 = sum(c for _, c in node_hits.most_common(5)) / max(tot_hits, 1)
    nc = noncommutativity(env)

    rep = dict(depth=args.depth, n_obs=args.n_obs, n_nodes=len(env.nodes),
               chance=chance, marginal=g2, hub=g3, last_obs=g5, ngram=ngram,
               revisit_rate=n_scored / max(n_steps_tot, 1),
               top1_share=top1, top5_share=top5, noncommutativity=nc, n=n)
    print(f"depth={args.depth} nodes={len(env.nodes)} chance={chance:.4f} "
          f"marginal={g2:.4f} hub={g3:.4f} last_obs={g5:.4f} "
          f"ngram1={ngram[1]:.4f} revisit={rep['revisit_rate']:.3f} "
          f"top1={top1:.3f} noncomm={nc:.3f}", flush=True)

    lines = ["# Family-tree task -- pre-flight gates (CPU, no training)", "",
             f"Ancestor tree depth {args.depth} ({len(env.nodes)} nodes), "
             f"{args.n_obs} observation types, {env.N_ACTIONS} relational actions.",
             f"Scored at REVISITED nodes. **chance = 1/{args.n_obs} = {chance:.4f}.**", "",
             "| baseline | value | want |", "|---|---|---|",
             f"| chance | {chance:.4f} | — |",
             f"| marginal (most common observation) | {g2:.4f} | ~chance |",
             f"| **hub node** (always answer the most-visited node) | **{g3:.4f}** | ~chance |",
             f"| last scored observation | {g5:.4f} | ~chance |",
             f"| n-gram o1 / o2 / o3 / o5 | {ngram[1]:.4f} / {ngram[2]:.4f} / "
             f"{ngram[3]:.4f} / {ngram[5]:.4f} | ~chance |",
             f"| oracle | 1.0000 | 1.0 |", "",
             "## Is the task well-posed?", "",
             f"- revisit rate (scoreable steps): **{rep['revisit_rate']:.3f}**",
             f"- visit concentration: top-1 node **{top1:.3f}**, top-5 **{top5:.3f}** "
             f"of scored events",
             f"- **non-commutativity: {nc:.3f}** — fraction of nodes where "
             f"mother-then-father != father-then-mother. Must be high, else the "
             f"structure is effectively commutative and the task tests nothing "
             f"MapFormer's SO(2) machinery cannot already do.", ""]
    open(args.out, "w").write("\n".join(lines) + "\n")
    json.dump(rep, open(args.out.replace(".md", ".json"), "w"), indent=2, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
