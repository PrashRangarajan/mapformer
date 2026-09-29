"""Gate for the two-condition decoy task (environment_textworld_ctx2.py), CPU, calling the task code.
Per condition: (1) the walk is reproduced from move-class direction words only; (2) the move/decoy class
is unpredictable from the MATCHED side (3 tokens after the direction word for "lead", 4 before for
"trail") and predictable from the CUE side -- a frequency table fit on one sample, scored on another,
against the majority base rate; (3) eval-set floors: constant, reversal-copy over move steps, word n-grams."""
from collections import Counter, defaultdict
import numpy as np
from mapformer.environment_textworld import DIRS
from mapformer.environment_textworld_ctx2 import TextWorldCtx2
from mapformer.environment import GridWorld

def sample(cue, seed, n, env_seed):
    env = TextWorldCtx2(seed=env_seed, cue=cue); np.random.seed(seed); out = []
    for _ in range(n):
        tok, obs, rev = env.generate_trajectory(1024)
        out.append((tok.tolist(), obs.numpy(), rev.numpy(), list(env.ctx), list(env.visited_locations)))
    return env, out

for cue in ("lead", "trail"):
    env, E = sample(cue, 10**6, 200, 10000)
    _, Tr = sample(cue, 5, 600, 0)
    w2a = {env.idx[w]: a for a, ws in DIRS.items() for w in ws}
    bad = 0
    for tl, obs, rev, ctx, locs in E:
        mv = [i for i, k in ctx if k == "move"]
        pos = None
        for j, i in enumerate(mv[:len(locs)]):
            d = np.array(GridWorld.ACTION_DELTAS[w2a[tl[i]]])
            pos = (np.array(locs[0]) if pos is None else (pos + d) % 64)
            bad += tuple(pos) != tuple(locs[j])
    kinds = Counter(k for e in E for _, k in e[3])
    def windows(data, side, n):
        X = []
        for tl, obs, rev, ctx, locs in data:
            for i, k in ctx:
                w = tuple(tl[i + 1:i + 1 + n]) if side == "after" else tuple(tl[max(0, i - n):i])
                X.append((w, k != "move"))
        return X
    def predictability(side, n):
        tab = defaultdict(Counter)
        for w, y in windows(Tr, side, n):
            tab[w][y] += 1
        te = windows(E, side, n); base = max(np.mean([y for _, y in te]), 1 - np.mean([y for _, y in te]))
        maj = Counter(y for _, y in windows(Tr, side, n)).most_common(1)[0][0]
        acc = np.mean([(tab[w].most_common(1)[0][0] if tab[w] else maj) == y for w, y in te])
        return acc, base
    matched, cueside = (("after", 3), ("before", 4)) if cue == "lead" else (("before", 4), ("after", 3))
    am, b = predictability(*matched); ac, _ = predictability(*cueside)
    # floors
    opp = {0: 1, 1: 0, 2: 3, 3: 2}; nothing = env.idx["nothing"]; ys, rc = [], []
    for tl, obs, rev, ctx, locs in E:
        slots = np.nonzero(obs)[0]; mv = [tl[i] for i, k in ctx if k == "move"]
        for k, i in enumerate(slots):
            if rev[i]:
                ys.append(tl[i])
                rc.append(tl[slots[k - 2]] == tl[i] if k >= 2 and w2a[mv[k]] == opp[w2a[mv[k - 1]]] else tl[i] == nothing)
    const = Counter(ys).most_common(1)[0][1] / len(ys)
    ng = {}
    for n in (1, 3, 5):
        tab = defaultdict(Counter)
        for tl, obs, rev, ctx, locs in Tr:
            for i in np.nonzero(rev)[0]:
                tab[tuple(tl[i - n:i])][tl[i]] += 1
        mode = Counter(y for e in Tr for y in [e[0][i] for i in np.nonzero(e[2])[0]]).most_common(1)[0][0]
        ng[n] = np.mean([(tab[tuple(tl[i - n:i])].most_common(1)[0][0] if tab[tuple(tl[i - n:i])] else mode) == tl[i]
                         for tl, obs, rev, ctx, locs in E for i in np.nonzero(rev)[0]])
    print(f"[{cue}] vocab {env.unified_vocab_size}; classes {dict(kinds)}; walk mismatches {bad}")
    print(f"[{cue}] class from MATCHED side {matched}: {am:.3f} vs base {b:.3f} | from CUE side {cueside}: {ac:.3f}")
    print(f"[{cue}] floors: {len(ys)} targets, const {const:.4f}, reversal-copy {np.mean(rc):.4f}, n-gram "
          + " ".join(f"{n}:{v:.3f}" for n, v in ng.items()))
    tl = E[0][0]; print(f"[{cue}] sample:", " ".join(env.vocab[t] for t in tl[:70]))
