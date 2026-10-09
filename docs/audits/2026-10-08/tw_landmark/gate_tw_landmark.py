"""Rule-11 gate for TW_LANDMARK_PREREG.md, CALLING the task code (environment_tw_landmark, tw_landmark_eval.build_eval).
CPU only. Eval stream = the batch's: held-out map 10000, np seed 10**6, 200 walks, T = 1024, common-move targets
(moves that fit in the rate-1 rendering), per training name rate r in {0, 0.5, 1} and per eval condition.

Reports, per r: object slots and revisit fraction per walk (own rendering); K1; the share of scored revisits whose
place is NAMED at least once before (name-solvable: a landmark cell, consistent names); floors on the scored targets:
  constant       best single answer
  rcopy          reversal-copy rule (copy the object 2 moves back when this move reverses the previous; else 'nothing')
  ncopy          name lookup: copy the object last seen 3 tokens after the same name (else 'nothing')
  ncopy+rcopy    name lookup where available, else the reversal-copy rule (the best trivial rule we know)
  n-gram k       word n-gram over the k preceding words (k = 1..5), fit on 1500 training walks (map seed 50) rendered
                 at the same rate and mode as the eval condition
Answer-leak checks: object distribution on landmark vs unnamed cells; a name -> object table fit across training walks
(names are redrawn per walk, so it must sit at the constant floor); direction words only inside movement clauses;
every token id inside the vocabulary.
"""
from collections import Counter, defaultdict

import numpy as np

from mapformer.environment_tw_landmark import TextWorldLandmark, render_conditions
from mapformer.tw_landmark_eval import build_eval, conds_for, EVAL_SEED

N_TRAIN, MAP_TRAIN = 1500, 50


def rules(toks, tg, slots, env):
    """Trivial predictors on the scored targets, from the rendering's own slot bookkeeping."""
    nothing = env.idx["nothing"]; names = set(env.name_ids); opp = {0: 1, 1: 0, 2: 3, 3: 2}
    w2d = {i: a for a in range(4) for i in env.dir_ids[a]}
    out = {"rcopy": [], "ncopy": [], "ncopy+rcopy": [], "nameable": []}
    ys = [t[2] for t in tg]
    cache = {}
    for (w, pos, y, land, conf, alt, _l05) in tg:
        if w not in cache:
            tok = toks[w].tolist()
            cache[w] = (tok, [w2d[x] for x in tok if x in w2d], {s_[1]: s_[0] for s_ in slots[w]},
                        {s_[0]: s_[1] for s_ in slots[w]})
        tok, dirs, pos_of, move_of = cache[w]
        k = move_of[pos]
        rc = tok[pos_of[k - 2]] if (k >= 2 and dirs[k] == opp[dirs[k - 1]]) else nothing
        out["rcopy"].append(rc == y)
        nc = None
        if tok[pos - 3] in names and tok[pos - 4] == env.mark_id:
            prev = [p_ for p_ in pos_of.values() if p_ < pos and tok[p_ - 3] == tok[pos - 3] and tok[p_ - 4] == env.mark_id]
            if prev:
                nc = tok[max(prev)]
        out["nameable"].append(nc is not None)
        out["ncopy"].append((nc if nc is not None else nothing) == y)
        out["ncopy+rcopy"].append((nc if nc is not None else rc) == y)
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["constant"] = max(Counter(ys).values()) / len(ys)
    return res


def ngrams(train_walks, toks, tg, fallback):
    res = {}
    for n in range(1, 6):
        tab = defaultdict(Counter)
        for t, slots in train_walks:
            for (pos, k, cell, land, rev, conf, alt) in slots:
                if rev:
                    tab[tuple(t[pos - n:pos])][t[pos]] += 1
        ok = []
        for (w, pos, y, *_r) in tg:
            ctx = tuple(toks[w, pos - n:pos].tolist())
            ok.append((tab[ctx].most_common(1)[0][0] if tab[ctx] else fallback) == y)
        res[n] = float(np.mean(ok))
    return res


def main():
    env = TextWorldLandmark(seed=10000)
    print(f"vocab {env.unified_vocab_size} (58 text-world words + '{env.vocab[env.mark_id]}' + {len(env.name_ids)} names)")
    np.random.seed(EVAL_SEED)
    for c in [(0.5, "consistent"), (1.0, "fresh"), (1.0, "conflict")]:
        t = env.generate_trajectory(1024, name_rate=c[0], mode=c[1])[0]
        print(f"sample {c}: " + " ".join(env.vocab[i] for i in t[:70].tolist()))
    tr_env = TextWorldLandmark(seed=MAP_TRAIN)
    summary = {}
    for r in (0.0, 0.5, 1.0):
        E = build_eval(r, 200)
        # revisit fraction and slots per walk on the own rendering (all moves, not only common ones)
        so = E["_slots"]["own"]
        nslot = np.mean([len(s) for s in so]); rf = np.mean([sum(x[4] for x in s) / len(s) for s in so])
        toks, tg = E["own"]
        land = np.mean([t[3] for t in tg]); u05 = sum(not t[6] for t in tg)
        assert int(max(int(E[c][0].max()) for c in conds_for(r))) < env.unified_vocab_size
        print(f"\n=== training rate r = {r}: own rendering {nslot:.1f} object slots / walk, revisit fraction {rf:.3f}; "
              f"scored common-move revisit targets {len(tg)} ({len(tg) / 200:.1f} / walk); "
              f"share of scored revisits whose place is named before (landmark): {land:.3f}; "
              f"in-distribution probe subset (cells unnamed at rate 0.5, Amendment 1): {u05} targets")
        ys = [t[2] for t in tg]
        lnd = [t[2] for t in tg if t[3]]; unn = [t[2] for t in tg if not t[3]]
        if lnd and unn:
            cl, cu = Counter(lnd), Counter(unn)
            print(f"  leak check: best constant on landmark targets {max(cl.values()) / len(lnd):.3f}, on unnamed "
                  f"{max(cu.values()) / len(unn):.3f}, overall {max(Counter(ys).values()) / len(ys):.3f}; "
                  f"top answer landmark '{env.vocab[cl.most_common(1)[0][0]]}' unnamed '{env.vocab[cu.most_common(1)[0][0]]}'")
        for cond in ("own", "strip", "uninf", "named", "conf"):
            if r == 0.0 and cond in ("uninf",):
                continue
            key = conds_for(r)[cond]
            if r == 1.0 and cond == "named":
                continue
            toks, tg = E[cond]
            fl = rules(toks, tg, E["_slots"][cond], env)
            np.random.seed(7)
            trw = []
            for _ in range(N_TRAIN):
                t, o, rv = tr_env.generate_trajectory(1024, name_rate=key[0], mode=key[1])
                trw.append((t.tolist(), tr_env.slots))
            ng = ngrams(trw, toks, tg, env.idx["nothing"])
            best = max(fl["constant"], fl["rcopy"], fl["ncopy"], fl["ncopy+rcopy"], *ng.values())
            summary[(r, cond)] = {**fl, **{f"ngram{n}": v for n, v in ng.items()}, "best": best}
            print(f"  [{cond:5s} rate {key[0]} {key[1]:10s}] const {fl['constant']:.4f} | rcopy {fl['rcopy']:.4f} | "
                  f"ncopy {fl['ncopy']:.4f} (nameable {fl['nameable']:.3f}) | ncopy+rcopy {fl['ncopy+rcopy']:.4f} | "
                  f"n-gram " + " ".join(f"{n}:{v:.4f}" for n, v in ng.items()) + f" | BEST TRIVIAL {best:.4f}")
    # name -> object association ACROSS walks (names redrawn per walk): must be at the constant floor
    np.random.seed(11); tab = defaultdict(Counter)
    for _ in range(N_TRAIN):
        t, o, rv = tr_env.generate_trajectory(1024, name_rate=1.0); t = t.tolist()
        for (pos, *_r) in tr_env.slots:
            tab[t[pos - 3]][t[pos]] += 1
    E = build_eval(1.0, 200); toks, tg = E["own"]
    ok = [(tab[int(toks[w, pos - 3])].most_common(1)[0][0] if tab[int(toks[w, pos - 3])] else env.idx["nothing"]) == y
          for (w, pos, y, *_r) in tg]
    print(f"\nname -> object table fit across {N_TRAIN} training walks (map {MAP_TRAIN}), applied to the rate-1 eval "
          f"targets: {np.mean(ok):.4f} (constant floor {summary[(1.0, 'own')]['constant']:.4f})")
    # direction words only in movement clauses: every direction word is followed (within the clause) by a seeing
    # phrase before the next '.', and the inserted clause adds no direction word
    dirs = {i for a in range(4) for i in env.dir_ids[a]}; dot = env.idx["."]
    np.random.seed(3); bad = 0
    for _ in range(200):
        t = env.generate_trajectory(1024, name_rate=1.0)[0].tolist()
        for i, x in enumerate(t):
            if x in dirs and (i == 0 or t[i - 1] not in {env.idx[v] for v in
                                                        ["walked", "went", "moved", "stepped", "headed", "wandered",
                                                         "slowly", "quickly", "carefully", "quietly"]}):
                bad += 1
    print(f"direction words not directly after a verb or adverb (outside a movement clause): {bad}")


if __name__ == "__main__":
    main()
