"""Rule-11 gate for TW_AMBIG_PREREG.md, CPU only, calling environment_tw_ambig.TextWorldAmbig (the task code the
trainer runs). Output: gate_tw_ambig_out.txt.

  1  render sample, vocabulary, tokens / steps / revisit fraction per sequence
  2  the walk is reproduced from MOVE-role direction words alone (non-movement uses do not move the walker)
  3  share of non-movement direction words, overall, per class and per synonym (role must not be readable
     from the word itself)
  4  role decidability: a frequency table on the LOCAL window (4 tokens before + 3 after, jointly) fit on a
     training-map sample, scored on the eval sample, per frame; and a sentence-level cue rule (decidable from
     context by construction); measured cue distances per class
  5  floors on the registered eval stream (map 10000, np seed 10**6, 200 walks, T=1024): best constant,
     reversal-copy rule (move-role words), word n-grams of order 1-5 fit on a training-map sample
  6  path oracles on the same stream: CLEAN (true cell), CONTAMINATED (every direction word integrated as a move:
     the context-free step's best case, the CF-ceiling estimate), DIRONLY-CAP (true cell, aside nouns stored at
     the cell, as a steps-on-move-words-only model would see them) -- the cap oracle is validated on plain
     TextWorld against TW_NORMSTEP's DirOnly (0.970-0.974)
  7  leak / construction: tagged stream = untagged after untag, tags exactly at non-movement direction words;
     the reported object in lead-form non-movement sentences is the current cell's; p_nm=0 equals TextWorld
"""
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
from mapformer.environment import GridWorld
from mapformer.environment_textworld import TextWorld, OBJECTS
from mapformer.environment_tw_ambig import (TextWorldAmbig, ALL_DIR, LEAD_NM, TRAIL_NM, NM_CLASSES, FORMS)

T, N_EVAL, N_TRAIN, EVAL_SEED, HELDOUT = 1024, 200, 600, 10**6, 10000
OPP = {0: 1, 1: 0, 2: 3, 3: 2}
NM_CUES = set(LEAD_NM + TRAIL_NM + ["pointing", "sound", "cold", "wind"])


def sample(env, seed, n):
    np.random.seed(seed); out = []
    for _ in range(n):
        tok, obs, rev = env.generate_trajectory(T)
        out.append(dict(t=tok.tolist(), obs=obs.numpy(), rev=rev.numpy(), ctx=list(getattr(env, "ctx", [])),
                        locs=[tuple(map(int, l)) for l in env.visited_locations],
                        nm=np.array(getattr(env, "nm_mask", [False] * len(tok)), bool)))
    return out


def w2a_of(env):
    return {env.idx[w]: a for a, ws in __import__("mapformer.environment_textworld", fromlist=["DIRS"]).DIRS.items()
            for w in ws}


def walk_check(env, E):
    w2a = w2a_of(env); bad = 0; n = 0
    for e in E:
        mv = [i for i, role, f in e["ctx"] if role == "move"]
        pos = None
        for j, i in enumerate(mv[:len(e["locs"])]):
            d = np.array(GridWorld.ACTION_DELTAS[w2a[e["t"][i]]])
            pos = np.array(e["locs"][0]) if pos is None else (pos + d) % 64
            bad += tuple(pos) != e["locs"][j]; n += 1
    return bad, n


def oracles(env, E, mode):
    """Most-recent-observation predictors at revisit targets. mode: 'clean' (true cell), 'contam' (every direction
    word moves the position), 'dironly' (true cell; aside nouns are stored at the current cell too).
    Memory: movement-clause object slots; for 'contam', the lead-form non-movement objects at their (contaminated)
    position as well; fallback: reversal-copy, else 'nothing'. -> (acc, acc on clean gaps, acc on contaminated
    gaps, n clean, n contam); a gap is contaminated if a non-movement direction word lies between the most recent
    earlier visit of the target's cell and the target."""
    w2a = w2a_of(env); nothing = env.idx["nothing"]; obj_ids = set(env.idx[o] for o in OBJECTS[:16])
    res = {"all": [], "clean": [], "contam": []}
    for e in E:
        t, obs, rev = e["t"], e["obs"], e["rev"]
        slots = list(np.nonzero(obs)[0]); locs = e["locs"]
        dir_at = {i: (role, f) for i, role, f in e["ctx"]}
        mem = {}; true_pos = None; cpos = np.zeros(2, int); started = False
        last_visit = {}; nm_seen = 0; nm_at_visit = {}; k = 0; mv_dirs = []
        for i, w in enumerate(t):
            if i in dir_at:
                role, f = dir_at[i]; a = w2a[w]; d = np.array(GridWorld.ACTION_DELTAS[a])
                cpos = (cpos + d) % 64
                if role == "move":
                    mv_dirs.append(a)
                else:
                    nm_seen += 1
            if mode != "clean" and w in obj_ids | {nothing} and not obs[i] and i > 0:
                # an object word outside a movement slot: an aside noun or a reported (lead-form) object
                in_nm = e["nm"][i]
                if mode == "contam" and in_nm:
                    mem[("c", tuple(cpos))] = w
                if mode == "dironly" and not in_nm and k > 0:
                    mem[("t", locs[k - 1])] = w
            if obs[i] and k < len(locs):
                cell = locs[k]
                key = ("c", tuple(cpos)) if mode == "contam" else ("t", cell)
                if rev[i]:
                    if key in mem:
                        pred = mem[key]
                    elif k >= 2 and len(mv_dirs) > k and mv_dirs[k] == OPP[mv_dirs[k - 1]]:
                        pred = t[slots[k - 2]]
                    else:
                        pred = nothing
                    ok = pred == w
                    res["all"].append(ok)
                    gap = "contam" if nm_seen > nm_at_visit.get(cell, nm_seen) else "clean"
                    res[gap].append(ok)
                mem[key] = w
                nm_at_visit[cell] = nm_seen
                k += 1
    m = lambda v: float(np.mean(v)) if v else float("nan")
    return m(res["all"]), m(res["clean"]), m(res["contam"]), len(res["clean"]), len(res["contam"])


def calibrate():
    """The contaminated oracle against a TRAINED context-free model: ctx3 pilot CF s0 (runs/ctxstep3_pilot, lead/far and
    trail/far decoys, p_decoy 0.3, T=2048), 60 eval walks (np seed 10**6, map 10000), accuracy split by gap."""
    from mapformer.environment_textworld_ctx3 import TextWorldCtx3
    from mapformer.model_rank import MapFormerWM_r4
    global T
    T0 = T; T = 2048
    for cue in ("lead", "trail"):
        env = TextWorldCtx3(seed=HELDOUT, cue=cue, dist="far"); E = sample(env, EVAL_SEED, 60)
        for e in E:
            e["ctx"] = [(i, "move" if k == "move" else "nm", "x") for i, k in e["ctx"]]
            nm = np.zeros(len(e["t"]), bool)
            for i, role, f in e["ctx"]:
                j = i
                while role == "nm" and j < len(e["t"]) and env.vocab[e["t"][j]] != ".":
                    nm[j] = True; j += 1
            e["nm"] = nm
        o = oracles(env, E, "contam")
        b = torch.load(f"/home/prashr/mapformer/runs/ctxstep3_pilot/{cue}_far_CF_L1_s0/CF.pt", map_location="cpu",
                       weights_only=False)
        m = MapFormerWM_r4(vocab_size=b["config"]["vocab_size"], d_model=128, n_heads=2, n_layers=1, grid_size=64)
        m.load_state_dict(b["model_state_dict"]); m.eval(); ok = {"clean": [], "contam": []}
        for e in E:
            with torch.no_grad():
                pred = m(torch.tensor(e["t"])[None, :-1])[0].argmax(-1)
            nm_dir = set(i for i, r, f in e["ctx"] if r == "nm"); seen = 0; last = {}; k = 0
            for i in range(len(e["t"])):
                seen += i in nm_dir
                if e["obs"][i] and k < len(e["locs"]):
                    c = e["locs"][k]
                    if e["rev"][i]:
                        ok["contam" if seen > last.get(c, seen) else "clean"].append(int(pred[i - 1]) == e["t"][i])
                    last[c] = seen; k += 1
        allm = ok["clean"] + ok["contam"]
        print(f"   calibration, ctx3 {cue}/far T=2048: contaminated oracle {o[0]:.4f} (contaminated gaps {o[2]:.4f}); "
              f"trained CF s0 {np.mean(allm):.4f} (clean gaps {np.mean(ok['clean']):.4f}, contaminated "
              f"{np.mean(ok['contam']):.4f})")
    T = T0
    print("   -> the contaminated oracle is an ESTIMATE, not a bound: a trained context-free model beat it on lead/far.")


def floors(env, E, Tr):
    w2a = w2a_of(env); nothing = env.idx["nothing"]; ys, rc = [], []
    for e in E:
        slots = np.nonzero(e["obs"])[0]
        mv = [w2a[e["t"][i]] for i, role, f in e["ctx"] if role == "move"] if e["ctx"] else \
            [w2a[w] for w in e["t"] if w in w2a]
        for k, i in enumerate(slots):
            if e["rev"][i]:
                ys.append(e["t"][i])
                rc.append(e["t"][slots[k - 2]] == e["t"][i] if k >= 2 and mv[k] == OPP[mv[k - 1]] else e["t"][i] == nothing)
    const = max(np.mean([y == v for y in ys]) for v in set(ys))
    ng = {}
    for n in range(1, 6):
        tab = defaultdict(Counter)
        for e in Tr:
            for i in np.nonzero(e["rev"])[0]:
                tab[tuple(e["t"][i - n:i])][e["t"][i]] += 1
        mode = Counter(e["t"][i] for e in Tr for i in np.nonzero(e["rev"])[0]).most_common(1)[0][0]
        ng[n] = float(np.mean([(tab[tuple(e["t"][i - n:i])].most_common(1)[0][0] if tab[tuple(e["t"][i - n:i])] else mode)
                               == e["t"][i] for e in E for i in np.nonzero(e["rev"])[0]]))
    return float(const), float(np.mean(rc)), ng, len(ys)


def main():
    torch.set_num_threads(1)
    te = TextWorldAmbig(seed=HELDOUT); tr = TextWorldAmbig(seed=7)
    E = sample(te, EVAL_SEED, N_EVAL); Tr = sample(tr, 5, N_TRAIN)
    print(f"== 1 vocabulary {te.unified_vocab_size} (TextWorld 58 + {te.unified_vocab_size - 58} new): {te.vocab[58:]}")
    print("sample:", " ".join(te.vocab[i] for i in E[0]["t"][:160]))
    steps = np.mean([int(e["obs"].sum()) for e in E]); rv = np.mean([e["rev"].sum() / max(1, e["obs"].sum()) for e in E])
    print(f"T={T}: object slots / seq {steps:.1f} (TextWorld ~132), revisit fraction {rv:.3f} (TextWorld 0.231), "
          f"scored targets / seq {np.mean([e['rev'].sum() for e in E]):.1f}")

    bad, n = walk_check(te, E)
    print(f"\n== 2 walk from move-role direction words only: {bad} mismatches of {n} steps")

    kinds = Counter((role, f) for e in E for _, role, f in e["ctx"])
    nmn = sum(v for (r, f), v in kinds.items() if r == "nm"); tot = sum(kinds.values())
    print(f"\n== 3 direction words: {tot}; non-movement {nmn} = {nmn / tot:.3f} of direction words; per seq "
          f"{nmn / len(E):.1f} non-movement, {(tot - nmn) / len(E):.1f} moves")
    print("   by (role, form):", {f"{r}/{f}": v for (r, f), v in sorted(kinds.items())})
    syn = defaultdict(Counter)
    for e in E:
        for i, role, f in e["ctx"]:
            syn[te.vocab[e["t"][i]]][role] += 1
    print("   P(non-movement | synonym):", " ".join(f"{w} {syn[w]['nm'] / sum(syn[w].values()):.3f}" for w in ALL_DIR))

    print("\n== 4 role decidability")
    def win(e, i, kind):
        b = tuple(e["t"][max(0, i - 4):i]); a = tuple(e["t"][i + 1:i + 4])
        return {"before4": b, "after3": a, "joint": b + ("|",) + a}[kind]
    def feats(e, i):                                  # positional word features of the 7-token window
        return [(o, e["t"][i + o] if 0 <= i + o < len(e["t"]) else -1) for o in (-4, -3, -2, -1, 1, 2, 3)]
    nb = defaultdict(Counter); prior = Counter()
    for e in Tr:
        for i, role, f in e["ctx"]:
            prior[role] += 1
            for ft in feats(e, i):
                nb[ft][role] += 1
    def nb_pred(e, i):                                # naive Bayes, Laplace-smoothed: generalises over unseen windows
        lp = {r: np.log(prior[r]) for r in ("move", "nm")}
        for ft in feats(e, i):
            for r in lp:
                lp[r] += np.log((nb[ft][r] + 1) / (prior[r] + 100))
        return max(lp, key=lp.get)
    tabs = {k: defaultdict(Counter) for k in ("before4", "after3", "joint")}
    for e in Tr:
        for i, role, f in e["ctx"]:
            for k in tabs:
                tabs[k][win(e, i, k)][role] += 1
    groups = {g: [] for g in ["base/nat"] + FORMS}
    for e in E:
        for i, role, f in e["ctx"]:
            g = f if f in FORMS else "base/nat"
            row = {}
            for k, tab in tabs.items():
                w = win(e, i, k); row[k] = (tab[w].most_common(1)[0][0] if tab[w] else "move") == role
            row["naive_bayes"] = nb_pred(e, i) == role
            groups[g].append((row, role))
    print("   per frame: role predicted from the LOCAL window by frequency tables (before-4, after-3, joint; unseen ->"
          " 'move') and by a smoothed naive-Bayes over the 7 positional words, fit on a training-map sample, scored on"
          " the eval sample, vs the majority base rate within the frame")
    for g, v in groups.items():
        base = max(np.mean([r == "nm" for _, r in v]), np.mean([r == "move" for _, r in v]))
        print(f"   {g:11s}: " + "  ".join(f"{k} {np.mean([x[k] for x, _ in v]):.3f}" for k in
                                         ("before4", "after3", "joint", "naive_bayes"))
              + f"  | base rate {base:.3f}  (n {len(v)})")
    ok = []; dist = defaultdict(list)
    for e in E:
        t = e["t"]
        for i, role, f in e["ctx"]:
            a = i
            while a > 0 and te.vocab[t[a - 1]] != ".":
                a -= 1
            b = i
            while b < len(t) - 1 and te.vocab[t[b]] != ".":
                b += 1
            cues = [j for j in range(a, b + 1) if te.vocab[t[j]] in NM_CUES]
            ok.append((len(cues) > 0) == (role == "nm"))
            if role == "nm" and cues:
                dist[f].append(min(abs(j - i) for j in cues))
    print(f"   sentence-level cue rule (nm iff the sentence holds a non-movement cue word): {np.mean(ok):.4f} correct "
          f"(n {len(ok)})")
    print("   cue distance to the direction word (tokens), non-movement classes:",
          {f: (min(v), max(v)) for f, v in sorted(dist.items())})

    const, rcopy, ng, nt = floors(te, E, Tr)
    print(f"\n== 5 floors on the eval stream ({nt} targets): best constant {const:.4f}; reversal-copy {rcopy:.4f}; "
          f"word n-gram " + " ".join(f"{n}:{v:.4f}" for n, v in ng.items()))

    print("\n== 6 path oracles (most recent observation at the same key; fallback reversal-copy, else 'nothing')")
    for mode in ("clean", "contam", "dironly"):
        a, ac, an, nc, nn = oracles(te, E, mode)
        print(f"   {mode:8s}: {a:.4f}  (clean gaps {ac:.4f} n {nc}; contaminated gaps {an:.4f} n {nn})")
    tw = TextWorld(seed=HELDOUT); Ew = sample(tw, EVAL_SEED, N_EVAL)
    for e in Ew:                              # plain TextWorld: every direction word is a move
        w2a = w2a_of(tw); e["ctx"] = [(i, "move", "base") for i, w in enumerate(e["t"]) if w in w2a]
    for mode in ("clean", "dironly"):
        a, *_ = oracles(tw, Ew, mode)
        print(f"   validation on plain TextWorld, {mode:8s}: {a:.4f}  (TW_NORMSTEP DirOnly 0.970-0.974 eval mode)")
    print("   -> the aside-overwrite ('dironly') oracle FAILS its validation (0.856 vs a trained DirOnly's 0.97): not used.")
    calibrate()
    for p in (0.15, 0.45):
        Ep = sample(TextWorldAmbig(seed=HELDOUT, p_nm=p), EVAL_SEED, N_EVAL)
        a, ac, an, *_ = oracles(te, Ep, "contam")
        print(f"   dose (no verdict): p_nm {p}: contaminated oracle {a:.4f}")

    print("\n== 7 leak / construction")
    tt = TextWorldAmbig(seed=HELDOUT, tag_roles=True); Et = sample(tt, EVAL_SEED, 30)
    same = all(torch.equal(tt.untag(torch.tensor(a["t"])), torch.tensor(b["t"])) for a, b in zip(Et, E[:30]))
    tag_ok = all(set(i for i, w in enumerate(a["t"]) if w >= tt.tag_offset) == set(i for i, r, f in a["ctx"] if r == "nm")
                 for a in Et)
    print(f"   tagged stream == untagged after untag: {same}; tags exactly at non-movement direction words: {tag_ok}")
    rep = []
    for e in E:
        slots = list(np.nonzero(e["obs"])[0]); k = 0
        for i, role, f in e["ctx"]:
            if role == "nm" and f.startswith("lead"):
                j = i + 1
                while j < len(e["t"]) and te.vocab[e["t"][j]] not in OBJECTS[:16] + ["nothing"]:
                    j += 1
                prev = [s for s in slots if s < i]
                if j < len(e["t"]) and prev:
                    rep.append(e["t"][j] == e["t"][prev[-1]])
    print(f"   lead-form reported object == the current cell's (last movement slot's) object: {np.mean(rep):.4f} "
          f"(n {len(rep)})")
    a = TextWorld(seed=3); b = TextWorldAmbig(seed=3, p_nm=0.0); same0 = True
    for k in range(20):
        np.random.seed(k); x = a.generate_trajectory(T); np.random.seed(k); y = b.generate_trajectory(T)
        same0 &= all(torch.equal(u, v) for u, v in zip(x, y)) and a.visited_locations == b.visited_locations
    print(f"   p_nm=0 stream byte-identical to TextWorld (20 walks): {same0}")
    tids = set(np.concatenate([np.array(e["t"])[e["rev"]] for e in E]).tolist())
    print(f"   targets are object words only: {tids <= set(te.idx[o] for o in OBJECTS[:16] + ['nothing'])}; "
          f"max token id {max(max(e['t']) for e in E)} < vocab {te.unified_vocab_size}")


if __name__ == "__main__":
    main()
