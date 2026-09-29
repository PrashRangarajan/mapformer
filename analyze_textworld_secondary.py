"""Declared secondaries for TEXTWORLD_PREREG.md (Amendment 1, written 2026-09-28 before any batch
result was read). They qualify, never replace, the registered verdicts A and B.

  S1 common/differential decomposition of the step table (rule 8: every movement clause holds one
     verb and one direction word, so a vector moved from the direction words onto the verbs leaves
     every object-slot phase unchanged). c = mean of the 4 synonym-averaged direction steps;
     opposition and |cos NE| of (direction - c); |c|/|N|; cos(c, mean verb step).
  S2 omega-scaled table: move ratio / opposition / synonym cos on Delta * omega (the (Delta*k,
     omega/k) gauge is exact, so the raw table is gauge-dependent).
  S3 functional ablation: held-out accuracy at T=1024 with Delta zeroed for every NON-direction
     word, and with Delta zeroed for the direction words.
  S4 cross-class cosines, the reference synonym cosine must be read against: mean |cos| between
     different directions, and within verbs / seeing-phrase words.
  S5 floors on the registered eval set (np seed 0, 200 trials, map 10000, T=1024): best constant, and
     a reversal-copy rule (if this move reverses the previous one, copy the object from two steps
     back, else the constant) that word n-grams cannot express.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_textworld import TextWorld, VERBS, SEE
from mapformer.probe_textworld import table
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/textworld/p0"; S = range(8)


def n(x):
    return float(np.linalg.norm(x))


def cos(x, y):
    return float(x @ y / (n(x) * n(y)))


def opp(A):
    return [n(A[0] + A[1]) / ((n(A[0]) + n(A[1])) / 2), n(A[2] + A[3]) / ((n(A[2]) + n(A[3])) / 2)]


def load(ck, dev):
    blob = torch.load(ck, map_location="cpu", weights_only=False); a = blob["config"]["args"]
    env = TextWorld(size=a["size"], seed=0)
    m = VARIANT_MAP[a["variant"]](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2,
                                  n_layers=a["n_layers"], grid_size=a["size"])
    m.load_state_dict(blob["model_state_dict"]); return m.to(dev).eval()


@torch.no_grad()
def eval_masked(m, keep, dev, T=1024, trials=200):
    """Accuracy with Delta multiplied by keep[token] (1 = unchanged, 0 = zeroed)."""
    te = TextWorld(size=64, seed=10000); np.random.seed(0)
    cur = {}
    h = m.action_to_lie.register_forward_hook(lambda mod, i, o: o * cur["k"][..., None, None])
    ok = tot = 0
    for _ in range(trials):
        tok, _o, rev = te.generate_trajectory(T)
        tok = tok.unsqueeze(0).to(dev); cur["k"] = keep[tok[:, :-1]]
        lp = F.log_softmax(m(tok[:, :-1]).float(), -1); mk = rev[1:].to(dev)
        ok += (lp.argmax(-1)[0][mk] == tok[0, 1:][mk]).sum().item(); tot += int(mk.sum())
    h.remove(); return ok / tot


def floors(T=1024, trials=200):
    te = TextWorld(size=64, seed=10000); np.random.seed(0)
    nothing = te.idx["nothing"]; opp_of = {0: 1, 1: 0, 2: 3, 3: 2}
    word2dir = {i: a for a in range(4) for i in te.dir_ids[a]}
    ys, rc = [], []
    for _ in range(trials):
        tok, obs, rev = te.generate_trajectory(T); tok = tok.tolist()
        slots = np.nonzero(obs.numpy())[0]
        dirs = []                                    # the direction word of each step, in order
        for i, w in enumerate(tok):
            if w in word2dir:
                dirs.append(word2dir[w])
        for k, i in enumerate(slots):
            if not rev[i]:
                continue
            ys.append(tok[i])
            if k >= 2 and dirs[k] == opp_of[dirs[k - 1]]:
                rc.append(tok[slots[k - 2]] == tok[i])
            else:
                rc.append(tok[i] == nothing)
    const = max(np.mean([y == v for y in ys]) for v in set(ys))
    return const, float(np.mean(rc)), len(ys)


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = {}
    const, rcopy, nt = floors()
    print(f"== S5 floors on the eval set ({nt} targets): best constant {const:.4f}; reversal-copy {rcopy:.4f}")
    out["floors"] = {"constant": const, "reversal_copy": rcopy, "n": nt}
    print("\n== S1-S4 per path seed (raw Delta; S2 in brackets = Delta * omega) ==")
    for s in S:
        ck = f"{R}/Vanilla_r4_L1_s{s}/Vanilla_r4.pt"
        env, D = table(ck)
        m = load(ck, dev)
        om = m.path_integrator.omega.detach().cpu().numpy().reshape(-1)
        row = {}
        for tag, X in (("raw", D), ("omega", D * om[None, :])):
            A = {a: X[env.dir_ids[a]].mean(0) for a in range(4)}
            c = np.mean([A[a] for a in range(4)], 0); v = X[[env.idx[w] for w in VERBS]].mean(0)
            Rr = {a: A[a] - c for a in range(4)}
            dir_ids = [i for a in range(4) for i in env.dir_ids[a]]
            other = [i for i in range(len(env.vocab)) if i not in dir_ids]
            nn_ = np.linalg.norm(X, axis=1)
            syn = np.mean([cos(X[i], X[j]) for a in range(4) for x, i in enumerate(env.dir_ids[a])
                           for j in env.dir_ids[a][x + 1:]])
            row[tag] = {"move_ratio": float(nn_[other].mean() / nn_[dir_ids].mean()),
                        "opposition_raw": float(np.mean(opp(A))), "opposition_minus_common": float(np.mean(opp(Rr))),
                        "cosNE_minus_common": abs(cos(Rr[0], Rr[3])), "common_over_N": n(c) / n(A[0]),
                        "cos_common_verb": cos(c, v), "synonym_cos": syn}
        X = D
        A = {a: X[env.dir_ids[a]].mean(0) for a in range(4)}
        cross = np.mean([abs(cos(A[a], A[b])) for a in range(4) for b in range(a + 1, 4)])
        vids = [env.idx[w] for w in VERBS]
        verb = np.mean([cos(X[i], X[j]) for x, i in enumerate(vids) for j in vids[x + 1:]])
        sids = list(dict.fromkeys(env.idx[w] for p in SEE for w in p))
        see = np.mean([cos(X[i], X[j]) for x, i in enumerate(sids) for j in sids[x + 1:]])
        row["S4"] = {"cross_direction_abs_cos": float(cross), "verb_cos": float(verb), "see_cos": float(see)}
        V = len(env.vocab); dmask = torch.zeros(V, device=dev)
        for a in range(4):
            dmask[env.dir_ids[a]] = 1.0
        row["S3"] = {"full": eval_masked(m, torch.ones(V, device=dev), dev),
                     "non_direction_zeroed": eval_masked(m, dmask, dev),
                     "direction_zeroed": eval_masked(m, 1 - dmask, dev)}
        out[f"s{s}"] = row
        r, o = row["raw"], row["omega"]
        print(f"  s{s}: opp {r['opposition_raw']:.3f} -> minus common {r['opposition_minus_common']:.3f} [{o['opposition_minus_common']:.3f}]"
              f" | |c|/|N| {r['common_over_N']:.3f} cos(c,verb) {r['cos_common_verb']:+.3f}"
              f" | move {r['move_ratio']:.3f} [{o['move_ratio']:.3f}] | syn {r['synonym_cos']:.3f} vs cross-dir {row['S4']['cross_direction_abs_cos']:.3f}"
              f" verbs {row['S4']['verb_cos']:.3f} see {row['S4']['see_cos']:.3f}"
              f" | acc full {row['S3']['full']:.3f} non-dir zeroed {row['S3']['non_direction_zeroed']:.3f} dir zeroed {row['S3']['direction_zeroed']:.3f}")
    json.dump(out, open(f"{REPO}/TEXTWORLD_SECONDARY.json", "w"), indent=1)


if __name__ == "__main__":
    main()
