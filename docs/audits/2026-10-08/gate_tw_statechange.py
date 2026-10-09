"""Rule-11 gate for TW_STATECHANGE_PREREG.md, CALLING the task code (environment_tw_statechange.StateChangeWorld) and the
registered floor code (tw_statechange_readouts.floors / replay_check). CPU only.

Eval stream = the batch's: held-out map env seed 10000, np seed 10**6, 200 walks, T = 1024 words. Word n-grams (orders
1-5 over the preceding words at revisit slots) are fit on 1500 walks of a TRAINING map (env seed 0; np seed 1), as the
text-world gate. Prints: vocabulary and a render sample; construction checks (no direction word or movement verb in a
state clause; no state word in a core position; every state clause starts after its own step's object slot; T2 answers
differ from the last 'saw' at the cell; the replay check -- targets rebuilt from tokens + locations + map); stream
statistics (words per move, state-clause share, strata); floors per stratum; the C floor F on T2drop; and the stream
statistics at p_take = p_drop in {0.2, 0.4, 0.6} (the chosen 0.4 is justified by them)."""
from collections import Counter

import numpy as np

from mapformer import tw_statechange_readouts as R
from mapformer.environment_tw_statechange import CLS, TAKE, DROP, DET
from mapformer.environment_textworld import DIRS, VERBS
from mapformer.train_tw_statechange import make_env

T = 1024
te = make_env(R.HELDOUT)
print(f"vocab {te.unified_vocab_size}: {te.vocab}")
np.random.seed(3); tok, _o, _r = te.generate_trajectory(T)
print("sample:", " ".join(te.vocab[i] for i in tok[:90].tolist()))

W = R.walks(te)
assert max(int(w["tok"].max()) for w in W) < te.unified_vocab_size
# ---- construction checks
dirw = {te.idx[w] for d in DIRS.values() for w in d}; mverb = {te.idx[w] for w in VERBS}
statew = {te.idx[w] for w in TAKE + DROP + [DET]}
bad_sc = bad_core = bad_order = bad_t2 = 0
for w in W:
    t = w["tok"].numpy(); pc = w["pc"]
    sc = np.isin(pc, (CLS["take"], CLS["drop"]))
    bad_sc += int(np.isin(t[sc], list(dirw | mverb)).sum())
    core = np.isin(pc, (CLS["verb"], CLS["dir"], CLS["see"], CLS["obj"], CLS["period"]))
    bad_core += int(np.isin(t[core], list(statew)).sum())
    slot_of = {}
    for k, s in enumerate(w["info"]):
        if s["stratum"] in ("T2take", "T2drop"):
            bad_t2 += int(s["answer"] == s["last_saw"])
    pos = [s["pos"] for s in w["info"]]
    for kind, a, e, c in w["clauses"]:
        if kind in ("take", "drop"):
            prev = max(p for p in pos if p < a)
            k = pos.index(prev)
            bad_order += int(tuple(w["locs"][k]) != c or w["events"][k] is None or w["events"][k][0] != kind)
print(f"\nchecks: direction/movement words inside state clauses {bad_sc}; state words in core positions {bad_core}; "
      f"state clauses not attached to the preceding slot's cell/event {bad_order}; T2 answers equal to last saw {bad_t2}")
rp = [R.replay_check(w, te) for w in W]
print(f"replay check (targets rebuilt from tokens + slot locations + map): {sum(rp)}/{len(rp)} walks reproduce every slot")
assert bad_sc == bad_core == bad_order == bad_t2 == 0 and all(rp)


def stats(W, label):
    nsl = [len(w["info"]) for w in W]; ntok = sum(len(w["tok"]) for w in W)
    ev = Counter(e[0] for w in W for e in w["events"] if e is not None)
    pcs = np.concatenate([w["pc"] for w in W])
    st = Counter(s["stratum"] for w in W for s in w["info"])
    rev = sum(v for k, v in st.items() if k != "first")
    print(f"{label}: moves/seq {np.mean(nsl):.1f}, words/move {ntok / sum(nsl):.2f}; state clauses per move: take "
          f"{ev['take'] / sum(nsl):.3f}, drop {ev['drop'] / sum(nsl):.3f}; aside {sum(1 for w in W for c in w['clauses'] if c[0] == 'aside') / sum(nsl):.3f}; "
          f"token share: state clauses {np.isin(pcs, (8, 9)).mean():.3f}, asides {np.mean(pcs == 7):.3f}, "
          f"optional total {np.isin(pcs, (1, 3, 7, 8, 9)).mean():.3f}")
    print(f"   revisit targets/seq {rev / len(W):.1f} (fraction of slots {rev / sum(nsl):.3f}); strata share of revisit targets: "
          + ", ".join(f"{k} {st[k] / rev:.3f} ({st[k]})" for k in ("T1", "T2take", "T2drop", "T3")))
    s1 = sum(1 for w in W for s in w["info"] if s["stratum"] == "T1" and s["straddle_sc"])
    print(f"   T1 targets straddling >= 1 state clause (T1s): {s1}")


print()
stats(W, "p_take = p_drop = 0.4 (registered)")
fl = R.floors(te, W, ngram_env=make_env(0))
cols = ("all", "T1", "T1s", "T2take", "T2drop", "T3")
print(f"\nfloors on the eval stream (accuracy of each rule, by stratum; n = {[fl['counts'].get(c, 0) for c in cols]}):")
print(f"  {'rule':14s}" + "".join(f"{c:>9s}" for c in cols))
for r in ("constant", "ngram1", "ngram2", "ngram3", "ngram4", "ngram5", "revcopy", "revcopy_state", "last_dropped",
          "first_saw", "last_saw"):
    print(f"  {r:14s}" + "".join(f"{fl[r].get(c, float('nan')):9.4f}" for c in cols))
print(f"  constant word: '{fl['constant_word']}'")
F, rule = R.nonpath_floor(fl, "T2drop")
print(f"\nC floor F (best location-free / short-range rule on T2drop): {F:.4f} ({rule})")
best_ng = max(fl[f"ngram{n}"]["all"] for n in range(1, 6))
print(f"no word n-gram beats the constant on all revisit targets: {best_ng:.4f} vs {fl['constant']['all']:.4f} -> "
      f"{'OK' if best_ng <= fl['constant']['all'] + 0.005 else 'CHECK'}")
for p in (0.2, 0.6):
    e = make_env(R.HELDOUT, p_take=p, p_drop=p); stats(R.walks(e, n=100), f"\np_take = p_drop = {p} (100 walks)")
