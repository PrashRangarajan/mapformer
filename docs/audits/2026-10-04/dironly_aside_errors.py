"""Post hoc (TW_NORMSTEP_RESULTS.md): why does DirOnly (steps on direction words only) stop at 0.972 on every seed?
Hypothesis: an aside sentence ("she thought about a cat .") follows the object of a move, so with no step on any
non-direction word the aside's object noun sits at the same phase as the cell's real object; a revisit query then
also matches it. Per run: errors on revisit targets split by whether the predicted word appeared as an ASIDE object
at an earlier visit to the same cell. Held-out map, eval stream np seed 10**6, 100 walks, eval mode. Also MapWM."""
import sys, numpy as np, torch
from mapformer.environment_textworld import TextWorld
from mapformer.tw_normstep_readouts import load
R = "/home/prashr/mapformer/runs/tw_normstep/p0"
for arm, s in [("DirOnly", 10), ("DirOnly", 13), ("MapWM", 16), ("NormStep", 12)]:
    m, _ = load(f"{R}/{arm}_s{s}/{arm}.pt"); te = TextWorld(size=64, seed=10000); np.random.seed(10**6)
    objs = set(te.obj_ids); n = err = err_aside = 0; aside_avail = 0
    with torch.no_grad():
        for _ in range(100):
            tok, obs, rev = te.generate_trajectory(1024); locs = te.visited_locations; t = tok.numpy()
            slots = list(np.nonzero(obs.numpy())[0][:len(locs)])
            pred = m(tok[None, :-1])[0].argmax(-1).numpy()
            asides = {}                                     # cell -> aside object words seen so far
            for k, i in enumerate(slots):
                L = tuple(locs[k]); end = slots[k + 1] if k + 1 < len(slots) else len(t)
                if rev[i]:
                    n += 1; a = asides.get(L, set()); aside_avail += bool(a - {t[i]})
                    if pred[i - 1] != t[i]:
                        err += 1; err_aside += pred[i - 1] in a
                seg = t[i + 2:end]                          # words after "obj ." up to the next move's object
                j = next((x for x in range(len(seg)) if seg[x] == te.idx["."]), None)
                if j is not None and j > 0 and seg[j - 1] in objs and seg[j - 1] != te.idx["nothing"]:
                    asides.setdefault(L, set()).add(int(seg[j - 1]))
    print(f"{arm:9s} s{s}: revisit acc {1 - err / n:.4f}; errors {err}, of which predicted an earlier ASIDE object "
          f"at that cell {err_aside} ({err_aside / max(err, 1):.0%}); targets with a different aside object at the cell "
          f"{aside_avail / n:.3f}")
