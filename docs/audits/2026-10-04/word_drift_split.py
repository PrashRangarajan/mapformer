"""Verifies the formal-theory claim that TW_NORMSTEP's registered per-word drift (verdict B) is mostly the aside
sentences. drift_opt_rad split by source: ASIDE positions (after the move's '.', up to the next move's verb) vs
ADVERB/FILLER positions (inside the move clause). Same walks and definition as tw_normstep_readouts.drift."""
import numpy as np, torch
torch.set_num_threads(8)
from mapformer.environment_textworld import TextWorld, VERBS
from mapformer.tw_normstep_readouts import load, core_mask, wrap
from mapformer.model_textstep import step_of
R = "/home/prashr/mapformer/runs/tw_normstep/p0"


@torch.no_grad()
def split(m, n_walks=40, T=1024):
    te = TextWorld(size=64, seed=10000); om = m.path_integrator.omega.detach().numpy(); verbset = set(te.idx[w] for w in VERBS)
    np.random.seed(0); dif = {"opt": [], "aside": [], "clause": []}
    for _ in range(n_walks):
        tok, obs, _ = te.generate_trajectory(T); locs = te.visited_locations
        d = step_of(m, tok[None])[0].numpy(); t = tok.numpy()
        slots = np.nonzero(obs.numpy())[0][:len(locs)]; core = core_mask(te, t, slots)
        aside = np.zeros(len(t), bool)
        for i in slots:                                  # aside = after "obj ." up to the next verb
            j = i + 2
            while j < len(t) and t[j] not in verbset:
                aside[j] = True; j += 1
        opt = ~core; masks = {"opt": opt, "aside": opt & aside, "clause": opt & ~aside}
        ths = {k: np.cumsum(d * v[:, None, None], 0) * om[None] for k, v in masks.items()}
        first = {}
        for k, i in enumerate(slots):
            L = tuple(locs[k])
            if L in first:
                for key, th in ths.items(): dif[key].append(wrap(th[i] - th[first[L]]))
            else: first[L] = i
    return {k: float(np.abs(np.array(v)).mean(0).mean()) for k, v in dif.items()}


for arm in ("MapWM", "NormStep"):
    rows = []
    for s in range(10, 18):
        m, _ = load(f"{R}/{arm}_s{s}/{arm}.pt"); r = split(m); rows.append(r)
        print(f"{arm:9s} s{s}: optional {r['opt']:.4f} rad = aside {r['aside']:.4f} + adverb/filler {r['clause']:.4f} (not additive: wrapped means)", flush=True)
    print(f"{arm:9s} mean: optional {np.mean([r['opt'] for r in rows]):.4f}  aside {np.mean([r['aside'] for r in rows]):.4f}  adverb/filler {np.mean([r['clause'] for r in rows]):.4f}")
