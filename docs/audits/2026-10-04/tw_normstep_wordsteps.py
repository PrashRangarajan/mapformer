"""Post hoc (TW_NORMSTEP_RESULTS.md): where NormStep's extra per-word drift comes from. Per run: mean ||step|| of the
OPTIONAL word types (adverbs, fillers, aside-only words) / mean ||direction-word step||, the same for the core
non-direction words (verbs, seeing words, '.', objects), and the embedding norm of optional words / of direction
words. Phase steps omega-scaled (the omega/step gauge cancels)."""
import json, numpy as np, torch
from mapformer.environment_textworld import TextWorld, ADVERBS, FILLERS, ASIDE, VERBS, SEE, DIRS
from mapformer.tw_normstep_readouts import load
from mapformer.model_textstep import step_of
R = "/home/prashr/mapformer/runs/tw_normstep/p0"; res = json.load(open("/home/prashr/mapformer/TW_NORMSTEP.json"))
env = TextWorld(size=64, seed=0); I = env.idx
dirs = [i for a in range(4) for i in env.dir_ids[a]]
core_words = set(VERBS) | {w for p in SEE for w in p} | {".", "nothing"}
opt_words = (set(ADVERBS) | set(FILLERS) | {w for p in ASIDE for w in p}) - core_words
opt = sorted(I[w] for w in opt_words if w in I)
core = sorted(I[w] for w in core_words if w in I)
print(f"optional word types {len(opt)}, core non-direction {len(core)}")
for arm in ("MapWM", "NormStep", "NormStepNB"):
    for s in range(10, 18):
        m, _ = load(f"{R}/{arm}_s{s}/{arm}.pt")
        with torch.no_grad():
            om = m.path_integrator.omega.detach()
            D = (step_of(m, torch.arange(env.unified_vocab_size)[None])[0] * om).reshape(env.unified_vocab_size, -1)
            n = D.norm(dim=1); e = m.token_emb.weight.norm(dim=1)
        r = res[f"{arm}_s{s}"]
        print(f"{arm:10s} s{s}: opt/dir step {float(n[opt].mean() / n[dirs].mean()):.4f}  core/dir {float(n[core].mean() / n[dirs].mean()):.4f}"
              f"  emb opt/dir {float(e[opt].mean() / e[dirs].mean()):.3f}  word drift {r['drift_opt_rad']:.4f}  drift {r['drift']:2d}  acc {r['acc']:.4f}")
