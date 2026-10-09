"""Which word classes carry position, and does identity within a class matter? (post hoc, CPU, eval-only)
Stored text-world models (runs/textworld/p0 Vanilla_r4 s0-7; runs/tw_normstep/p0 MapWM / NormStep / NormStepNB s10-17).
Steps are context-free, so each model's step is a table D[token]; the forward is rebuilt with a modified table
(verified equal to the model's own logits). Held-out map 10000, eval walks np seed 10**6, 100 walks, T=1024, with the
attention-dropout scale correction (x1/(1-p), docs/audits/2026-10-04/DROPOUT_RESCORE.md; one-layer models). 40 walks,
batched (CPU).
Interventions per class C (tokens assigned to exactly one class; words shared between lists form class 'shared'):
  mean-sub(C): every token in C gets C's mean step -> cost = what identity WITHIN the class carries (gauge-safe).
  zero(C): C's steps set to 0 -> cost = what the class's steps carry at all. For classes that occur exactly once per
           move (verb, direction, see, object slot, '.') zeroing is gauge-dependent (a constant can move between them);
           reported, but read mean-sub for those. Optional classes (adverb, filler, aside-only) are gauge-free."""
import math, sys, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(8)
from mapformer.environment_textworld import TextWorld, DIRS, VERBS, ADVERBS, FILLERS, SEE, OBJECTS, ASIDE
from mapformer.model import _apply_rope
from mapformer.analyze_textworld_secondary import load as load_tw
from mapformer.tw_normstep_readouts import load as load_ns
from mapformer.model_textstep import step_of

te = TextWorld(size=64, seed=10000); I = te.idx
lists = {"direction": [w for d in DIRS.values() for w in d], "verb": VERBS, "adverb": ADVERBS, "filler": FILLERS,
         "see": [w for p in SEE for w in p], "object": OBJECTS[:16], "nothing": ["nothing"], "period": ["."],
         "aside": [w for p in ASIDE for w in p]}
cnt = {}
for c, ws in lists.items():
    for w in set(ws):
        cnt.setdefault(w, []).append(c)
CLS = {}
for w, cs in cnt.items():
    CLS.setdefault(cs[0] if len(cs) == 1 else "shared", []).append(I[w])
np.random.seed(10**6); W = [te.generate_trajectory(1024) for _ in range(40)]
TOK = torch.stack([w[0] for w in W]); REV = torch.stack([w[2] for w in W])


@torch.no_grad()
def fwd(m, tok, D, scale=True):
    x = m.token_emb(tok); cos_a, sin_a = m.path_integrator(D[tok]); T = tok.shape[1]
    cm = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for L in m.layers:
        B = x.shape[0]; h = L.norm1(x)
        Q = _apply_rope(L.q_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        K = _apply_rope(L.k_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2), cos_a, sin_a)
        V = L.v_proj(h).view(B, T, L.n_heads, L.d_head).transpose(1, 2)
        a = F.softmax((Q @ K.transpose(-1, -2) / math.sqrt(L.d_head)).masked_fill(cm, float("-inf")), -1)
        if scale:
            a = a / (1 - L.dropout.p)                              # dropout-scale correction
        x = x + L.o_proj((a @ V).transpose(1, 2).reshape(B, T, L.d_model))
        x = x + L.ffn[2](L.ffn[1](L.ffn[0](L.norm2(x))))
    return m.out_proj(m.out_norm(x))


@torch.no_grad()
def acc(m, D):
    ok = tot = 0
    for i in range(0, len(TOK), 10):
        tok, rev = TOK[i:i + 10], REV[i:i + 10]
        lg = fwd(m, tok[:, :-1], D); msk = rev[:, 1:]
        ok += int((lg.argmax(-1)[msk] == tok[:, 1:][msk]).sum()); tot += int(msk.sum())
    return ok / tot


runs = [("textworld MapWM", f"runs/textworld/p0/Vanilla_r4_L1_s{s}/Vanilla_r4.pt", load_tw) for s in range(8)]
for arm in ("MapWM", "NormStep"):
    runs += [(f"tw_normstep {arm}", f"runs/tw_normstep/p0/{arm}_s{s}/{arm}.pt", load_ns) for s in range(10, 18)]
order = ["direction", "verb", "see", "object", "nothing", "period", "adverb", "filler", "aside", "shared"]
res = {}
for lab, path, ld in runs:
    p = "/home/prashr/mapformer/" + path
    m = ld(p, "cpu") if ld is load_tw else ld(p)[0]
    m.eval()
    with torch.no_grad():
        D = step_of(m, torch.arange(te.unified_vocab_size)[None])[0]          # (V, H, nb)
        tok = TOK[:2, :-1]; err = (fwd(m, tok, D, scale=False) - m(tok)).abs().max().item()
        assert err < 1e-4, ("rebuilt forward differs from the model", path, err)
    base = acc(m, D); row = {"base": base}
    for c in order:
        ids = torch.tensor(CLS.get(c, []))
        if len(ids) == 0:
            continue
        Dm = D.clone(); Dm[ids] = D[ids].mean(0, keepdim=True); Dz = D.clone(); Dz[ids] = 0
        row[c] = (acc(m, Dm) - base, acc(m, Dz) - base)
    res.setdefault(lab, []).append(row)
    print(lab, path.split("/")[-2], f"base {base:.4f}  " + "  ".join(f"{c}:{row[c][0]:+.3f}/{row[c][1]:+.3f}" for c in order if c in row), flush=True)

print("\n== medians over seeds: accuracy change with class steps mean-substituted / zeroed (dropout-scale corrected) ==")
for lab, rows in res.items():
    print(f"{lab:22s} base {np.median([r['base'] for r in rows]):.4f}  " + "  ".join(
        f"{c} {np.median([r[c][0] for r in rows]):+.3f}/{np.median([r[c][1] for r in rows]):+.3f}" for c in order if c in rows[0]))
print("classes:", {c: [te.vocab[i] for i in CLS[c]] for c in order if c in CLS})
