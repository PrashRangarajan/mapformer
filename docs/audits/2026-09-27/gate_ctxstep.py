"""Gate for the context-step task and models (CONTEXT_STEP_DESIGN.md), CPU, calling the task code."""
from collections import Counter, defaultdict
import numpy as np, torch
from mapformer.environment_textworld import TextWorld, DIRS
from mapformer.environment_textworld_ctx import TextWorldCtx
from mapformer.environment import GridWorld
from mapformer.model_context_step import CtxGateWM, HiddenStepWM
from mapformer.train_variant import VARIANT_MAP

# 1. p_decoy = 0 reproduces TextWorld exactly
for s in (0, 7):
    np.random.seed(s); a = TextWorld(seed=3).generate_trajectory(1024)
    np.random.seed(s); b = TextWorldCtx(seed=3, p_decoy=0.0).generate_trajectory(1024)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "p_decoy=0 differs from TextWorld"
print("1. p_decoy=0 stream identical to TextWorld: PASS")

# 2. decoys do not move the walker: integrate only 'move' direction words
te = TextWorldCtx(seed=10000, p_decoy=0.3); np.random.seed(1)
w2a = {te.idx[w]: a for a, ws in DIRS.items() for w in ws}
bad = cls = 0; kinds = Counter()
for _ in range(100):
    tok, obs, rev = te.generate_trajectory(1024); tl = tok.tolist()
    start = te.grid.visited_locations[0]
    moves = [i for i, k in te.ctx if k == "move"]; kinds.update(k for _, k in te.ctx)
    assert all(tl[i] in w2a for i, _ in te.ctx)
    pos = np.array(start) - np.array(GridWorld.ACTION_DELTAS[w2a[tl[moves[0]]]])
    for j, i in enumerate(moves[:len(te.visited_locations)]):
        pos = (pos + np.array(GridWorld.ACTION_DELTAS[w2a[tl[i]]])) % 64
        bad += tuple(pos) != tuple(te.visited_locations[j])
    n_dir = sum(t in w2a for t in tl); cls += n_dir != len(te.ctx)
print(f"2. walk reproduced from move-class words only: mismatches {bad}; every direction word classed: {cls == 0}; classes {dict(kinds)}")

# 3. floors on the eval set (seed 10**6, 200 trials): constant, reversal-copy over MOVE steps, n-grams 1-5 (train map 0)
def targets(env, n, seed):
    np.random.seed(seed); out = []
    for _ in range(n):
        tok, obs, rev = env.generate_trajectory(1024); tl = tok.tolist()
        slots = np.nonzero(obs.numpy())[0]; mv = [tl[i] for i, k in env.ctx if k == "move"]
        out.append((tl, slots, rev.numpy(), mv))
    return out
E = targets(TextWorldCtx(seed=10000, p_decoy=0.3), 200, 10**6); Tr = targets(TextWorldCtx(seed=0, p_decoy=0.3), 800, 5)
opp = {0: 1, 1: 0, 2: 3, 3: 2}; nothing = te.idx["nothing"]
ys, rc = [], []
for tl, slots, rev, mv in E:
    for k, i in enumerate(slots):
        if rev[i]:
            ys.append(tl[i])
            rc.append(tl[slots[k - 2]] == tl[i] if k >= 2 and w2a[mv[k]] == opp[w2a[mv[k - 1]]] else tl[i] == nothing)
const = Counter(ys).most_common(1)[0][1] / len(ys)
ng = {}
for n in range(1, 6):
    tab = defaultdict(Counter)
    for tl, slots, rev, mv in Tr:
        for i in np.nonzero(rev)[0]:
            tab[tuple(tl[i - n:i])][tl[i]] += 1
    mode = Counter(y for tl, s, rev, mv in Tr for y in [tl[i] for i in np.nonzero(rev)[0]]).most_common(1)[0][0]
    ng[n] = np.mean([(tab[tuple(tl[i - n:i])].most_common(1)[0][0] if tab[tuple(tl[i - n:i])] else mode) == tl[i]
                     for tl, slots, rev, mv in E for i in np.nonzero(rev)[0]])
steps = np.mean([len(s) for _, s, _, _ in E]); rvf = len(ys) / sum(len(s) for _, s, _, _ in E)
print(f"3. eval set: {len(ys)} targets, {steps:.1f} object slots/seq, revisit {rvf:.3f}; floors const {const:.4f}, "
      f"reversal-copy {np.mean(rc):.4f}, n-gram " + " ".join(f"{n}:{v:.3f}" for n, v in ng.items()))

# 4. models: shared init, gate start, causal leak, params
V = te.unified_vocab_size
def build(cls, s, L):
    torch.manual_seed(s); np.random.seed(s); return cls(vocab_size=V, d_model=128, n_heads=2, n_layers=L, grid_size=64)
for s in (0, 5):
    cf, cg = build(VARIANT_MAP["Vanilla_r4"], s, 1), build(CtxGateWM, s, 1)
    cf2, hs = build(VARIANT_MAP["Vanilla_r4"], s, 2), build(HiddenStepWM, s, 2)
    for a_, b_, nm in ((cf, cg, "CG vs CF"), (cf2, hs, "HS vs CF2")):
        sa, sb = a_.state_dict(), b_.state_dict(); sh = [k for k in sa if k in sb]
        print(f"4. seed {s} {nm}: {len(sh)}/{len(sa)} shared tensors, max diff {max((sa[k] - sb[k]).abs().max().item() for k in sh):.1e}; "
              f"params {sum(p.numel() for p in b_.parameters())} vs {sum(p.numel() for p in a_.parameters())}")
t = torch.randint(0, V, (2, 256)); t2 = t.clone(); t2[:, 200:] = torch.randint(0, V, (2, 56))
for nm, m in (("CG", cg), ("HS", hs), ("SR", build(VARIANT_MAP["SRoPEGen"], 0, 1)), ("RoPE2", build(VARIANT_MAP["RoPE"], 0, 2))):
    m.eval()
    with torch.no_grad():
        leak = (m(t)[:, :200] - m(t2)[:, :200]).abs().max().item()
    print(f"4. {nm}: causal leak {leak:.1e}, params {sum(p.numel() for p in m.parameters())}")
with torch.no_grad():
    print(f"4. CG initial gate mean {cg.gate(cg.token_emb(t)).mean().item():.3f}")
