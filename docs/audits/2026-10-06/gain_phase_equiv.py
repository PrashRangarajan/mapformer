"""Construction and equivalence checks for GAIN_PHASE_PREREG.md (CPU, untrained models built exactly as the trainer
builds them: torch/np seeded, env built, arm built, use_object_codes). Output: gain_phase_equiv_out.txt.

1 matched initialisation at every batch seed (8-15) and pilot seed (110, 111): MapWM = NormStep on every MapWM tensor,
  GainRaw = GainPhase on every GainRaw tensor, MapWM = GainRaw on every tensor they both have (all but the score's content
  projections); global CPU RNG state and initial seed identical after construction across the four arms (so the data
  stream -- ParallelBatchGenerator's base seed is torch.initial_seed() -- and the later draws are the same).
2 limits: NormStep with step_ln = identity equals MapWM; GainPhase with step_ln = identity equals GainRaw; GainRaw equals
  model_em_pope.MapFormerWM_GainScalar(bottleneck_r=4) with the same weights (the wrapper adds nothing to the score);
  positive controls differ.
3 causality per arm; 4 rank 4, 32 angles per head, identical initial omega; parameter counts.
5 the gain score reads the code embedding: object gains depend on the code, are invariant to its norm (LayerNorm),
  the score's input is the token embedding itself, and the encoder A gets gradient through the gains.
6 train-mode RNG consumption: one train-mode forward leaves the CPU RNG in the same state in all four arms.
"""
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/prashr")
from mapformer.environment_newobj import NewObjectWorld, N_SPECIAL
from mapformer.model_codes import use_object_codes, set_pool
from mapformer.model_em_pope import MapFormerWM_GainScalar, GainKernelLayer
from mapformer.model_gain_phase import ARMS
from mapformer import train_newobj
from mapformer.model_gain_phase import register

register(train_newobj.ARMS)
torch.set_num_threads(4)
P, SIZE = 1000, 32
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def build(arm, seed, cls=None):
    """train_newobj.main's construction order."""
    torch.manual_seed(seed); np.random.seed(seed)
    env = NewObjectWorld(size=SIZE, seed=seed, pool_size=P, pool="train")
    m = (cls or train_newobj.ARMS[arm])(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1,
                                         grid_size=SIZE)
    use_object_codes(m, N_SPECIAL, P); set_pool(m, "train")
    return m, torch.get_rng_state().clone(), torch.initial_seed()


def sd(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


def dmax(a, b):
    """max |a - b| over finite logits (the inactive pool is -inf by construction); inf if the -inf patterns differ."""
    f = torch.isfinite(a)
    return (a[f] - b[f]).abs().max().item() if torch.equal(f, torch.isfinite(b)) else float("inf")


def walks(n=4, T=256, seed=7):
    env = NewObjectWorld(size=SIZE, seed=10000, pool_size=P, pool="test"); np.random.seed(seed)
    return torch.stack([env.generate_trajectory(T)[0] for _ in range(n)])[:, :-1]


print("== 1. matched initialisation ==")
for s in list(range(8, 16)) + [110, 111]:
    B = {a: build(a, s) for a in ARMS}
    S = {a: sd(B[a][0]) for a in ARMS}
    w, n, g, p = S["MapWM"], S["NormStep"], S["GainRaw"], S["GainPhase"]
    eq = lambda A, Bd, keys: all(torch.equal(A[k], Bd[k]) for k in keys)
    shared = sorted(set(w) & set(g))
    onlyw = sorted(set(w) - set(g)); onlyg = sorted(set(g) - set(w))
    r1 = eq(w, n, w) and sorted(set(n) - set(w)) == ["step_ln.bias", "step_ln.weight"]
    r2 = eq(g, p, g) and sorted(set(p) - set(g)) == ["step_ln.bias", "step_ln.weight"]
    r3 = eq(w, g, shared)
    r4 = all(torch.equal(B[a][1], B["MapWM"][1]) and B[a][2] == B["MapWM"][2] for a in ARMS)
    r5 = not torch.equal(g["layers.0.q_gain.weight"], build("GainRaw", s + 1000)[0].state_dict()["layers.0.q_gain.weight"])
    check(f"seed {s}", r1 and r2 and r3 and r4 and r5,
          f"MapWM=NormStep {r1} GainRaw=GainPhase {r2} MapWM=GainRaw on {len(shared)} shared tensors {r3} "
          f"RNG state + initial seed equal {r4} gain init seed-dependent {r5}")
    if s == 8:
        print(f"    rotary-only tensors: {onlyw}\n    gain-only tensors: {onlyg}")

print("\n== 2. limits (eval mode, 4 held-out walks of 255 tokens, test pool) ==")
x = walks()
for s in (8, 110):
    M = {a: build(a, s)[0].eval() for a in ARMS}
    for a in ARMS:
        set_pool(M[a], "test")
    gen = torch.Generator().manual_seed(s)              # move step_ln off its (1, 0) init, identically in both arms
    with torch.no_grad():
        dw, db = 0.3 * torch.randn(128, generator=gen), 0.3 * torch.randn(128, generator=gen)
        for a in ("NormStep", "GainPhase"):
            M[a].step_ln.weight.add_(dw); M[a].step_ln.bias.add_(db)
        gb = sd(M["GainRaw"])
        lg = {a: M[a](x) for a in ARMS}
        for a, ref in (("NormStep", "MapWM"), ("GainPhase", "GainRaw")):
            ln = M[a].step_ln; M[a].step_ln = nn.Identity()
            d0 = dmax(M[a](x), lg[ref]); M[a].step_ln = ln
            d1 = dmax(lg[a], lg[ref])
            check(f"seed {s}: {a} with step_ln = identity equals {ref}", d0 == 0.0 and d1 > 1e-3,
                  f"(max |dlogit| {d0:.1e}; positive control, with step_ln: {d1:.1e})")
        gs = build("GainRaw", s, cls=lambda **kw: MapFormerWM_GainScalar(bottleneck_r=4, **kw))[0].eval()
        set_pool(gs, "test"); gs.load_state_dict(gb)
        d2 = dmax(gs(x), lg["GainRaw"])
        gs.layers[0].amp_raw.add_(0.5); d3 = dmax(gs(x), lg["GainRaw"])
        check(f"seed {s}: GainRaw equals MapFormerWM_GainScalar(r=4) with the same weights", d2 == 0.0 and d3 > 1e-3,
              f"(max |dlogit| {d2:.1e}; positive control A_c x softplus shift: {d3:.1e})")

print("\n== 3. causality / 4. rank, angles, omega, parameters ==")
om = None
for a in ARMS:
    m = build(a, 8)[0].eval(); set_pool(m, "test")
    with torch.no_grad():
        y0 = m(x); x2 = x.clone(); t = 300; x2[:, t] = (x2[:, t] + 1) % 4          # position 300: an action token
        y1 = m(x2)
    leak = dmax(y1[:, :t], y0[:, :t]); moved = dmax(y1[:, t:], y0[:, t:])
    o = m.path_integrator.omega.detach()
    om = o if om is None else om
    check(f"{a}: causal", leak == 0.0 and moved > 0, f"(max |dlogit| before t {leak:.1e}, at/after t {moved:.1e})")
    check(f"{a}: rank 4, 32 angles/head, omega as MapWM's",
          m.action_to_lie.w_in.out_features == 4 and m.n_blocks == 32 and torch.equal(o, om),
          f"(r {m.action_to_lie.w_in.out_features}, n_blocks {m.n_blocks}, omega {o[0, 0]:.4f} .. {o[0, -1]:.4f}; "
          f"params {sum(p.numel() for p in m.parameters()):,})")

print("\n== 5. the gain score reads the code embedding ==")
for a in ("GainRaw", "GainPhase"):
    m = build(a, 8)[0].eval(); L = m.layers[0]
    assert isinstance(L, GainKernelLayer) and L.n_modules == 1
    seen = {}
    h1 = L.norm1.register_forward_pre_hook(lambda mod, args: seen.__setitem__("in", args[0]))
    h2 = m.token_emb.register_forward_hook(lambda mod, args, out: seen.__setitem__("emb", out))
    with torch.no_grad():
        m(x)
    h1.remove(); h2.remove()
    check(f"{a}: the score layer's input is the token embedding (CodeEmbedding output)", torch.equal(seen["in"], seen["emb"]))
    ids = torch.tensor([[N_SPECIAL + P, N_SPECIAL + P + 1, N_SPECIAL + P + 2]])
    with torch.no_grad():
        e = m.token_emb(ids); gq, gk = L.gains(L.norm1(e))
        codes = m.token_emb.codes.clone(); m.token_emb.codes = codes * 2.0
        gq2, gk2 = L.gains(L.norm1(m.token_emb(ids))); m.token_emb.codes = codes
    spread = (gk[0, :, :, 0].max(1).values - gk[0, :, :, 0].min(1).values).min().item()
    inv = max((gq2 - gq).abs().max().item(), (gk2 - gk).abs().max().item())
    check(f"{a}: object gains depend on the code and not on its norm", spread > 1e-4 and inv < 1e-5,
          f"(gain range over 3 codes {spread:.2e}; x2 code norm changes gains by {inv:.1e})")
    m.train(False); m.zero_grad()
    for prm in m.parameters():
        prm.requires_grad_(False)
    m.token_emb.encoder.weight.requires_grad_(True)
    hq, hk = L.norm1(m.token_emb(x[:1])), L.norm1(m.token_emb(x[:1]))
    cos_a, sin_a = m.path_integrator(m.step(x[:1], m.token_emb(x[:1])))
    sc = L.score(hq, hk, cos_a, sin_a, cos_a, sin_a); sc.sum().backward()
    gnorm = m.token_emb.encoder.weight.grad.norm().item()
    check(f"{a}: the code encoder A gets gradient through the gain score", gnorm > 0, f"(|grad| {gnorm:.2e})")

print("\n== 6. train-mode RNG consumption ==")
st = {}
for a in ARMS:
    m = build(a, 8)[0].train(); torch.manual_seed(123)
    with torch.no_grad():
        m(x)
    st[a] = torch.get_rng_state()
check("one train-mode forward consumes the same RNG in all four arms", all(torch.equal(st[a], st["MapWM"]) for a in ARMS))
print(f"\nALL {'PASS' if ok_all else 'FAIL'}")
