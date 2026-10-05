"""Construction / equivalence checks for GAIN_GRAIN_PREREG.md (model_em_pope.py). CPU, float64 where stated.
Output: gain_grain_equiv_out.txt. Every check prints PASS / FAIL; positive controls must FAIL (differ).

1. Nesting: GainScalar (M=1) c GainMod4 (M=4) c GainMod32 (M=32) == MapPoPE-Pair with the two element gains of each
   angle tied and delta = 0, at A_c = 1. Each inclusion: build the larger model, copy the smaller one's weights with
   gain rows tied within its groups, compare logits on held-out torus walks.
2. Pure position kernel: GainScalar with its gains fixed at 1 == MapEM with content factor fixed at 1 (A_X = 1) and
   q0_c = k0_c = (sqrt(2 A_c), 0), i.e. a MapEM-PosOnly-like model with zero phase offsets.
3. Sign arm: VanillaEM_NonNeg with content_map = identity == VanillaEM, bit for bit; matched initialisation.
4. Matched initialisation of the gain arms with MapPoPE-Pair (all but the score's content projections) at the batch's
   seeds and the pilot's.
5. Causality (0 leak), angles per head, omega range, rank, parameter counts, module -> omega bands.
"""
import math
import sys

import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, "/home/prashr")
from mapformer import train_variant
from mapformer.model_em_pope import register, MapFormerWM_GainMod32, _inv_softplus
from mapformer.environment import GridWorld

V = register(train_variant.VARIANT_MAP)
ARMS6 = ["Vanilla", "MapPoPE-Pair", "VanillaEM", "VanillaEM_NonNeg", "GainScalar", "GainMod4"]
TOL = 1e-5
fails = []


def build(name, seed, cls=None):
    torch.manual_seed(seed)
    return (cls or V[name])(vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64).eval()


def walks(n=4, T=128):
    env = GridWorld(size=64, n_obs_types=16, seed=10000); np.random.seed(0)
    return torch.stack([env.generate_trajectory(T)[0] for _ in range(n)])[:, :-1]


def check(name, a, b, expect_equal=True, tol=TOL):
    d = (a - b).abs().max().item()
    ok = (d <= tol) if expect_equal else (d > 1e-3)
    tag = "PASS" if ok else "FAIL"
    if not ok:
        fails.append(name)
    print(f"  [{tag}] {name}: max |dlogit| {d:.2e}" + ("" if expect_equal else "  (positive control: must differ)"))


def copy_shared(dst, src):
    """Copy every parameter whose name exists in both with the same shape (embeddings, bottleneck, omega, readout,
    v/o/norms/FFN)."""
    sd = src.state_dict(); dd = dst.state_dict(); n = 0
    for k, v in sd.items():
        if k in dd and dd[k].shape == v.shape:
            dd[k] = v.clone(); n += 1
    dst.load_state_dict(dd)
    return n


toks = walks()
H, NB, DH = 2, 32, 64

print("== 1. nesting chain (eval mode, 4 held-out walks of T=128, 255 input tokens) ==")
with torch.no_grad():
    # (a) Mod32 at A_c = 1  ==  MapPoPE-Pair with element gains tied within each angle's pair and delta = 0
    P = build("MapPoPE-Pair", 7)
    for nm in ("q_proj", "k_proj"):                                       # random but non-trivial biases
        getattr(P.layers[0], nm).bias.normal_(0, 0.5)
    raw = P(toks)
    G32 = build(None, 8, MapFormerWM_GainMod32); copy_shared(G32, P)
    L, Lp = G32.layers[0], P.layers[0]
    for g, p in (("q_gain", "q_proj"), ("k_gain", "k_proj")):
        W, b = getattr(Lp, p).weight.view(H, DH, -1), getattr(Lp, p).bias.view(H, DH)
        W[:, 1::2] = W[:, 0::2]; b[:, 1::2] = b[:, 0::2]                 # tie element 2c+1 to 2c (in place in P)
        getattr(L, g).weight.copy_(W[:, 0::2].reshape(H * NB, -1)); getattr(L, g).bias.copy_(b[:, 0::2].reshape(-1))
    assert float(Lp.pope_delta.abs().max()) == 0.0
    check("Mod32 (A_c = 1) == MapPoPE-Pair, element gains tied, delta 0", G32(toks), P(toks))
    check("  control: MapPoPE-Pair UNTIED (its own random q/k rows)", G32(toks), raw, expect_equal=False)
    Lp.pope_delta.fill_(-0.5)
    check("  control: MapPoPE-Pair tied but delta = -0.5", G32(toks), P(toks), expect_equal=False)
    Lp.pope_delta.zero_()
    L.amp_raw.fill_(_inv_softplus(1.5))
    check("  control: Mod32 with A_c = 1.5", G32(toks), P(toks), expect_equal=False)

    # (b) Mod4 == Mod32 with gain rows tied within each module of 8 contiguous channels; A_c random
    G4 = build("GainMod4", 9)
    G4.layers[0].amp_raw.normal_(0, 1.0)
    for nm in ("q_gain", "k_gain"):
        getattr(G4.layers[0], nm).bias.normal_(0, 0.5)
    G32 = build(None, 10, MapFormerWM_GainMod32); copy_shared(G32, G4)
    G32.layers[0].amp_raw.copy_(G4.layers[0].amp_raw)
    for nm in ("q_gain", "k_gain"):
        src = getattr(G4.layers[0], nm); dst = getattr(G32.layers[0], nm)
        dst.weight.copy_(src.weight.view(H, 4, -1).repeat_interleave(8, dim=1).reshape(H * NB, -1))
        dst.bias.copy_(src.bias.view(H, 4).repeat_interleave(8, dim=1).reshape(-1))
    check("Mod4 == Mod32 with rows tied within contiguous modules (channels 8m..8m+7)", G4(toks), G32(toks))
    for nm in ("q_gain", "k_gain"):                                       # wrong grouping: channel c -> module c % 4
        src = getattr(G4.layers[0], nm); dst = getattr(G32.layers[0], nm)
        dst.weight.copy_(src.weight.view(H, 4, -1).repeat(1, 8, 1).reshape(H * NB, -1))
        dst.bias.copy_(src.bias.view(H, 4).repeat(1, 8).reshape(-1))
    check("  control: Mod32 tied with INTERLEAVED modules (c % 4)", G4(toks), G32(toks), expect_equal=False)

    # (c) Scalar == Mod4 with all four module rows equal
    G1 = build("GainScalar", 11)
    G1.layers[0].amp_raw.normal_(0, 1.0)
    for nm in ("q_gain", "k_gain"):
        getattr(G1.layers[0], nm).bias.normal_(0, 0.5)
    G4 = build("GainMod4", 12); copy_shared(G4, G1)
    G4.layers[0].amp_raw.copy_(G1.layers[0].amp_raw)
    for nm in ("q_gain", "k_gain"):
        src = getattr(G1.layers[0], nm); dst = getattr(G4.layers[0], nm)
        dst.weight.copy_(src.weight.view(H, 1, -1).repeat_interleave(4, dim=1).reshape(H * 4, -1))
        dst.bias.copy_(src.bias.view(H, 1).repeat_interleave(4, dim=1).reshape(-1))
    check("Scalar == Mod4 with all module rows equal", G1(toks), G4(toks))
    getattr(G4.layers[0], "q_gain").bias.view(H, 4)[:, 3] += 1.0
    check("  control: Mod4 with module 3's query bias moved by +1", G1(toks), G4(toks), expect_equal=False)

print("\n== 2. GainScalar with gains fixed at 1 == MapEM with A_X fixed at 1, q0_c = k0_c = (sqrt(2 A_c), 0) ==")
with torch.no_grad():
    G1 = build("GainScalar", 13); L1 = G1.layers[0]
    L1.amp_raw.normal_(0, 1.0)
    for nm in ("q_gain", "k_gain"):
        getattr(L1, nm).weight.zero_(); getattr(L1, nm).bias.fill_(_inv_softplus(1.0))
    E = build("VanillaEM", 14); copy_shared(E, G1); LE = E.layers[0]
    for nm in ("q_content", "k_content"):                                # A_X = (sqrt 8)^2 / 8 = 1 for every pair
        getattr(LE, nm).weight.zero_(); b = getattr(LE, nm).bias.view(H, DH); b.zero_(); b[:, 0] = math.sqrt(8.0)
    A = L1.amplitude()                                                   # (H, NB)
    E.q0_pos.zero_(); E.k0_pos.zero_()
    E.q0_pos.view(H, NB, 2)[..., 0] = torch.sqrt(2 * A); E.k0_pos.view(H, NB, 2)[..., 0] = torch.sqrt(2 * A)
    check("GainScalar(mu = 1) == MapEM(A_X = 1, zero phase offsets): a pure position kernel", G1(toks), E(toks))
    E.q0_pos.view(H, NB, 2)[..., 1] = 0.3
    check("  control: MapEM with a non-zero q0 phase offset", G1(toks), E(toks), expect_equal=False)

print("\n== 3. sign arm: VanillaEM_NonNeg ==")
with torch.no_grad():
    for s in (26, 45, 100):
        E, N = build("VanillaEM", s), build("VanillaEM_NonNeg", s)
        same = all(torch.equal(a, b) for a, b in zip(E.state_dict().values(), N.state_dict().values())) and \
            list(E.state_dict()) == list(N.state_dict())
        print(f"  [{'PASS' if same else 'FAIL'}] seed {s}: initial state_dict of VanillaEM_NonNeg == VanillaEM, bit for bit")
        if not same:
            fails.append(f"init NonNeg s{s}")
    E, N = build("VanillaEM", 26), build("VanillaEM_NonNeg", 26)
    E.q0_pos.normal_(0, 1.0); E.k0_pos.normal_(0, 1.0)                  # trained-scale q0/k0 so A_P is not ~0
    N.q0_pos.copy_(E.q0_pos); N.k0_pos.copy_(E.k0_pos)
    N.layers[0].content_map = lambda z: z                                 # instance attribute: identity
    check("NonNeg with content_map = identity == VanillaEM", N(toks), E(toks), tol=0.0)
    del N.layers[0].content_map
    check("  control: NonNeg with softplus (the trained arm)", N(toks), E(toks), expect_equal=False)

print("\n== 4. matched initialisation at the batch seeds (26-45) and the pilot seeds (100, 101) ==")
SHARED_WITH_PAIR = lambda k: not any(t in k for t in ("q_proj", "k_proj", "q_gain", "k_gain", "pope_delta", "amp_raw"))
for s in list(range(26, 46)) + [100, 101]:
    P = build("MapPoPE-Pair", s).state_dict(); W = build("Vanilla", s).state_dict()
    row = []
    for a in ("GainScalar", "GainMod4"):
        G = build(a, s).state_dict()
        ks = [k for k in G if SHARED_WITH_PAIR(k)]
        ok = all(torch.equal(G[k], P[k]) for k in ks) and set(ks) == {k for k in P if SHARED_WITH_PAIR(k)}
        row.append(f"{a} {'PASS' if ok else 'FAIL'} ({len(ks)} tensors)")
        if not ok:
            fails.append(f"init {a} s{s}")
    base = [k for k in W if k.split(".")[0] in ("token_emb", "action_to_lie", "path_integrator", "out_norm", "out_proj")]
    ok = all(torch.equal(W[k], P[k]) for k in base)
    row.append(f"Vanilla/Pair base {'PASS' if ok else 'FAIL'}")
    if s in (26, 45, 100, 101) or "FAIL" in " ".join(row):
        print(f"  seed {s}: " + "; ".join(row))
print("  (seeds 27-44 checked identically; printed only on failure)")
print("  shared with MapPoPE-Pair: token_emb, action_to_lie (rank-2 bottleneck), omega, layer v/o/norms/FFN, out_norm/out_proj;"
      "\n  not shared: the score's content projections (Pair q_proj/k_proj 64 per head; gain arms q_gain/k_gain 1 or 4 per head).")

print("\n== 5. causality, angles, omega, rank, parameters ==")
with torch.no_grad():
    for a in ARMS6:
        m = build(a, 0)
        t1 = toks[:1].clone(); t2 = t1.clone(); cut = 100
        t2[:, cut + 1:] = torch.randint(0, 21, t2[:, cut + 1:].shape, generator=torch.Generator().manual_seed(1))
        leak = (m(t1)[:, :cut + 1] - m(t2)[:, :cut + 1]).abs().max().item()
        om = m.path_integrator.omega
        npar = sum(p.numel() for p in m.parameters())
        ok = leak == 0.0 and om.shape == (2, 32) and m.action_to_lie.w_in.out_features == 2
        if not ok:
            fails.append(f"basic {a}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {a:17s} leak {leak:.1e}  angles/head {om.shape[1]}  omega {om.max().item():.4f}..{om.min().item():.4f}"
              f"  rank {m.action_to_lie.w_in.out_features}  params {npar:,}")
    om = build("Vanilla", 0).path_integrator.omega[0]
    same = all(torch.equal(build(a, 0).path_integrator.omega, build("Vanilla", 0).path_integrator.omega) for a in ARMS6)
    print(f"  [{'PASS' if same else 'FAIL'}] initial omega identical in all six arms (2pi = {2 * math.pi:.4f} .. 2pi/64 = {2 * math.pi / 64:.4f})")
    if not same:
        fails.append("omega")
    print("  GainMod4 modules (omega order, init): " + "; ".join(
        f"m{m}: ch {8 * m}-{8 * m + 7}, omega {om[8 * m].item():.3f}..{om[8 * m + 7].item():.3f} (period {2 * math.pi / om[8 * m].item():.1f}-{2 * math.pi / om[8 * m + 7].item():.1f} steps)"
        for m in range(4)))

print(f"\nOVERALL: {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
