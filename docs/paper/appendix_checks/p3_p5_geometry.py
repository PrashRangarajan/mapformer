"""P3 (cut circle), P4 (sticky wall), P5 (pi-mirror), checked.
P3: brute force on a grid: from starts (a+, a-) in (0, pi)^2 the infimum distance to a NON-blind cancelling point is S / sqrt 2
    for signed angles and (2 pi - S) / sqrt 2 in the closed quadrant (S = a+ + a-); any quadrant path to it crosses S = pi.
P4: AO_s800 (|.|): autograd dL/ds = sign(s) dL/da for the pre-|.| move table s, a = |s| (eval-mode loss on revisits).
P5: (i) every same-cell pair on ring walks has an even move count g; (ii) the point reflection Pm: a -> pi - a (both moves,
    every channel) negates every even-g phase mod 2 pi, so V_s (the same-cell cosine term) is invariant to float precision;
    (iii) the S -> 2 pi - S mirror at fixed delta (a+, a-) -> (pi - a-, pi - a+) is exact on a reflection-closed sample
    (walk and its mirror image), approximate on a plain p_plus = 0.5 sample, broken at p_plus = 0.75 unless delta N in pi Z;
    (iv) the all-keys term V_a is not mirror-symmetric; (v) model level, SO_s800 float64: Pm applied to the move angles plus
    the T1 reflection in every channel leaves every even-g attention score unchanged and negates every odd-g score.
"""
import copy
import math
import numpy as np
import torch
from common import REPO, ring_batch, reflect_qk
from mapformer.model import _apply_rope
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, MapFormerWM_AbsAO_r4

out = []
P = lambda s: (print(s), out.append(s))
TWO_PI = 2 * math.pi

# P3
g = np.linspace(-4 * math.pi, 4 * math.pi, 1001)
A, B = np.meshgrid(g, g, indexing="ij")
canc = (np.abs(np.angle(np.exp(1j * (A + B)))) < 1e-9 + 0.5 * (g[1] - g[0])) & ~((np.abs(A) < 1e-9) & (np.abs(B) < 1e-9))
quad = (A >= 0) & (B >= 0)
rng = np.random.RandomState(0); worst = [0.0, 0.0]
for _ in range(60):
    a = rng.uniform(0.05, math.pi - 0.05, 2); S = a.sum()
    if S >= TWO_PI:
        continue
    dist = np.hypot(A - a[0], B - a[1])
    ds, dq = dist[canc].min(), dist[canc & quad].min()
    worst[0] = max(worst[0], abs(ds - min(S, TWO_PI - S) / math.sqrt(2))); worst[1] = max(worst[1], abs(dq - (TWO_PI - S) / math.sqrt(2)))
P(f"P3: 60 starts in (0, pi)^2: max |grid distance - min(S, 2pi - S)/sqrt2| (signed) {worst[0]:.4f}, |grid distance - (2pi - S)/sqrt2| "
  f"(quadrant) {worst[1]:.4f} (grid step {g[1] - g[0]:.4f}); the two coincide iff S >= pi")

# P4
toks, pos, mv, env = ring_batch(64, seed=5)
rev = np.zeros((64, 128), bool)
for i in range(64):
    seen = set()
    for t in range(128):
        rev[i, t] = pos[i, t] in seen; seen.add(pos[i, t])
b = torch.load(f"{REPO}/runs/wrap_mech/p0/AO_s800/AO.pt", map_location="cpu", weights_only=False)
ao = MapFormerWM_AbsAO_r4(vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1))
ao.load_state_dict(b["model_state_dict"]); ao = ao.double().eval()


def fwd_with_table(m, table):   # table: (2, H, nb) move steps
    full = torch.zeros(19, *table.shape[1:], dtype=table.dtype); full = full.index_put((m.act,), table)
    cos_a, sin_a = m.path_integrator(full[toks]); x = m.token_emb(toks); L = toks.shape[1]
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    for layer in m.layers:
        x = layer(x, cos_a, sin_a, mask)
    lg = m.out_proj(m.out_norm(x))[:, 0::2]
    return torch.nn.functional.cross_entropy(lg[torch.from_numpy(rev)], toks[:, 1::2][torch.from_numpy(rev)])


with torch.no_grad():
    s0 = ao.action_to_lie.w_out(ao.action_to_lie.w_in(ao.token_emb.weight[ao.act])).view(2, 2, 32)
s = s0.clone().requires_grad_(True); fwd_with_table(ao, s.abs()).backward()
a = s0.abs().clone().requires_grad_(True); fwd_with_table(ao, a).backward()
P(f"P4: AO_s800: max |dL/ds - sign(s) dL/da| = {(s.grad - torch.sign(s0) * a.grad).abs().max():.2e} (scale {a.grad.abs().max():.2e}); "
  f"entries with s < 0: {int((s0 < 0).sum())}/128 (on those the gradient on s is the negated gradient on a)")

# P5 (i)-(iv)
def pairs(seed, p_plus, n=600, mirror=False):
    """Pair histograms W_s[m, k], W_a[m, k] (per revisit query: 1/#same over same-cell keys, 1/#all over all keys, with m, k the
    +1 / -1 move counts in (key, query]); V(a) = sum W cos(a+ m + a- k) / #queries."""
    tk, ps, m_, _ = ring_batch(n, seed=seed, p_plus=p_plus)
    if mirror:
        ps = np.concatenate([ps, (-ps) % 32]); m_ = np.concatenate([m_, -m_])
    Ws = np.zeros((129, 129)); Wa = np.zeros((129, 129)); nq = 0; gs = []
    for bb in range(len(ps)):
        cp = np.concatenate([[0], np.cumsum(m_[bb] == 1)]); cm = np.concatenate([[0], np.cumsum(m_[bb] == -1)])
        for t in range(1, 128):
            ks = np.arange(t); same = ps[bb, ks] == ps[bb, t]
            if same.any():
                dp = cp[t + 1] - cp[ks + 1]; dm = cm[t + 1] - cm[ks + 1]; nq += 1
                np.add.at(Ws, (dp[same], dm[same]), 1.0 / same.sum()); np.add.at(Wa, (dp, dm), 1.0 / t)
                gs.append((dp + dm)[same])
    return (Ws / nq, Wa / nq, np.concatenate(gs), nq)


def V(Q, ap, am):
    Ws, Wa = Q[0], Q[1]; mm = np.arange(129)
    ep, em = np.exp(1j * ap * mm), np.exp(1j * am * mm)
    return float((ep @ Ws @ em).real), float((ep @ Wa @ em).real)


for label, seed, pp, mir in (("p_plus 0.5", 11, 0.5, False), ("p_plus 0.5 + mirrored walks", 11, 0.5, True), ("p_plus 0.75", 12, 0.75, False)):
    Q = pairs(seed, pp, mirror=mir)
    allg = Q[2]
    e_pm = e_pp = e_va = 0.0
    for _ in range(20):
        ap, am = rng.uniform(-1, 4, 2)
        v0 = V(Q, ap, am); v1 = V(Q, math.pi - ap, math.pi - am); v2 = V(Q, math.pi - am, math.pi - ap)
        e_pm = max(e_pm, abs(v1[0] - v0[0])); e_pp = max(e_pp, abs(v2[0] - v0[0])); e_va = max(e_va, abs(v1[1] - v0[1]), abs(v2[1] - v0[1]))
    lat = []
    for _ in range(5):   # P' on the wrap lattice: delta N in pi Z
        S = rng.uniform(0, TWO_PI); dl = math.pi * rng.randint(1, 5) / 32
        v0 = V(Q, S / 2 + dl, S / 2 - dl); v2 = V(Q, math.pi - (S / 2 - dl), math.pi - (S / 2 + dl)); lat.append(abs(v2[0] - v0[0]))
    P(f"P5 {label}: {Q[3]} revisit queries; same-cell move counts all even: {bool((allg % 2 == 0).all())}; max |V_s(Pm a) - V_s(a)| "
      f"{e_pm:.1e}; max |V_s(mirror_S a) - V_s(a)| {e_pp:.1e} (on the lattice delta N in pi Z: {max(lat):.1e}); all-keys term max change {e_va:.2f}")

# P5 (v) model level
b = torch.load(f"{REPO}/runs/wrap_mech/p0/SO_s800/SO.pt", map_location="cpu", weights_only=False)
so = MapFormerWM_SignedAO_r4(vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1))
so.load_state_dict(b["model_state_dict"]); so = so.double().eval()
pm = copy.deepcopy(so)
with torch.no_grad():
    om = so.path_integrator.omega.reshape(-1); Z = so.token_emb.weight[so.act] @ so.action_to_lie.w_in.weight.T
    ang = so.step_table()[so.act].reshape(2, -1) * om
    pm.action_to_lie.w_out.weight.copy_((torch.linalg.pinv(Z) @ ((math.pi - ang) / om)).T)
reflect_qk(pm, [(h, c) for h in range(2) for c in range(32)])
a2 = (pm.step_table()[pm.act].reshape(2, -1) * om)
opp = ang[0] * ang[1] < 0
P(f"P5 (v) SO_s800: Pm built exactly (max |a' - (pi - a)| {(a2 - (math.pi - ang)).abs().max():.1e}); source |a| < pi in "
  f"{int((ang.abs() < math.pi).sum())}/128 entries, image angles >= 0 in {int((a2 >= 0).sum())}/128; on the {int(opp.sum())} opposite-sign "
  f"channels source sums S in [{ang.sum(0)[opp].min():.3f}, {ang.sum(0)[opp].max():.3f}], image sums 2 pi - S in "
  f"[{a2.sum(0)[opp].min():.3f}, {a2.sum(0)[opp].max():.3f}]")


def scores(m):
    with torch.no_grad():
        layer = m.layers[0]; h = layer.norm1(m.token_emb(toks)); Bn, L, _ = h.shape
        cos_a, sin_a = m.get_position_state(toks)
        Qm = _apply_rope(layer.q_proj(h).view(Bn, L, 2, 64).transpose(1, 2), cos_a, sin_a)
        Km = _apply_rope(layer.k_proj(h).view(Bn, L, 2, 64).transpose(1, 2), cos_a, sin_a)
        return Qm @ Km.transpose(-1, -2)


pm_noref = copy.deepcopy(pm); reflect_qk(pm_noref, [(h, c) for h in range(2) for c in range(32)])   # undo the reflection
S0, S1, S1n = scores(so), scores(pm), scores(pm_noref)
ismove = (toks == 0) | (toks == 1); cmv = torch.cumsum(ismove.long(), 1)                 # moves up to and including position
G = cmv[:, :, None] - cmv[:, None, :]                                                     # moves in (key, query]
causal = torch.tril(torch.ones(256, 256, dtype=torch.bool))[None]
ev = (G % 2 == 0) & causal; od = (G % 2 == 1) & causal
ev, od = ev[:, None].expand_as(S0), od[:, None].expand_as(S0)
P(f"    all causal scores: even-g max |S' - S| {(S1 - S0)[ev].abs().max():.1e}; odd-g max |S' + S| {(S1 + S0)[od].abs().max():.1e} "
  f"(score scale {S0[ev | od].abs().max():.1f}); control without the reflection: even-g max |S' - S| {(S1n - S0)[ev].abs().max():.1f}")
open(__file__.replace(".py", "_out.txt"), "w").write("\n".join(out) + "\n")
