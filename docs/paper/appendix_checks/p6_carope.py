"""P6 (CARoPE cannot cancel).
(a) Trained ring CARoPEF checkpoints (runs/generality/ring/CARoPEF_s700..707, eval-only): per channel, the phase difference
    between the move-token query 2t and the observation key 2s+1 of the previous visit, over all same-cell pairs: the
    out-and-back (g = 2) advance lies in (0, 3]; no channel reads every return at phase 0 mod 2 pi (share of channels with
    |wrap(phase)| < 0.1 at every return, and at every g = 2 return).
(b) The clock-or-blind bound: random rates c+, c-, c_obs in (0, 1]; on the symmetric excursion (J forward then J back) the
    nested return phases P_1..P_J; whenever all |wrap(P_j)| <= eps < pi - 2, check J (c+ + c-) <= eps.
"""
import math
import numpy as np
import torch
from common import REPO, ring_batch
from mapformer.model_mono_baselines import MapFormerWM_CARoPEF

out = []
P = lambda s: (print(s), out.append(s))
wr = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
toks, pos, mv, env = ring_batch(100, seed=31)
for s in range(700, 708):
    b = torch.load(f"{REPO}/runs/generality/ring/CARoPEF_s{s}/CARoPEF.pt", map_location="cpu", weights_only=False)
    m = MapFormerWM_CARoPEF(vocab_size=19, d_model=128, n_heads=2, n_layers=1); m.load_state_dict(b["model_state_dict"]); m.eval()
    with torch.no_grad():
        cos_a, sin_a = m.phase(toks); ang = torch.atan2(sin_a, cos_a)                       # wrapped; need raw: recompute
        f = 1.0 / (torch.nn.functional.softplus(m.rate(m.token_emb(toks))) + 1.0)
        raw = (f[..., None] ** m.expo).cumsum(1).double()                                     # (B, L, H, nb) raw phase
    allret = np.ones(64, bool); g2ret = np.ones(64, bool); g2max = 0.0; g2min = 1e9; npair = 0
    for bb in range(len(pos)):
        for t in range(1, 128):
            ks = np.where(pos[bb, :t] == pos[bb, t])[0]
            if len(ks) == 0:
                continue
            ph = (raw[bb, 2 * t] - raw[bb, 2 * ks + 1]).reshape(len(ks), -1).numpy()          # (keys, 64)
            ok = np.abs(wr(ph)) < 0.1
            allret &= ok.all(0); npair += len(ks)
            g2 = (t - ks) == 2
            if g2.any():
                g2ret &= ok[g2].all(0); g2max = max(g2max, ph[g2].max()); g2min = min(g2min, ph[g2].min())
    # are the channels that read every return near 0 blind? max |wrap(phase)| over ALL query-move / observation-key pairs
    q = raw[:, 0::2].reshape(len(pos), 128, -1); k = raw[:, 1::2].reshape(len(pos), 128, -1)
    blindmax = 0.0; stepmax = float((f[..., None] ** m.expo).reshape(-1, 64)[:, allret].max()) if allret.any() else float("nan")
    for bb in range(0, len(pos), 10):
        d = (q[bb][:, None, :] - k[bb][None, :, :])[np.tril_indices(128, -1)]             # key s < query t
        if allret.any():
            blindmax = max(blindmax, float(np.abs(wr(d[:, allret].numpy())).max()))
    P(f"(a) CARoPEF s{s}: {npair} same-cell pairs; out-and-back (g = 2) advance range [{g2min:.3f}, {g2max:.3f}] rad (bound (0, 3]); "
      f"channels reading every g = 2 return within 0.1 rad of 0: {int(g2ret.sum())}/64; every return: {int(allret.sum())}/64, "
      f"and those are blind: max per-token step {stepmax:.1e} rad, max |phase| over ALL query/key pairs (any cells) {blindmax:.3f} rad")

rng = np.random.RandomState(0); viol = 0; checked = 0
for _ in range(200000):
    cp, cm = rng.uniform(0, 1, 2) ** rng.uniform(0, 8); J = rng.randint(1, 17); eps = rng.uniform(0.01, math.pi - 2 - 1e-3)
    co = rng.uniform(0, 1, 2 * J) ** rng.uniform(0, 8)                                        # obs steps at the visited cells
    # nested returns: P_1 = c+ + c- + c(o at turn); P_{j+1} = P_j + c+ + c- + 2 c(o at cell J-j)
    Pj = [cp + cm + co[0]]
    for j in range(1, J):
        Pj.append(Pj[-1] + cp + cm + 2 * co[j])
    if all(abs(wr(p)) <= eps for p in Pj):
        checked += 1; viol += J * (cp + cm) > eps + 1e-12
P(f"(b) {checked} random channels/excursions with every nested return within eps < pi - 2 of 0: J (c+ + c-) > eps in {viol} "
  f"(the bound J (c+ + c-) <= eps held in all)")
open(__file__.replace(".py", "_out.txt"), "w").write("\n".join(out) + "\n")
