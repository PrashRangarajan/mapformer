"""P2 (rescaling sensitivity).
(a) On registered ring walks (32 cells, T 128, p_plus 0.5), for every same-cell (query move t, key = observation after move s)
    pair: raw phase phi = a+ n+ + a- n-. Rescale the channel by (1 + eps). Predicted error mod 2 pi: signed cancelling code
    (a- = -a+): eps a+ d; wrapped code (a- = 2 pi - a+): eps (a+ d + 2 pi n-). Checked pair by pair.
(b) The same rescaling on real models: AO_bal (wrapped) and SO_bal (signed, unwrapped) built from SO_s800 by
    model_wrap_exist.wrap_construct; omega multiplied by (1 + eps) in every channel; revisit accuracy (eval-only).
(c) Integrated decoupled weight decay over the registered continuations (300 epochs x 98 batches, cosine 5% warm-up to 10%
    of peak, wd 0.05): wd * sum(lr) for the ring (lr 1e-3) and torus (lr 1e-4) recipes, from train.py's schedule formula.
"""
import math
import numpy as np
import torch
from common import REPO, ring_batch
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4, wrap_construct

out = []
P = lambda s: (print(s), out.append(s))
toks, pos, mv, env = ring_batch(400, seed=2024)
rng = np.random.RandomState(0)
wr = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
maxerr = {"signed": 0.0, "wrapped": 0.0}; n_pairs = 0; nminus = []; d0 = 0
for eps in (1e-3, 1e-2):
    for b in range(len(pos)):
        ap = rng.uniform(0.1, 3.0)
        cp = np.concatenate([[0], np.cumsum(mv[b] == 1)]); cm = np.concatenate([[0], np.cumsum(mv[b] == -1)])
        for t in range(1, 128):
            ks = np.where(pos[b, :t] == pos[b, t])[0]
            for s in ks:
                npl, nmi = cp[t + 1] - cp[s + 1], cm[t + 1] - cm[s + 1]; d = npl - nmi
                for code, am in (("signed", -ap), ("wrapped", 2 * math.pi - ap)):
                    phi = ap * npl + am * nmi
                    err = wr((1 + eps) * phi - phi)
                    pred = eps * ap * d if code == "signed" else eps * (ap * d + 2 * math.pi * nmi)
                    maxerr[code] = max(maxerr[code], abs(wr(err - pred)))
                if eps == 1e-3:
                    n_pairs += 1; nminus.append(nmi); d0 += d == 0
P(f"(a) {n_pairs} same-cell pairs (400 walks), {d0 / n_pairs:.3f} at displacement 0; max |measured - predicted| phase error "
  f"under (1 + eps) rescaling, eps in {{1e-3, 1e-2}}: signed {maxerr['signed']:.1e}, wrapped {maxerr['wrapped']:.1e} rad")
nminus = np.array(nminus)
P(f"    n- over same-cell pairs: mean {nminus.mean():.1f}, rms {np.sqrt((nminus ** 2).mean()):.1f}, max {nminus.max()}; "
  f"predicted wrapped-code rms error at eps = 0.01: 2 pi eps rms(n-) = {2 * math.pi * 0.01 * np.sqrt((nminus ** 2).mean()):.2f} rad; "
  f"signed code at d = 0 pairs: 0")

# (b) real constructed models
b = torch.load(f"{REPO}/runs/wrap_mech/p0/SO_s800/SO.pt", map_location="cpu", weights_only=False)
so = MapFormerWM_SignedAO_r4(vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1))
so.load_state_dict(b["model_state_dict"]); so.eval()
kw = dict(d_model=128, n_heads=2, n_layers=1, grid_size=32)
ao_bal, rep_a = wrap_construct(so, 19, (0, 1), mode="balanced", **kw)
so_bal, rep_s = wrap_construct(so, 19, (0, 1), mode="signed_balanced", **kw)
ao_bal.eval(); so_bal.eval()
tk, ps, _, _ = ring_batch(200, seed=99)
rev = np.zeros((200, 128), bool)
for i in range(200):
    seen = set()
    for t in range(128):
        rev[i, t] = ps[i, t] in seen; seen.add(ps[i, t])
tgt = tk[:, 1::2]


def acc(m, eps):
    with torch.no_grad():
        om0 = m.path_integrator.omega.clone(); m.path_integrator.omega.mul_(1 + eps)
        lg = m(tk)[:, 0::2]; m.path_integrator.omega.copy_(om0)
    return float((lg.argmax(-1) == tgt)[torch.from_numpy(rev)].double().mean())


P(f"(b) constructions from SO_s800: AO_bal angle error {rep_a['angle_err_max']:.1e}, SO_bal {rep_s['angle_err_max']:.1e}")
for eps in (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
    P(f"    omega x (1 + {eps:g}) in every channel: revisit accuracy wrapped (AO_bal) {acc(ao_bal, eps):.3f}, signed (SO_bal) {acc(so_bal, eps):.3f}")

# (c) integrated decay
for name, lr in (("ring", 1e-3), ("torus", 1e-4)):
    total = 300 * 98; warm = max(1, int(0.05 * total)); s = 0.0
    for step in range(total):
        f = (step + 1) / warm if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((step - warm) / max(1, total - warm), 1)))
        s += lr * f
    P(f"(c) {name} continuation: sum(lr) = {s:.2f}; wd * sum(lr) = {0.05 * s:.3f}; unopposed decay factor per tensor "
      f"exp(-wd sum lr) = {math.exp(-0.05 * s):.3f}; an angle is a product of 4 decayed tensors (token_emb, W_in, W_out, omega): "
      f"{math.exp(-4 * 0.05 * s):.3f}")
open(__file__.replace(".py", "_out.txt"), "w").write("\n".join(out) + "\n")
