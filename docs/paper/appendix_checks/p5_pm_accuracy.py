"""P5 addendum (descriptive): revisit accuracy of SO_s800 and of its exact mirror image Pm (move angles a -> pi - a in every
channel, plus the T1 reflection), which keeps every even-gap score and negates every odd-gap score (p3_p5_geometry.py (v))."""
import copy, math
import numpy as np
import torch
from common import REPO, ring_batch, reflect_qk
from mapformer.model_wrap_exist import MapFormerWM_SignedAO_r4

b = torch.load(f"{REPO}/runs/wrap_mech/p0/SO_s800/SO.pt", map_location="cpu", weights_only=False)
so = MapFormerWM_SignedAO_r4(vocab_size=19, d_model=128, n_heads=2, n_layers=1, grid_size=32, action_ids=(0, 1))
so.load_state_dict(b["model_state_dict"]); so = so.double().eval()
pm = copy.deepcopy(so)
with torch.no_grad():
    om = so.path_integrator.omega.reshape(-1); Z = so.token_emb.weight[so.act] @ so.action_to_lie.w_in.weight.T
    ang = so.step_table()[so.act].reshape(2, -1) * om
    pm.action_to_lie.w_out.weight.copy_((torch.linalg.pinv(Z) @ ((math.pi - ang) / om)).T)
reflect_qk(pm, [(h, c) for h in range(2) for c in range(32)])
tk, ps, _, _ = ring_batch(200, seed=99)
rev = np.zeros((200, 128), bool)
for i in range(200):
    seen = set()
    for t in range(128):
        rev[i, t] = ps[i, t] in seen; seen.add(ps[i, t])
with torch.no_grad():
    acc = lambda m: float((m(tk)[:, 0::2].argmax(-1) == tk[:, 1::2])[torch.from_numpy(rev)].double().mean())
    line = f"SO_s800 revisit accuracy {acc(so):.3f}; its mirror image Pm (sums 2 pi - S, non-negative angles) {acc(pm):.3f} (no retraining)"
print(line); open(__file__.replace(".py", "_out.txt"), "w").write(line + "\n")
