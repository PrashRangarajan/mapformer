"""Content-pair kernel argmax restricted to ODD token offsets (POST HOC, CPU, seconds); docs/paper/review_draft5/
ANALYSIS_SCRIPT_AUDIT.md mappope MINOR 3. kernel_probe.py's argmax runs over D = 1..16, including even offsets, where no
observation key sits relative to an action query (observations are at odd offsets). Same kernel K_ao(D) as
kernel_probe.content_kernel_checks (index arms of PAPER2X2, seeds 0-7: RoPE and PoPE-Flat), argmax over D in {1, 3, ..., 15};
pooled (head, action, observation) counts, and how many land on a retrace offset 4j - 1.
  cd /home/prashr && PYTHONPATH=/home/prashr python3 .../mappope/kernel_argmax_odd.py > .../mappope/kernel_argmax_odd_out.txt"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_probe as kp  # noqa: E402

F = kp.F
torch.set_num_threads(4)


@torch.no_grad()
def kernel(m):
    lay = m.layers[0]; H, dh = m.n_heads, lay.d_head; E = m.token_emb.weight
    q = lay.q_proj(lay.norm1(E[:kp.NA])).view(kp.NA, H, dh); k = lay.k_proj(lay.norm1(E[kp.NA:kp.NA + kp.NO])).view(kp.NO, H, dh)
    D = torch.arange(0, 17, dtype=torch.float32)
    if hasattr(m, "theta_c"):
        th = m.theta_c; mq, mk = F.softplus(q), F.softplus(k)
        d = lay.pope_delta.clamp(kp.DELTA_MIN, kp.DELTA_MAX)
        Kd = torch.einsum("ahc,ohc,hcd->haod", mq, mk, torch.cos(th[None, :, None] * D[None, None] - d[..., None]))
    else:
        th = m.inv_freq
        qc = torch.complex(q[..., 0::2], q[..., 1::2]); kc = torch.complex(k[..., 0::2], k[..., 1::2])
        Z = qc[:, None] * kc[None].conj()
        Kd = (Z[..., None] * torch.exp(1j * th[None, None, None, :, None] * D)).real.sum(-2).permute(2, 0, 1, 3)
    return Kd.reshape(H, kp.NA, kp.NO, -1)


cfg = kp.BATCHES["paper2x2"]
for v in ("RoPE", "PoPE-Flat"):
    hall = np.zeros(17, int); hodd = np.zeros(17, int)
    for s in cfg["seeds"]:
        m, _ = kp.load(cfg["dir"], v, s)
        Kd = kernel(m)
        hall += np.bincount((Kd[..., 1:].argmax(-1) + 1).flatten().numpy(), minlength=17)
        odd = torch.arange(1, 17, 2)
        hodd += np.bincount(odd[Kd[..., odd].argmax(-1)].flatten().numpy(), minlength=17)
    rt = sum(hodd[d] for d in (3, 7, 11, 15))
    print(f"{v}: argmax over D 1..16: " + " ".join(f"{d}:{c}" for d, c in enumerate(hall) if c))
    print(f"{v}: argmax over odd D: " + " ".join(f"{d}:{c}" for d, c in enumerate(hodd) if c)
          + f"  | on a retrace offset 4j-1 (3, 7, 11, 15): {rt} of {hodd.sum()}")
