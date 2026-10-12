"""POST HOC companion (CLOCK_SPEED Amendment 1, D3; CPU): the loss-AUC readout (mean training loss over the 300 epochs)
on RING_DIAL's runs, paired by seed (Signed_r4 and Abs_r4 share initial weights and data at a seed), so the ring (map) side
and the recency (clock) side can be read with the same measure. Descriptive."""
import numpy as np, torch
from mapformer.stats_core import signflip_p
R = "/home/prashr/mapformer/runs/ring_dial/p0"
for p in (0.5, 0.75, 0.9):
    auc = {a: [float(np.mean(torch.load(f"{R}/{a}_p{p}_s{s}/{a}.pt", map_location="cpu", weights_only=False)["losses"]))
               for s in range(30, 38)] for a in ("Signed_r4", "Abs_r4")}
    dd = [x - y for x, y in zip(auc["Abs_r4"], auc["Signed_r4"])]
    print(f"ring p_plus {p}: AUC Signed {np.mean(auc['Signed_r4']):.3f}, Abs {np.mean(auc['Abs_r4']):.3f}; paired Abs - Signed "
          f"{np.mean(dd):+.3f} (sign-flip p {signflip_p(dd)['p']:.4f}; Abs slower on {sum(x > 0 for x in dd)}/8 seeds)")
