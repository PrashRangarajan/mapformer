"""POST HOC (CPU, after MECHANISM's registered analysis): how do signed models leave the |.| start? For each arm and seed,
over channels whose final |wrap(angle(+1))| > 0.05: the share where the two move angles have OPPOSITE raw signs at the end
(cancelling through zero, a(-1) ~ -a(+1)) and, among cancelling channels (|wrap(sum)| < 0.1), the share that cancel by
wrapping (both raw angles >= 0 or both <= 0, raw sum near +-2 pi k, k != 0)."""
import numpy as np
from mapformer.analyze_wrap_mech import load
from mapformer.model_wrap_exist import action_angles
R = "/home/prashr/mapformer/runs/wrap_mech/p0"
for arm in ("SO", "SO_same", "SO_absmag", "SO_high", "AO_high", "AO_mid"):
    opp, wrapc, canc_share = [], [], []
    for s in range(800, 808):
        A = action_angles(load(R, arm, s)["model"]).double().numpy()
        sel = np.abs(np.angle(np.exp(1j * A[0]))) > 0.05
        canc = np.abs(np.angle(np.exp(1j * (A[0] + A[1])))) < 0.1
        o = np.sign(A[0]) != np.sign(A[1])
        opp.append(o[sel].mean()); canc_share.append(canc[sel].mean())
        c = sel & canc
        wrapc.append((np.abs(A[0] + A[1])[c] > np.pi).mean() if c.any() else np.nan)
    print(f"{arm:10s}: opposite-sign channels at end {np.mean(opp):.2f}; cancelling {np.mean(canc_share):.2f}; "
          f"of cancelling, by wrapping (|raw sum| > pi) {np.nanmean(wrapc):.2f}")
