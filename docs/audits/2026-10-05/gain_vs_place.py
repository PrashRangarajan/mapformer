"""Does a louder object at the wrong place ever outscore a quieter one at the right place? (post hoc, CPU)
For trained 1-layer paper-torus models, the exact pre-softmax score S(action a, object o, displacement d) from
remap_probe.kernel_matrix (verified against the model's logits there). Per head and query action: the WEAKEST object's
score at d = 0 vs the STRONGEST object's score at any d != 0 within |d| <= 16, and the margin in units of the d = 0 spread."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-10-05"); sys.path.insert(0, "/home/prashr/mapformer/docs/audits/2026-09-27")
import remap_probe as RP, probe_whatwhere as PW
NA, NO = 4, 17
for arm in ("MapPoPE_r4", "MapPoPE-Flat", "Vanilla_r4"):
    for s in range(4):
        m, _ = PW.load(f"/home/prashr/mapformer/runs/paper2x2/p0/{arm}_s{s}/{arm}.pt")
        X, disp = RP.kernel_matrix(m); i0 = disp.index((0, 0)); off = [i for i in range(len(disp)) if i != i0]
        out = []
        for h in range(X.shape[0]):
            S = X[h].reshape(NA, NO, -1)[:, :16, :]                    # 16 objects (blank excluded)
            for a in range(NA):
                weak0 = S[a, :, i0].min(); strong0 = S[a, :, i0].max(); strong_off = S[a][:, off].max()
                out.append((weak0, strong0, strong_off))
        o = np.array(out)
        wins = (o[:, 0] > o[:, 2]).mean()
        print(f"{arm:13s} s{s}: weakest object at d=0 beats strongest object at any d!=0 in {wins:.0%} of (head, action); "
              f"median scores: weakest@0 {np.median(o[:, 0]):6.2f}, strongest@0 {np.median(o[:, 1]):6.2f}, strongest elsewhere {np.median(o[:, 2]):6.2f}")

print("\n-- by distance: weakest object at d=0 vs strongest object at L1 distance k (median over heads x actions) --")
for arm in ("MapPoPE_r4", "Vanilla_r4"):
    for s in range(2):
        m, _ = PW.load(f"/home/prashr/mapformer/runs/paper2x2/p0/{arm}_s{s}/{arm}.pt")
        X, disp = RP.kernel_matrix(m); D = np.abs(np.array(disp)).sum(1); i0 = disp.index((0, 0)); row = []
        for k in (1, 2, 3, 4, 6, 8):
            idx = np.nonzero(D == k)[0]; w = []
            for h in range(X.shape[0]):
                S = X[h].reshape(NA, NO, -1)[:, :16, :]
                for a in range(NA):
                    w.append(S[a, :, i0].min() > S[a][:, idx].max())
            row.append(f"k={k}: {np.mean(w):.0%}")
        argd = []
        for h in range(X.shape[0]):
            S = X[h].reshape(NA, NO, -1)[:, :16, :]
            for a in range(NA):
                j = np.unravel_index(np.argmax(np.where(D[None] > 0, S[a], -np.inf)), S[a].shape)[1]; argd.append(D[j])
        print(f"{arm:11s} s{s}: weakest@0 wins against strongest at distance " + "  ".join(row) + f"   | where the strongest wrong-place key sits: L1 {sorted(argd)}")
