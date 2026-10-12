"""POST HOC (CPU): does the non-negative (Abs / Pos) model on the 2D torus find the wrap solution that exists for it?
Checkpoints of the matched-length sign batch (`runs/sign_matched`, T = 1024, 900 epochs). Per opposite action pair (N/S,
E/W) and channel with |step| > 0.05 rad: |wrap(step(a) + step(opposite a))| (0 = the pair cancels mod 2 pi). Reported: the
median over channels and the share below 0.1 rad, per arm (mean over seeds). Same readout as RING_DIAL's secondary."""
import glob, numpy as np, torch
from mapformer.train_variant import VARIANT_MAP
for arm in ("Signed_r4", "Abs_r4", "Pos_r4"):
    med, share = [], []
    for ck in sorted(glob.glob(f"/home/prashr/mapformer/runs/sign_matched/p0/{arm}_s*/{arm}.pt")):
        b = torch.load(ck, map_location="cpu", weights_only=False); c = b["config"]
        m = VARIANT_MAP[arm](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                             grid_size=c["grid_size"]); m.load_state_dict(b["model_state_dict"])
        with torch.no_grad():
            A = (m.action_to_lie(m.token_emb.weight[None])[0] * m.path_integrator.omega).reshape(c["vocab_size"], -1).numpy()
        ws = []
        for a, o in ((0, 1), (2, 3)):                                   # torus actions: +x/-x, +y/-y (environment order)
            big = np.abs(A[a]) > 0.05
            ws.append(np.abs(np.angle(np.exp(1j * (A[a] + A[o]))))[big])
        w = np.concatenate(ws); med.append(np.median(w)); share.append((w < 0.1).mean())
    print(f"{arm:10s} n={len(med)} |wrap(a + opposite)| median {np.mean(med):.3f} rad; share of channels cancelling < 0.1 rad {np.mean(share):.2f}")
