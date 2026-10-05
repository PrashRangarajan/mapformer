"""Which bytes move the phase? Step table of the path-integrated CODE models (runs/code2048: byte-level Python,
trained and tested at 2048, 9 layers, 8 heads, rank 4; MapWM = Vanilla, MapPoPE-Flat), read from the weights:
step(b) = omega * W_out W_in emb(b) -- exactly the model's per-token phase increment. Post hoc, CPU. Readouts:
  top bytes by omega-scaled step norm; frequency-weighted share of total step norm carried by the top 10 bytes;
  clock share: || freq-weighted mean step || / freq-weighted mean || step || (1 = every byte steps the same way);
  bracket opposition after removing the common (clock) component: || s(open) + s(close) || / mean norm
  (0 = cancel, 2 = identical), against the same statistic for random pairs of bytes with frequency > 1e-4."""
import numpy as np, torch
R = "/home/prashr/mapformer/runs/code2048"
data = np.fromfile("/home/prashr/mapformer/data/code_train.bin", dtype=np.uint8)
freq = np.bincount(data[: 20_000_000], minlength=256).astype(float); freq /= freq.sum()
name = lambda b: {10: "\\n", 32: "' '", 9: "\\t"}.get(b, chr(b) if 33 <= b < 127 else f"<{b}>")
rng = np.random.default_rng(0)
for arm in ("Vanilla", "MapPoPE-Flat"):
    for s in (0, 1, 2):
        sd = torch.load(f"{R}/{arm}_s{s}.best.pt", map_location="cpu", weights_only=False)["state_dict"]
        E, Wi, Wo, om = sd["token_emb.weight"], sd["action_to_lie.w_in.weight"], sd["action_to_lie.w_out.weight"], sd["path_integrator.omega"]
        st = ((E @ Wi.T @ Wo.T).view(256, *om.shape) * om).reshape(256, -1).numpy()       # (256, H*nb), radians per byte
        n = np.linalg.norm(st, axis=1); w = freq
        mean_step = (w[:, None] * st).sum(0); clock = np.linalg.norm(mean_step) / (w * n).sum()
        top = np.argsort(-(n * (w > 1e-5)))[:12]
        share10 = (w * n)[np.argsort(-(w * n))[:10]].sum() / (w * n).sum()
        d = st - mean_step                                                              # remove the common component
        nd = np.linalg.norm(d, axis=1)
        opp = lambda a, b: float(np.linalg.norm(d[a] + d[b]) / ((nd[a] + nd[b]) / 2))
        brackets = {p: opp(ord(p[0]), ord(p[1])) for p in ("()", "[]", "{}")}
        common = np.nonzero(w > 1e-4)[0]
        base = [opp(*rng.choice(common, 2, replace=False)) for _ in range(2000)]
        print(f"{arm:12s} s{s}: clock share {clock:.3f}; top-10 bytes carry {share10:.0%} of freq-weighted step; "
              f"largest steps: {' '.join(name(int(b)) for b in top)}")
        print(f"{'':16s} bracket opposition (common removed) " + "  ".join(f"{k} {v:.2f}" for k, v in brackets.items())
              + f"   | random byte pairs: median {np.median(base):.2f}, 5th pct {np.percentile(base, 5):.2f}")
