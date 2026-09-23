"""Registered readouts for RANK_MATCHED_PREREG.md, from the committed JSONs and checkpoints."""
import json
import numpy as np
import torch

REPO = "/home/prashr/mapformer"
V2, V4, SEEDS = "Vanilla", "Vanilla_r4", list(range(8))
M = json.load(open(f"{REPO}/RANK_MATCHED.json"))
S = json.load(open(f"{REPO}/RANK_MATCHED_STRATA.json"))
O = json.load(open(f"{REPO}/RANK_SWEEP_STRATA.json"))


def pair(a, b, label):
    a, b = np.array(a, float), np.array(b, float); d = b - a
    sd = d.std(ddof=1); mde = 2.8 * sd / np.sqrt(len(d))
    tag = "DETECTABLE" if abs(d.mean()) > mde else "unmeasured"
    print(f"  {label:34s} r2 {a.mean():.3f}  r4 {b.mean():.3f}  d {d.mean():+.3f}  MDE {mde:.3f}  "
          f"{(d > 0).sum()}/{len(d)} positive  {tag}")
    return d


def get(J, v, T, k="acc"):
    return [dict((x[0], x) for x in J[f"0.0|{v}|{T}"])[s][1 if k == "acc" else 2] for s in SEEDS]


print("== overall (eval_noise_refine), trained at T=1024 ==")
for T in (512, 1024, 2048):
    pair(get(M, V2, T), get(M, V4, T), f"acc T={T}" + ("  [PRIMARY]" if T == 1024 else ""))
    pair(get(M, V2, T, "nll"), get(M, V4, T, "nll"), f"NLL T={T} (negative = r4 better)")

for name, J in (("NEW trained T=1024", S), ("OLD trained T=128", O)):
    print(f"\n== strata, {name} ==")
    for T in (1024, 2048):
        for k in ("all", "plain_lag<128", "plain_lag>=128", "wrap"):
            a = [J[f"{V2}|{s}|{T}"][k]["acc"] for s in SEEDS]
            b = [J[f"{V4}|{s}|{T}"][k]["acc"] for s in SEEDS]
            fl = np.mean([J[f"{V2}|{s}|{T}"][k]["floor"] for s in SEEDS])
            pair(a, b, f"T={T} {k} (floor {fl:.3f})")

print("\n== per seed, T=1024 overall acc and training loss ==")
L = {}
for v in (V2, V4):
    for s in SEEDS:
        b = torch.load(f"{REPO}/runs/rank_matched/p0/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
        l = np.array(b["losses"], float); assert b["config"]["n_steps"] == 1024
        flat = abs(l[270:].mean() - l[240:270].mean()) <= 0.05 * l[240:270].mean()
        L[(v, s)] = (l[-1], flat, l[270:].mean() / l[240:270].mean())
    acc = get(M, v, 1024)
    print(f"  {v:11s} acc  " + " ".join(f"{x:.3f}" for x in acc))
    print(f"  {v:11s} loss " + " ".join(f"{L[(v, s)][0]:.4f}" for s in SEEDS)
          + f"   flat {sum(L[(v, s)][1] for s in SEEDS)}/8"
          + "   last30/prev30 " + " ".join(f"{L[(v, s)][2]:.2f}" for s in SEEDS))
l2 = [L[(V2, s)][0] for s in SEEDS]; l4 = [L[(V4, s)][0] for s in SEEDS]
print(f"  loss ranges: r2 {min(l2):.4f}-{max(l2):.4f}  r4 {min(l4):.4f}-{max(l4):.4f}  "
      f"overlap: {max(min(l2), min(l4)) <= min(max(l2), max(l4))}")
x = np.log([*l2, *l4]); y = np.array([*get(M, V2, 1024), *get(M, V4, 1024)])
print(f"  r(log final loss, T=1024 acc): pooled {np.corrcoef(x, y)[0, 1]:+.3f}  "
      f"r2 {np.corrcoef(x[:8], y[:8])[0, 1]:+.3f}  r4 {np.corrcoef(x[8:], y[8:])[0, 1]:+.3f}")
# loss-matched residual: regress acc on log loss pooled, compare arm means of residuals
beta = np.polyfit(x, y, 1); res = y - np.polyval(beta, x)
d = res[8:] - res[:8]
print(f"  loss-matched r4-r2: {d.mean():+.3f}  MDE {2.8 * d.std(ddof=1) / np.sqrt(8):.3f}  (slope {beta[0]:+.3f}/log-loss)")
