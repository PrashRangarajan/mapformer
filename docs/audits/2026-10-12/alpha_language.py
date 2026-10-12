"""Growth exponent alpha of the path-integrated accumulator on language data (POST HOC, CPU; run from /home/prashr with
PYTHONPATH=/home/prashr). The "narrow surviving use" of alpha named in .claude-memory/project_clock_vs_map.md: it needs only
trajectories, so it can be read on text where opposite actions are not labelled.
Definition as in LOCALISATION.md / analyze_t1.py: S = cumsum over tokens of W_out W_in emb(token) (before omega), per channel;
range(T) = max - min of S over the first T tokens, averaged over channels and windows; alpha = slope of log range(T) on log T
for T in 64 .. 2048. MapFormer's step is context-free, so S is computed from the step table on held-out text directly
(windows longer than a model's training context are allowed: alpha is a property of the step table on the data).
Reference points (committed): torus signed r4 0.52, r2 0.62, monotone 0.94 (LOCALISATION.md); unconstrained arm torus 0.59,
recency 0.97 (RECENCY_RESULTS.md); Bach MapWM ~1.00 (project_clock_vs_map). A random walk gives 0.5, a clock 1.0.
Also printed: the share of the variance of S explained by the token index (R^2 of S on t), a direct clock readout; and
alpha of the CONTENT part of the step (the step table minus its frequency-weighted mean step, i.e. with the clock removed),
on the real token order and on the same tokens shuffled within each window (iid order: a random walk, alpha ~0.5). Real-order
alpha below the shuffled one means the content part cancels over text (bounded, map-like, as nesting depth would); above it
means it persists. (A first version printed a "sign-shuffled" reference that kept each token's sign fixed and so still
drifted, alpha 0.90-0.99; it was not a random-walk reference and is removed.)"""
import json, numpy as np, torch

REPO = "/home/prashr/mapformer"
TS = [64, 128, 256, 512, 1024, 2048]
rng = np.random.default_rng(0)


def stats(D, toks, n_win=32):
    """D: (V, C) step table; toks: 1D int array. Returns alpha, mean range at 512 / 2048, R^2 of S on t."""
    starts = rng.choice(len(toks) - TS[-1] - 1, n_win, replace=False)
    W = np.stack([toks[s:s + TS[-1]] for s in starts]).astype(np.int64)            # (n_win, 2048)
    S = np.cumsum(D[W], axis=1)                                                     # (n_win, 2048, C)
    r = [float((S[:, :T].max(1) - S[:, :T].min(1)).mean()) for T in TS]
    alpha = float(np.polyfit(np.log(TS), np.log(r), 1)[0])
    t = np.arange(1, TS[-1] + 1, dtype=float); t = (t - t.mean())[None, :, None]
    Sc = S - S.mean(1, keepdims=True)
    b = (Sc * t).sum(1, keepdims=True) / (t ** 2).sum()
    r2 = float(1 - ((Sc - b * t) ** 2).sum() / (Sc ** 2).sum())
    return alpha, r[3], r[5], r2


def content_alpha(D, toks, shuffle, n_win=32):
    f = np.bincount(np.asarray(toks[:5_000_000], dtype=np.int64), minlength=D.shape[0]).astype(float); f /= f.sum()
    Dc = D - (f[:, None] * D).sum(0)                                                 # clock removed
    starts = rng.choice(len(toks) - TS[-1] - 1, n_win, replace=False)
    W = np.stack([toks[s:s + TS[-1]] for s in starts]).astype(np.int64)
    if shuffle:
        W = np.stack([rng.permutation(w) for w in W])
    S = np.cumsum(Dc[W], axis=1)
    r = [float((S[:, :T].max(1) - S[:, :T].min(1)).mean()) for T in TS]
    return float(np.polyfit(np.log(TS), np.log(r), 1)[0])


def report(tag, D, toks):
    a, r5, r20, r2 = stats(D, toks)
    ca, cs = content_alpha(D, toks, False), content_alpha(D, toks, True)
    print(f"  {tag:28s} alpha {a:.3f} | range at 512 {r5:9.2f}, at 2048 {r20:9.2f} | R^2(S ~ t) {r2:.3f} | "
          f"content part (clock removed): alpha {ca:.3f} real order vs {cs:.3f} shuffled")
    return a, r2


print("== TinyStories (word level, 6 layers, MapWM r4; trained at 512) ==")
from mapformer.train_tinystories import build as ts_build
val = np.memmap(f"{REPO}/data/tinystories/val.bin", dtype=np.uint16, mode="r")
for s in range(3):
    ck = torch.load(f"{REPO}/runs/tinystories/p0/Vanilla_s{s}.best.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
    m = ts_build("Vanilla", c["vocab"], c["dim"], c["heads"], c["n_layers"], c["seq_len"], c["rank"]); m.load_state_dict(ck["state_dict"])
    with torch.no_grad():
        D = m.action_to_lie(m.token_emb.weight[None])[0].reshape(c["vocab"], -1).numpy()
    report(f"TinyStories MapWM s{s}", D, np.asarray(val))
print("  clock reference (every step +1): alpha 1.000 by construction")

print("== code (byte level, 9 layers, trained and tested at 2048) ==")
from mapformer.train_hourglass_enwik8 import build as code_build
cval = np.fromfile(f"{REPO}/data/code_val.bin", dtype=np.uint8)
for arm in ("Vanilla", "MapPoPE-Flat"):
    for s in range(3):
        ck = torch.load(f"{REPO}/runs/code2048/{arm}_s{s}.best.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
        m = code_build(c["model"], dim=c["dim"], heads=c["heads"], n_layers=c["n_layers"], grid_size=c["grid_size"],
                       bottleneck_r=c["bottleneck_r"])
        m.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            D = m.action_to_lie(m.token_emb.weight[None])[0].reshape(256, -1).numpy()
        report(f"code {arm} s{s}", D, cval)
