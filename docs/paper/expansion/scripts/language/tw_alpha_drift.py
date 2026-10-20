"""Clock vs map as drift vs content in the learned phase (POST HOC, eval only, CPU).

Decomposition. With a context-free step table s_w (per channel, raw, before omega) and token frequencies f_w, write
s_w = mu + d_w, mu = sum_w f_w s_w (the frequency-weighted mean step: the CLOCK part) and d_w the content part. The
accumulated phase over T tokens is S(T) = mu T + sum d. Readouts per model, pooled over channels exactly as in
docs/audits/2026-10-12/alpha_language.py (range = max - min of S over the first T tokens, averaged over channels and
windows; alpha = slope of log range on log T, T = 64..2048):
  alpha        growth exponent of S (1 = clock; the walk's own exponent = map; 0.5 = iid random walk);
  rho(1024)    drift-to-content ratio sum_c |mu_c| T / sum_c range(content_c, T) at T = 1024 (the training length of
               the text world): rho >> 1 means the clock part dominates the phase over a training window;
  alpha_pred   alpha of the curve |mu| T + range(content, T) (the two-part model), to check that drift + content
               accounts for the measured alpha;
  text world only: alpha of the TRUE unwrapped walk coordinates (the reference a map must reproduce), and alpha / rho
               restricted to the drifting and non-drifting channels of tw_step_account.json.
Data: text world 32 walks x 2048 words on the held-out map (np seed 424243); TinyStories val (word level) and code val
(bytes) as in alpha_language.py (32 windows x 2048, rng seed 0).
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/language")
from tw_lib import TextWorld, gen, load, ck_tw, ck_ns, step_table, REPO  # noqa

torch.set_num_threads(8)
D_ = "/home/prashr/mapformer/docs/paper/expansion/scripts/language"
TS = [64, 128, 256, 512, 1024, 2048]


def ranges(S):
    """S: (n_win, T, C) -> (len(TS), C) mean range per channel."""
    return np.stack([(S[:, :T].max(1) - S[:, :T].min(1)).mean(0) for T in TS])


def alpha_of(r):
    return float(np.polyfit(np.log(TS), np.log(r), 1)[0])


def readouts(st, W, f, chans=None):
    """st (V, C) raw steps; W (n_win, 2048) token windows; f (V,) frequencies."""
    if chans is not None:
        st = st[:, chans]
    mu = (f[:, None] * st).sum(0)
    S = np.cumsum(st[W], axis=1)
    Sc = np.cumsum((st - mu)[W], axis=1)
    r, rc = ranges(S), ranges(Sc)
    a = alpha_of(r.mean(1)); ac = alpha_of(rc.mean(1))
    i = TS.index(1024)
    rho = float(np.abs(mu).sum() * 1024 / rc[i].sum())
    pred = np.abs(mu)[None] * np.array(TS)[:, None] + rc
    t = np.arange(1, W.shape[1] + 1, dtype=float); t = (t - t.mean())[None, :, None]
    S0 = S - S.mean(1, keepdims=True); b = (S0 * t).sum(1, keepdims=True) / (t ** 2).sum()
    r2 = float(1 - ((S0 - b * t) ** 2).sum() / (S0 ** 2).sum())
    return {"alpha": a, "alpha_content": ac, "rho1024": rho, "alpha_pred": alpha_of(pred.mean(1)), "R2_t": r2}


def textworld():
    te = TextWorld(size=64, seed=10000); np.random.seed(424243)
    G = [gen(te, 2048) for _ in range(32)]
    W = np.stack([g["tok"] for g in G])
    V = len(te.vocab)
    f = np.bincount(W.ravel(), minlength=V).astype(float); f /= f.sum()
    # true unwrapped coordinates per token
    mv = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
    word2dir = {i: a for a in range(4) for i in te.dir_ids[a]}
    XY = np.zeros((len(G), 2048, 2))
    for k, g in enumerate(G):
        step = np.array([mv[word2dir[w]] if w in word2dir else (0, 0) for w in g["tok"]], float)
        XY[k] = np.cumsum(step, 0)
    ref = alpha_of(ranges(XY).mean(1))
    print(f"text world: true unwrapped walk coordinates alpha {ref:.3f} (the map reference); clock reference 1.000")
    SA = json.load(open(f"{D_}/tw_step_account.json"))
    out = {"walk_alpha": ref}
    models = [(f"tw_s{s}", ck_tw(s)) for s in range(8)]
    models += [(f"{arm}_s{s}", ck_ns(arm, s)) for arm in ("MapWM", "NormStep", "DirOnly") for s in range(10, 18)]
    for name, ck in models:
        m, _ = load(ck)
        st = step_table(m, V).reshape(V, -1).numpy()
        drift = np.array(SA[name]["drift_obs"]) > 1.0
        r = {"all": readouts(st, W, f)}
        if drift.sum() >= 4:
            r["drift_ch"] = readouts(st, W, f, np.nonzero(drift)[0])
            r["map_ch"] = readouts(st, W, f, np.nonzero(~drift)[0])
        out[name] = r
        a = r["all"]
        line = (f"  {name:13s} alpha {a['alpha']:.3f} (pred {a['alpha_pred']:.3f}) rho {a['rho1024']:7.3f} "
                f"content alpha {a['alpha_content']:.3f} R2(t) {a['R2_t']:.3f}")
        if "drift_ch" in r:
            line += (f" | drift ch ({int(drift.sum())}): alpha {r['drift_ch']['alpha']:.3f} rho {r['drift_ch']['rho1024']:.2f}"
                     f" | map ch: alpha {r['map_ch']['alpha']:.3f} rho {r['map_ch']['rho1024']:.3f}")
        print(line, flush=True)
    return out


def language():
    rng = np.random.default_rng(0)
    out = {}

    def run(tag, st, toks):
        f = np.bincount(np.asarray(toks[:5_000_000], dtype=np.int64), minlength=st.shape[0]).astype(float); f /= f.sum()
        starts = rng.choice(len(toks) - TS[-1] - 1, 32, replace=False)
        W = np.stack([toks[s:s + TS[-1]] for s in starts]).astype(np.int64)
        r = readouts(st, W, f); out[tag] = r
        print(f"  {tag:24s} alpha {r['alpha']:.3f} (pred {r['alpha_pred']:.3f}) rho {r['rho1024']:7.3f} "
              f"content alpha {r['alpha_content']:.3f} R2(t) {r['R2_t']:.3f}", flush=True)

    print("TinyStories (MapWM r4, 6 layers, trained at 512):")
    from mapformer.train_tinystories import build as ts_build
    val = np.memmap(f"{REPO}/data/tinystories/val.bin", dtype=np.uint16, mode="r")
    for s in range(3):
        ck = torch.load(f"{REPO}/runs/tinystories/p0/Vanilla_s{s}.best.pt", map_location="cpu", weights_only=False)
        c = ck["cfg"]
        m = ts_build("Vanilla", c["vocab"], c["dim"], c["heads"], c["n_layers"], c["seq_len"], c["rank"])
        m.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            st = m.action_to_lie(m.token_emb.weight[None])[0].reshape(c["vocab"], -1).numpy()
        run(f"TinyStories s{s}", st, np.asarray(val))
    print("code (bytes, MapWM, 9 layers, trained at 2048):")
    from mapformer.train_hourglass_enwik8 import build as code_build
    cval = np.fromfile(f"{REPO}/data/code_val.bin", dtype=np.uint8)
    for s in range(3):
        ck = torch.load(f"{REPO}/runs/code2048/Vanilla_s{s}.best.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
        m = code_build(c["model"], dim=c["dim"], heads=c["heads"], n_layers=c["n_layers"], grid_size=c["grid_size"],
                       bottleneck_r=c["bottleneck_r"])
        m.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            st = m.action_to_lie(m.token_emb.weight[None])[0].reshape(256, -1).numpy()
        run(f"code MapWM s{s}", st, cval)
    return out


def main():
    out = {"textworld": textworld(), "language": language()}
    json.dump(out, open(f"{D_}/tw_alpha_drift.json", "w"), indent=1)


if __name__ == "__main__":
    main()
