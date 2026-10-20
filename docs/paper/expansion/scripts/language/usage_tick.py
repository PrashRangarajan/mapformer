"""Is the clock in the channels attention uses? (POST HOC, eval only, CPU.)

The pooled growth exponent (tw_alpha_drift.py) weights channels by their raw phase range, so a large tick in channels
attention ignores can make a model look like a clock. Here every rotary channel c gets
  rho_c = |mu_c| L / range(content_c, L)    (L = training length; mu = frequency-weighted mean step; omega-free)
  phi_c = |mu_c omega_c| L / (2 pi)          (turns the clock part advances over one training window)
and a USAGE weight u_c = sum over layers and heads of mean_t ||q_c,t|| x mean_t ||k_c,t|| (2D block norms; rotation does
not change them), measured on real windows. Reported: usage share in clock-dominated channels (rho_c > 1 AND phi_c > 0.5
turn: the clock part both dominates the channel and moves it by more than half a turn per window), and usage-weighted
median rho_c.
Text world: same models / stream as tw_alpha_drift.py (L = 1024; usage from the revisit-query and object words as in
tw_step_account.py, and also the all-token definition used for language, for comparability). TinyStories (L = 512)
and code (L = 2048): 8 val windows for usage, 32 for ranges.
For the text world the share of channels whose CLAUSE tick is 0 mod pi (|wrap(2 tau)|/2 < 0.05 rad: on the even
64-torus every return takes an even number of moves, so a pi tick is invisible to revisits) is printed from
tw_step_account.json.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/language")
from tw_lib import TextWorld, gen, load, ck_tw, ck_ns, step_table, REPO, wrap  # noqa

torch.set_num_threads(16)
D_ = "/home/prashr/mapformer/docs/paper/expansion/scripts/language"


def per_channel(st, W, f, L):
    mu = (f[:, None] * st).sum(0)
    Sc = np.cumsum((st - mu)[W[:, :L]], axis=1)
    rc = (Sc.max(1) - Sc.min(1)).mean(0)
    return np.abs(mu) * L / np.maximum(rc, 1e-12), mu


@torch.no_grad()
def usage_alltok(m, W):
    caps = []
    hooks = []
    for ly in m.layers:
        def hq(mod, i, o, ly=ly):
            caps.append(("q", ly, o))
        def hk(mod, i, o, ly=ly):
            caps.append(("k", ly, o))
        hooks += [ly.q_proj.register_forward_hook(hq), ly.k_proj.register_forward_hook(hk)]
    u = None
    for w in W:
        caps.clear()
        m(torch.as_tensor(np.asarray(w, dtype=np.int64))[None])
        qs = [o for t, _, o in caps if t == "q"]; ks = [o for t, _, o in caps if t == "k"]
        acc = 0
        for q, k in zip(qs, ks):
            H = m.n_heads; dh = q.shape[-1] // H
            qn = q[0].view(-1, H, dh // 2, 2).norm(dim=-1).mean(0)
            kn = k[0].view(-1, H, dh // 2, 2).norm(dim=-1).mean(0)
            acc = acc + (qn * kn).reshape(-1)
        u = acc if u is None else u + acc
    for h in hooks:
        h.remove()
    u = u.numpy(); return u / u.sum()


def summarize(tag, rho, mu, om, u, L, extra=""):
    phi = np.abs(mu * om) * L / (2 * np.pi)
    clock = (rho > 1) & (phi > 0.5)
    o = np.argsort(rho); cw = np.cumsum(u[o]); med = float(rho[o][np.searchsorted(cw, 0.5)])
    r = {"usage_in_clock_ch": float(u[clock].sum()), "n_clock_ch": int(clock.sum()), "n_ch": int(len(rho)),
         "usage_wtd_median_rho": med, "unweighted_median_rho": float(np.median(rho))}
    print(f"  {tag:14s} clock-dominated channels {r['n_clock_ch']:3d}/{r['n_ch']} carry usage {r['usage_in_clock_ch']:.3f}"
          f" | usage-weighted median rho {med:8.3f} (unweighted {r['unweighted_median_rho']:8.3f}){extra}", flush=True)
    return r


def main():
    out = {"textworld": {}, "language": {}}
    te = TextWorld(size=64, seed=10000); np.random.seed(424243)
    G = [gen(te, 2048) for _ in range(32)]
    W = np.stack([g["tok"] for g in G]); V = len(te.vocab)
    f = np.bincount(W.ravel(), minlength=V).astype(float); f /= f.sum()
    SA = json.load(open(f"{D_}/tw_step_account.json"))
    print("text world (L = 1024): usage from retrieval words [all-token usage in brackets]")
    models = [(f"tw_s{s}", ck_tw(s)) for s in range(8)]
    models += [(f"{arm}_s{s}", ck_ns(arm, s)) for arm in ("MapWM", "NormStep") for s in range(10, 18)]
    for name, ck in models:
        m, _ = load(ck)
        st = step_table(m, V).reshape(V, -1).numpy()
        om = m.path_integrator.omega.detach().numpy().reshape(-1)
        rho, mu = per_channel(st, W, f, 1024)
        u = np.array(SA[name]["usage"])
        ua = usage_alltok(m, W[:8, :1024])
        tau = np.array(SA[name]["tau"])
        zpi = np.abs(wrap(2 * tau)) / 2 < 0.05
        r = summarize(name, rho, mu, om, u, 1024,
                      extra=f" [all-token usage in clock ch {float(ua[(rho > 1) & (np.abs(mu * om) * 1024 / (2 * np.pi) > 0.5)].sum()):.3f}]"
                            f" | clause tick 0 mod pi: {int(zpi.sum())} ch, usage {float(u[zpi].sum()):.3f}"
                            f" (of which ~pi: {int((zpi & (np.abs(wrap(tau)) > 1.5)).sum())} ch, usage {float(u[zpi & (np.abs(wrap(tau)) > 1.5)].sum()):.3f})")
        r["usage_tick0modpi"] = float(u[zpi].sum())
        out["textworld"][name] = r
    rng = np.random.default_rng(0)
    print("TinyStories (L = 512; all-token usage over 6 layers x 8 heads):")
    from mapformer.train_tinystories import build as ts_build
    val = np.asarray(np.memmap(f"{REPO}/data/tinystories/val.bin", dtype=np.uint16, mode="r"))
    for s in range(3):
        ck = torch.load(f"{REPO}/runs/tinystories/p0/Vanilla_s{s}.best.pt", map_location="cpu", weights_only=False)
        c = ck["cfg"]
        m = ts_build("Vanilla", c["vocab"], c["dim"], c["heads"], c["n_layers"], c["seq_len"], c["rank"])
        m.load_state_dict(ck["state_dict"]); m.eval()
        st = m.action_to_lie(m.token_emb.weight[None])[0].reshape(c["vocab"], -1).detach().numpy()
        om = m.path_integrator.omega.detach().numpy().reshape(-1)
        fv = np.bincount(val[:5_000_000].astype(np.int64), minlength=c["vocab"]).astype(float); fv /= fv.sum()
        starts = rng.choice(len(val) - 2049, 32, replace=False)
        Wv = np.stack([val[x:x + 2048] for x in starts]).astype(np.int64)
        rho, mu = per_channel(st, Wv, fv, 512)
        u = usage_alltok(m, Wv[:8, :512])
        out["language"][f"TinyStories_s{s}"] = summarize(f"TinyStories s{s}", rho, mu, om, u, 512)
    print("code (L = 2048; all-token usage over 9 layers x 8 heads):")
    from mapformer.train_hourglass_enwik8 import build as code_build
    cval = np.fromfile(f"{REPO}/data/code_val.bin", dtype=np.uint8)
    for s in range(3):
        ck = torch.load(f"{REPO}/runs/code2048/Vanilla_s{s}.best.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
        m = code_build(c["model"], dim=c["dim"], heads=c["heads"], n_layers=c["n_layers"], grid_size=c["grid_size"],
                       bottleneck_r=c["bottleneck_r"])
        m.load_state_dict(ck["state_dict"]); m.eval()
        st = m.action_to_lie(m.token_emb.weight[None])[0].reshape(256, -1).detach().numpy()
        om = m.path_integrator.omega.detach().numpy().reshape(-1)
        fv = np.bincount(cval.astype(np.int64), minlength=256).astype(float); fv /= fv.sum()
        starts = rng.choice(len(cval) - 4097, 32, replace=False)
        Wv = np.stack([cval[x:x + 2048] for x in starts]).astype(np.int64)
        rho, mu = per_channel(st, Wv, fv, 2048)
        u = usage_alltok(m, Wv[:4, :2048])
        out["language"][f"code_s{s}"] = summarize(f"code s{s}", rho, mu, om, u, 2048)
    json.dump(out, open(f"{D_}/usage_tick.json", "w"), indent=1)


if __name__ == "__main__":
    main()
