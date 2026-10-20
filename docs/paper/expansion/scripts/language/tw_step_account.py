"""What a one-layer path model learns from words: per-seed step account (POST HOC, eval only, CPU).

Models: the 8 registered text-world path runs (runs/textworld/p0, seeds 0-7) and the 8 MapWM, NormStep and DirOnly runs
of TW_NORMSTEP (runs/tw_normstep/p0, seeds 10-17). Eval data: held-out map (env seed 10000), np seed 424242 (used by no
registered script), 40 walks x 1024 words.

Per model (all steps omega-scaled, i.e. radians per token per rotary channel; 64 channels = 2 heads x 32 blocks):
  1. step size by word class, relative to the direction differential |D_N| (the clause-level north step minus the tick);
  2. clause regression: each full clause's net phase change (all its words, aside included) regressed on its direction:
     per-channel TICK tau_c = mean over the 4 directions (what every clause adds whatever its direction) and
     DIFFERENTIALS D_a = A_a - tau; opposition |D_N + D_S| / |D_N| etc.;
  3. per-channel drift between consecutive visits to a cell (mean |wrap(dtheta)| at the object slots; > 1 rad = drifting,
     random ~ pi/2) and the drift PREDICTED by the tick alone, mean |wrap(lag * tau_c)| over the same revisit pairs;
  4. channel usage at revisit retrieval: mean ||q_c|| over the query words (the word before the object slot) x mean
     ||k_c|| over object words (1 layer, pre-norm: q and k are functions of the word alone);
  5. clock share (TinyStories probe definition): |sum_w f_w s_w| / sum_w f_w |s_w|, f = token frequency on the stream;
  6. aside sentence: net phase (sum over its tokens) and the aside noun's offset from its clause's object slot, both
     relative to |D_N| over the usage-weighted channels.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/language")
from tw_lib import (TextWorld, ROLES, VERBS, gen, load, ck_tw, ck_ns, omega, step_table, wrap)  # noqa

torch.set_num_threads(8)
OUT = "/home/prashr/mapformer/docs/paper/expansion/scripts/language/tw_step_account"
NW, SEED = 40, 424242


def walks():
    te = TextWorld(size=64, seed=10000); np.random.seed(SEED)
    return te, [gen(te, 1024) for _ in range(NW)]


def nrm(x, w=None):
    return float(np.sqrt((x ** 2 * (1 if w is None else w)).sum()))


@torch.no_grad()
def usage(m, te):
    ly = m.layers[0]
    E = m.token_emb.weight
    h = ly.norm1(E)
    H, dh = ly.n_heads, ly.d_head
    q = ly.q_proj(h).view(-1, H, dh // 2, 2).norm(dim=-1)    # (V, H, nb) block norms
    k = ly.k_proj(h).view(-1, H, dh // 2, 2).norm(dim=-1)
    qw = [te.idx[w] for w in ("saw", "found", "noticed", "was")]
    u = (q[qw].mean(0) * k[te.obj_ids].mean(0)).reshape(-1).numpy()
    return u / u.sum()


def account(m, te, W):
    V = len(te.vocab)
    om = omega(m).numpy().reshape(-1)
    st = step_table(m, V).reshape(V, -1).numpy() * om[None]               # radians per token
    # clause regression
    X, Y, NETA, NOFF = [], [], [], []
    freq = np.zeros(V)
    pairs_d, pairs_lag = [], []
    for g in W:
        t, cl, ro = g["tok"], g["clause"], g["role"]
        freq += np.bincount(t, minlength=V)
        d = st[t]; th = np.cumsum(d, 0)
        ncl = cl.max()                                                      # last clause may be cut
        net = np.zeros((ncl, st.shape[1]))
        np.add.at(net, cl[cl < ncl], d[cl < ncl])
        X.append(g["dirs"][:ncl]); Y.append(net)
        # aside net and noun offset
        for c in range(ncl):
            ii = np.nonzero(cl == c)[0]
            rr = ro[ii]
            if (rr == ROLES.index("anoun")).any():
                a = ii[np.isin(rr, [ROLES.index("aside"), ROLES.index("anoun"), ROLES.index("adot")])]
                NETA.append(d[a].sum(0))
                o = ii[rr == ROLES.index("obj")][0]; n_ = ii[rr == ROLES.index("anoun")][0]
                NOFF.append(th[n_] - th[o])
        # consecutive visits
        slots = np.nonzero(g["obs"])[0][:len(g["locs"])]
        last = {}
        for k_, i in enumerate(slots):
            L = tuple(g["locs"][k_])
            if L in last:
                j, kk = last[L]; pairs_d.append(wrap(th[i] - th[j])); pairs_lag.append(k_ - kk)
            last[L] = (i, k_)
    X = np.concatenate(X); Y = np.concatenate(Y)
    A = np.stack([Y[X == a].mean(0) for a in range(4)])                    # (4, C)
    tau = A.mean(0); D = A - tau
    resid = Y - A[X]
    dN = (nrm(D[0]) + nrm(D[1]) + nrm(D[2]) + nrm(D[3])) / 4
    opp = (nrm(D[0] + D[1]) / ((nrm(D[0]) + nrm(D[1])) / 2) + nrm(D[2] + D[3]) / ((nrm(D[2]) + nrm(D[3])) / 2)) / 2
    pd = np.abs(np.array(pairs_d)); lag = np.array(pairs_lag)
    drift_obs = pd.mean(0)
    drift_pred = np.abs(wrap(lag[:, None] * tau[None])).mean(0)
    u = usage(m, te)
    f = freq / freq.sum()
    clock_share = nrm((f[:, None] * st).sum(0)) / float((f * np.linalg.norm(st, axis=1)).sum())
    # word classes (word-type step norms, relative to dN)
    cls = {"direction": [i for a in range(4) for i in te.dir_ids[a]], "verbs": [te.idx[w] for w in VERBS],
           "adverbs": [te.idx[w] for w in ("slowly", "quickly", "carefully", "quietly")],
           "see": [te.idx[w] for w in ("saw", "found", "noticed", "there", "was")],
           "objects+nothing": te.obj_ids, "period": [te.idx["."]],
           "aside-only": [te.idx[w] for w in ("thought", "about", "remembered", "story")],
           "fillers": [te.idx[w] for w in ("then", "paused", "looked", "around", "for", "moment")],
           "shared (and/a/she)": [te.idx[w] for w in ("and", "a", "she")]}
    cs = {k: float(np.linalg.norm(st[v], axis=1).mean() / dN) for k, v in cls.items()}
    neta = np.array(NETA); noff = np.array(NOFF)
    # usage-weighted relative sizes
    uw = u / u.sum()
    dN_u = np.mean([nrm(D[a], uw) for a in range(4)])
    res = {
        "dN_rad": dN, "opposition_D": opp, "tick_over_dN": nrm(tau) / dN,
        "tick_over_dN_usage": nrm(tau, uw) / dN_u,
        "resid_sd_over_dN": float(np.sqrt((resid ** 2).sum(1).mean()) / dN),
        "n_drift_obs": int((drift_obs > 1.0).sum()), "n_drift_pred": int((drift_pred > 1.0).sum()),
        "drift_agree": float(((drift_obs > 1.0) == (drift_pred > 1.0)).mean()),
        "r_drift_obs_pred": float(np.corrcoef(drift_obs, drift_pred)[0, 1]),
        "usage_on_drift": float(u[drift_obs > 1.0].sum()),
        "median_lag": float(np.median(lag)), "n_pairs": int(len(lag)),
        "clock_share": clock_share, "class_rel_dN": cs,
        "aside_net_over_dN_usage": float(np.mean([nrm(x, uw) for x in neta]) / dN_u),
        "aside_noun_offset_over_dN_usage": float(np.mean([nrm(wrap(x), uw) for x in noff]) / dN_u),
        "aside_noun_offset_rad_usage_mean_abs": float(np.mean([(np.abs(wrap(x)) * uw).sum() for x in noff])),
        "cos_tick_verb": float(tau @ st[[te.idx[w] for w in VERBS]].mean(0)
                               / (nrm(tau) * nrm(st[[te.idx[w] for w in VERBS]].mean(0)) + 1e-12)),
        "drift_obs": drift_obs.tolist(), "drift_pred": drift_pred.tolist(), "usage": u.tolist(), "tau": tau.tolist(),
    }
    return res


def main():
    te, W = walks()
    out = {}
    models = [(f"tw_s{s}", ck_tw(s)) for s in range(8)]
    models += [(f"{arm}_s{s}", ck_ns(arm, s)) for arm in ("MapWM", "NormStep", "DirOnly") for s in range(10, 18)]
    print(f"{'model':14s} {'opp':>6s} {'tick/dN':>8s} {'tick/dN(u)':>10s} {'resid':>6s} {'drift obs/pred':>14s} "
          f"{'agree':>5s} {'r':>5s} {'use@drift':>9s} {'clock':>6s} {'asideNet':>8s} {'nounOff':>7s} {'nounOff rad':>11s} cos(tick,verb)")
    for name, ck in models:
        m, arm = load(ck)
        r = account(m, te, W); out[name] = r
        print(f"{name:14s} {r['opposition_D']:6.3f} {r['tick_over_dN']:8.3f} {r['tick_over_dN_usage']:10.3f} "
              f"{r['resid_sd_over_dN']:6.3f} {r['n_drift_obs']:7d}/{r['n_drift_pred']:<6d} {r['drift_agree']:5.2f} "
              f"{r['r_drift_obs_pred']:5.2f} {r['usage_on_drift']:9.3f} {r['clock_share']:6.3f} "
              f"{r['aside_net_over_dN_usage']:8.3f} {r['aside_noun_offset_over_dN_usage']:7.3f} "
              f"{r['aside_noun_offset_rad_usage_mean_abs']:11.3f} {r['cos_tick_verb']:+.3f}", flush=True)
        print("     class step / dN: " + ", ".join(f"{k} {v:.3f}" for k, v in r["class_rel_dN"].items()), flush=True)
    print(f"median lag between consecutive visits (clauses): {out['tw_s0']['median_lag']}, pairs {out['tw_s0']['n_pairs']}")
    json.dump(out, open(OUT + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
