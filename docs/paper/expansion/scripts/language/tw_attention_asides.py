"""Attention at revisits, causal aside manipulation, and clock-channel ablation (POST HOC, eval only, CPU).

Same models and eval stream as tw_step_account.py (held-out map 10000, np seed 424242, 40 walks x 1024 words).
Accuracy is read with the 1/(1-p) attention-dropout scale (x1.111, the registered re-score of TW_NORMSTEP /
TW_STATECHANGE); eval-mode (x1) is printed beside the intact number.

Per model:
  attention  at each revisit query (the word before a revisited object slot), the attention mass (mean over the 2 heads)
             on: the object slot of the most recent earlier visit to that cell (obj_last), object slots of older visits
             (obj_old), aside nouns told at earlier visits to that cell (anoun), any other token of an earlier clause at
             that cell (same_other), and everything else (rest);
  collapse   every aside sentence's words are put at its clause's OBJECT-SLOT phase, the phase after the aside unchanged
             (net displacement kept; only the within-sentence offset is removed). Prediction of the aside account: the
             learned-step models start making DirOnly's error (naming an aside object told at that cell);
  lift       the reverse on every model: the aside's words are offset by a fixed vector v (raw step space) and the phase
             after the aside is unchanged. v = off-plane (random, orthogonal to the 4 direction steps, norm = one move
             |D_N| raw) or in-plane (half a north move). Prediction: DirOnly's aside errors vanish;
  aside-err  share of errors whose prediction is an object told in an aside at an earlier visit to the same cell;
  clock      on models with drifting channels (tw_step_account.json: mean |wrap dtheta| > 1 rad between consecutive
             visits): accuracy with those channels removed from the score (q, k components zeroed), with the same
             number of non-drifting channels removed at random (5 draws; capped at the number of non-drifting channels), and with all non-drifting channels removed.
Accuracy is also split by lag (clauses since the last visit to the cell): 1-2, 3-16, > 16.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr/mapformer/docs/paper/expansion/scripts/language")
from tw_lib import (TextWorld, ROLES, SCALE, gen, load, ck_tw, ck_ns, delta_of, forward, step_table, wrap)  # noqa

torch.set_num_threads(8)
D_ = "/home/prashr/mapformer/docs/paper/expansion/scripts/language"
NW, SEED = 40, 424242
R_AS = [ROLES.index("aside"), ROLES.index("anoun")]
R_ADOT = ROLES.index("adot")


def walks():
    te = TextWorld(size=64, seed=10000); np.random.seed(SEED)
    return te, [gen(te, 1024) for _ in range(NW)]


def meta(te, g):
    """per revisit slot i: cell, lag, earlier same-cell object slots, aside nouns (positions, words), same-cell tokens."""
    t, cl, ro = g["tok"], g["clause"], g["role"]
    ncl = cl.max() + 1
    cells = g["cells"]
    slots = np.nonzero(g["obs"])[0][:len(g["locs"])]
    by_cell = {}
    rows = []
    for k, i in enumerate(slots):
        L = cells[cl[i]]
        prev = by_cell.get(L, [])
        if g["rev"][i] and prev:
            prev_cl = [c for c, _ in prev]
            same = np.isin(cl[:i], prev_cl)
            obj_pos = [p for _, p in prev]
            an = np.nonzero(same & (ro[:i] == ROLES.index("anoun")))[0]
            rows.append(dict(i=i, lag=cl[i] - prev[-1][0], obj_last=prev[-1][1], obj_old=obj_pos[:-1],
                             anoun=an, anoun_words=set(int(x) for x in t[an]), same=np.nonzero(same)[0]))
        by_cell.setdefault(L, []).append((cl[i], i))
    return rows


def aside_spans(g):
    """list of (first aside token, noun, adot, clause object slot) per aside sentence."""
    t, cl, ro = g["tok"], g["clause"], g["role"]
    out = []
    for c in np.unique(cl):
        ii = np.nonzero(cl == c)[0]; rr = ro[ii]
        if (rr == R_ADOT).any() and (rr == ROLES.index("obj")).any():
            a = ii[np.isin(rr, R_AS)]
            out.append((a[0], a[-1], ii[rr == R_ADOT][0], ii[rr == ROLES.index("obj")][0]))
    return out


def collapse(delta, spans):
    d = delta.clone(); C = torch.cumsum(delta, 0)
    for a0, an, ad, o in spans:
        d[a0] = C[o] - C[a0 - 1]
        d[a0 + 1:an + 1] = 0
        d[ad] = C[ad] - C[o]
    return d


def lift(delta, spans, v):
    d = delta.clone()
    for a0, an, ad, o in spans:
        d[a0] = d[a0] + v; d[ad] = d[ad] - v
    return d


def score(m, te, W, M, mode=None, v=None, qk=None, scale=SCALE, attn=False):
    ok, aside_err, err = [], 0, 0
    lag_ok = {"1-2": [], "3-16": [], ">16": []}
    att = np.zeros(5); natt = 0
    for g, rows in zip(W, M):
        delta = delta_of(m, g["tok"])
        if mode == "collapse":
            delta = collapse(delta, aside_spans(g))
        elif mode == "lift":
            delta = lift(delta, aside_spans(g), v)
        res = forward(m, g["tok"], delta, scale=scale, want_attn=attn, qk_mask=qk)
        logits, A = (res if attn else (res, None))
        pred = logits.argmax(-1).numpy(); t = g["tok"]
        if attn:
            A = A.mean(0).numpy()                          # mean over heads, (T, T)
        for r in rows:
            i = r["i"]; good = pred[i - 1] == t[i]; ok.append(good)
            lag_ok["1-2" if r["lag"] <= 2 else ("3-16" if r["lag"] <= 16 else ">16")].append(good)
            if not good:
                err += 1; aside_err += int(pred[i - 1]) in (r["anoun_words"] - {int(t[i])})
            if attn:
                a = A[i - 1]
                ol = a[r["obj_last"]]; oo = a[r["obj_old"]].sum() if len(r["obj_old"]) else 0.0
                an = a[r["anoun"]].sum() if len(r["anoun"]) else 0.0
                so = a[r["same"]].sum() - ol - oo - an
                att += np.array([ol, oo, an, so, 1 - ol - oo - an - so]); natt += 1
    out = {"acc": float(np.mean(ok)), "n": len(ok), "err": err, "aside_err": aside_err,
           "aside_err_share": aside_err / max(err, 1),
           "lag": {k: (float(np.mean(v_)), len(v_)) for k, v_ in lag_ok.items()}}
    if attn:
        out["attn"] = dict(zip(["obj_last", "obj_old", "anoun", "same_other", "rest"], (att / natt).tolist()))
    return out


def raw_moves(m, te, W):
    """raw (pre-omega) clause-level direction differentials, for the lift vectors and wrap-aware opposition."""
    V = len(te.vocab)
    st = step_table(m, V).reshape(V, -1).numpy()
    X, Y = [], []
    for g in W:
        cl = g["clause"]; ncl = cl.max(); d = st[g["tok"]]
        net = np.zeros((ncl, st.shape[1])); np.add.at(net, cl[cl < ncl], d[cl < ncl])
        X.append(g["dirs"][:ncl]); Y.append(net)
    X = np.concatenate(X); Y = np.concatenate(Y)
    A = np.stack([Y[X == a].mean(0) for a in range(4)]); D = A - A.mean(0)
    om = m.path_integrator.omega.detach().numpy().reshape(-1)
    oppw = np.mean([np.linalg.norm(wrap((D[0] + D[1]) * om)) / np.linalg.norm(D[0] * om),
                    np.linalg.norm(wrap((D[2] + D[3]) * om)) / np.linalg.norm(D[2] * om)])
    return D, float(oppw)


def main():
    te, W = walks()
    M = [meta(te, g) for g in W]
    SA = json.load(open(f"{D_}/tw_step_account.json"))
    models = [(f"tw_s{s}", ck_tw(s)) for s in range(8)]
    models += [(f"{arm}_s{s}", ck_ns(arm, s)) for arm in ("MapWM", "NormStep", "DirOnly") for s in range(10, 18)]
    out = {}
    rng = np.random.default_rng(7)
    for name, ck in models:
        m, arm = load(ck)
        D, oppw = raw_moves(m, te, W)
        r = {"opp_wrapaware": oppw}
        r["intact"] = score(m, te, W, M, attn=True)
        r["intact_x1"] = score(m, te, W, M, scale=1.0)["acc"]
        r["collapse"] = score(m, te, W, M, mode="collapse")
        g0 = rng.standard_normal(D.shape[1])
        B = np.linalg.qr(D.T)[0]                              # basis of the direction span (raw)
        off = g0 - B @ (B.T @ g0); off = off / np.linalg.norm(off) * np.linalg.norm(D[0])
        H, nb = m.path_integrator.omega.shape
        r["lift_off1"] = score(m, te, W, M, mode="lift", v=torch.tensor(off, dtype=torch.float32).view(H, nb))
        r["lift_in0.5"] = score(m, te, W, M, mode="lift", v=torch.tensor(0.5 * D[0], dtype=torch.float32).view(H, nb))
        drift = np.array(SA[name]["drift_obs"]) > 1.0
        if drift.sum() >= 8:
            qk = torch.ones(H * nb); qk[torch.tensor(drift)] = 0
            r["no_clock_ch"] = score(m, te, W, M, qk=qk.view(H, nb))
            ctl = []
            for _ in range(5):
                pool = np.nonzero(~drift)[0]
                pick = rng.choice(pool, min(int(drift.sum()), len(pool)), replace=False)
                q2 = torch.ones(H * nb); q2[torch.tensor(pick)] = 0
                ctl.append(score(m, te, W, M, qk=q2.view(H, nb))["acc"])
            r["random_ch_ctl"] = ctl
            q3 = torch.ones(H * nb); q3[torch.tensor(~drift)] = 0
            r["only_clock_ch"] = score(m, te, W, M, qk=q3.view(H, nb))
        out[name] = r
        a = r["intact"]
        line = (f"{name:13s} oppW {oppw:.3f} | acc {a['acc']:.4f} (x1 {r['intact_x1']:.4f}) errs {a['err']:4d} aside {a['aside_err_share']:.2f}"
                f" | lag 1-2 {a['lag']['1-2'][0]:.3f} 3-16 {a['lag']['3-16'][0]:.3f} >16 {a['lag']['>16'][0]:.3f}"
                f" | attn last {a['attn']['obj_last']:.3f} old {a['attn']['obj_old']:.3f} anoun {a['attn']['anoun']:.3f}"
                f" same {a['attn']['same_other']:.3f} rest {a['attn']['rest']:.3f}"
                f" | collapse {r['collapse']['acc']:.4f} aside {r['collapse']['aside_err_share']:.2f}"
                f" | lift-off {r['lift_off1']['acc']:.4f} ({r['lift_off1']['aside_err_share']:.2f})"
                f" lift-in {r['lift_in0.5']['acc']:.4f} ({r['lift_in0.5']['aside_err_share']:.2f})")
        if "no_clock_ch" in r:
            c = r["no_clock_ch"]
            line += (f"\n              clock ch {int(drift.sum())}: removed -> {c['acc']:.4f} (lag 1-2 {c['lag']['1-2'][0]:.3f},"
                     f" >16 {c['lag']['>16'][0]:.3f}); random same-count -> {np.mean(r['random_ch_ctl']):.4f}"
                     f" [{min(r['random_ch_ctl']):.4f}, {max(r['random_ch_ctl']):.4f}]; only clock ch kept -> "
                     f"{r['only_clock_ch']['acc']:.4f}")
        print(line, flush=True)
    print(f"revisit targets per model: {out['tw_s0']['intact']['n']}; lag bins n: "
          + ", ".join(f"{k} {v[1]}" for k, v in out['tw_s0']['intact']['lag'].items()))
    json.dump(out, open(f"{D_}/tw_attention_asides.json", "w"), indent=1)


if __name__ == "__main__":
    main()
