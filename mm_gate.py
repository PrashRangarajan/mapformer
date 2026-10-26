"""Memory Maze CPU gate (registration: MEMORY_MAZE_GATE.md). numpy only, streams episodes from the official zips.

Usage (CPU job; no GPU, no torch):
  python3 -I mm_gate.py --eval-zip <size>/eval.zip --train-zip <15x15>/train-part0.zip --out gate_15x15.json
Writes one JSON (per-size summary, stop-rule inputs and verdicts) and prints a text summary.
Definitions (pose bin, scored re-entry step, gap, backtrack, readouts R1/R2, floors) are those of the registration.
"""
import argparse
import io
import json
import sys
import time
import zipfile

import numpy as np

ACTIONS = ["noop", "forward", "left", "right", "forward_left", "forward_right"]
FORWARD_TYPE = (1, 4, 5)
LONG_GAP = 128
N_HB = 8
VR_VARIANTS = [(1, 0), (1, 16), (4, 0), (4, 16)]  # (m frames matched, k recent steps excluded)
SPACES = ["pix16", "pcaw64"]


# ---------------------------------------------------------------- data
def iter_episodes(path, n, keys=("image", "action", "agent_pos", "agent_dir", "maze_layout")):
    z = zipfile.ZipFile(path)
    names = sorted(x for x in z.namelist() if x.endswith(".npz"))
    for name in names[:n]:
        ep = np.load(io.BytesIO(z.read(name)))
        out = {k: ep[k] for k in keys}
        out["name"] = name
        yield out


def n_members(path):
    return sum(1 for x in zipfile.ZipFile(path).namelist() if x.endswith(".npz"))


def pix16(img):
    t = img.shape[0]
    return img.reshape(t, 16, 4, 16, 4, 3).astype(np.float32).mean(axis=(2, 4)).reshape(t, -1) / 255.0


def heading_bin(d):
    ang = np.arctan2(d[:, 1], d[:, 0])
    return np.round(ang / (np.pi / 4)).astype(int) % N_HB


def actions_of(ep):
    a = ep["action"].argmax(1)
    a[ep["action"].sum(1) == 0] = -1  # row 0: no action yet
    return a


def sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


# ---------------------------------------------------------------- fits on train episodes
def fit(train_zip, n_train, n_pca_frames=60000, ridge=1.0, rng=None):
    t0 = time.time()
    frames, contXX, contXY = [], {}, {}
    disp_sum = {}  # (align, a, hb) -> [sum_dx, sum_dy, n]
    turn = {a: [] for a in range(6)}
    lsC = {}  # key -> [F^T F, F^T d]; F = [dir_{t-1}, dir_t], per action (C) or per (a_{t-1}, a_t) (E)
    probe_X, probe_h, probe_c = [], [], []
    all_d, all_keys = {"prev": [], "cur": []}, {"prev": [], "cur": []}
    n_ep = 0
    for ep in iter_episodes(train_zip, n_train):
        n_ep += 1
        X = pix16(ep["image"])
        a = actions_of(ep)
        pos, dr = ep["agent_pos"], ep["agent_dir"]
        hb = heading_bin(dr)
        N = ep["maze_layout"].shape[-1]
        sel = rng.choice(len(X), size=min(len(X), max(1, n_pca_frames // max(n_train, 1))), replace=False)
        frames.append(X[sel])
        probe_X.append(X[sel]); probe_h.append(hb[sel])
        cell = np.clip(np.floor(pos).astype(int), 0, N - 1)
        probe_c.append(cell[sel, 0] * N + cell[sel, 1])
        # continuation regression per action: [x_{t-1}, x_{t-2}, 1] -> x_t
        for t_act in range(6):
            ts = np.where(a[2:] == t_act)[0] + 2
            if len(ts) == 0:
                continue
            F = np.hstack([X[ts - 1], X[ts - 2], np.ones((len(ts), 1), np.float32)]).astype(np.float64)
            contXX[t_act] = contXX.get(t_act, 0) + F.T @ F
            contXY[t_act] = contXY.get(t_act, 0) + F.T @ X[ts].astype(np.float64)
        # action tables
        d = np.diff(pos, axis=0)  # d[t-1] = pos[t] - pos[t-1], caused by a[t]
        th = np.arctan2(dr[:, 1], dr[:, 0])
        dth = (np.diff(th) + np.pi) % (2 * np.pi) - np.pi
        Fh = np.hstack([dr[:-1], dr[1:]])
        for t in range(1, len(a)):
            if a[t] < 0:
                continue
            for key in (("C", int(a[t])), ("E", int(a[t - 1]), int(a[t]))):
                acc = lsC.setdefault(key, [np.zeros((4, 4)), np.zeros((4, 2))])
                acc[0] += np.outer(Fh[t - 1], Fh[t - 1]); acc[1] += np.outer(Fh[t - 1], d[t - 1])
            turn[a[t]].append(dth[t - 1])
            for align, h in (("prev", hb[t - 1]), ("cur", hb[t])):
                key = (align, int(a[t]), int(h))
                s = disp_sum.setdefault(key, [0.0, 0.0, 0])
                s[0] += d[t - 1, 0]; s[1] += d[t - 1, 1]; s[2] += 1
                all_d[align].append(d[t - 1]); all_keys[align].append((int(a[t]), int(h)))
    F = np.concatenate(frames).astype(np.float64)
    mu = F.mean(0)
    C = np.cov(F - mu, rowvar=False)
    ev, V = np.linalg.eigh(C)
    idx = np.argsort(ev)[::-1][:64]
    W = (V[:, idx] / np.sqrt(ev[idx] + 1e-6)).astype(np.float32)
    B = {}
    for t_act in contXX:
        XX = contXX[t_act]
        B[t_act] = np.linalg.solve(XX + ridge * np.eye(XX.shape[0]), contXY[t_act]).astype(np.float32)
    tables = {}
    r2 = {}
    for align in ("prev", "cur"):
        tab = np.zeros((6, N_HB, 2))
        for (al, act, h), (sx, sy, n) in disp_sum.items():
            if al == align and n > 0:
                tab[act, h] = (sx / n, sy / n)
        tables[align] = tab
        D = np.array(all_d[align]); K = np.array(all_keys[align])
        pred = tab[K[:, 0], K[:, 1]]
        r2[align] = float(1 - ((D - pred) ** 2).sum() / ((D - D.mean(0)) ** 2).sum())
    align = max(r2, key=r2.get)
    MC = np.zeros((6, 4, 2)); ME = np.zeros((7, 6, 4, 2))  # E: a_{t-1} = -1 (none) stored at index 6
    for key, (FF, Fd) in lsC.items():
        M = np.linalg.solve(FF + 1e-6 * np.eye(4), Fd)
        if key[0] == "C":
            MC[key[1]] = M
        else:
            ME[key[1] if key[1] >= 0 else 6, key[2]] = M
    turn_tab = np.array([float(np.mean(turn[k])) if turn[k] else 0.0 for k in range(6)])
    # probes: ridge (one-hot least squares) from pcaw64 -> heading bin, cell
    PX = (np.concatenate(probe_X) - mu.astype(np.float32)) @ W
    PX1 = np.hstack([PX, np.ones((len(PX), 1), np.float32)]).astype(np.float64)
    G = PX1.T @ PX1 + 1.0 * np.eye(PX1.shape[1])
    ph = np.concatenate(probe_h); pc = np.concatenate(probe_c)
    Wh = np.linalg.solve(G, PX1.T @ np.eye(N_HB)[ph])
    ncell = N * N
    Wc = np.linalg.solve(G, PX1.T @ np.eye(ncell)[pc])
    print(f"fit: {n_ep} train episodes, {len(F)} PCA frames, table align={align} r2={r2}, {time.time()-t0:.0f}s",
          flush=True)
    return dict(mu=mu.astype(np.float32), W=W, B=B, tables=tables, align=align, table_r2_train=r2,
                turn_tab=turn_tab, MC=MC, ME=ME, Wh=Wh, Wc=Wc, ncell=ncell, N_fit=N, n_train_eps=n_ep,
                pca_var_frac=float(ev[idx].sum() / ev.sum()))


# ---------------------------------------------------------------- per-episode gate
ORIENT = [lambda L: L, lambda L: L.T, lambda L: L[::-1], lambda L: L[:, ::-1], lambda L: L[::-1, ::-1],
          lambda L: L.T[::-1], lambda L: L.T[:, ::-1], lambda L: L.T[::-1, ::-1]]


def rle(seq):
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def episode(ep, fitd):
    X16 = pix16(ep["image"])
    XP = (X16 - fitd["mu"]) @ fitd["W"]
    a = actions_of(ep)
    pos, dr = ep["agent_pos"], ep["agent_dir"]
    L = ep["maze_layout"][0]
    N = L.shape[-1]
    T = len(a)
    cell = np.clip(np.floor(pos).astype(int), 0, N - 1)
    cid = cell[:, 0] * N + cell[:, 1]
    hb = heading_bin(dr)
    pose = cid * N_HB + hb

    # layout orientation and open cells
    visited = np.unique(cell, axis=0)
    best = None
    for oi, f in enumerate(ORIENT):
        LT = np.asarray(f(L))
        vals = LT[visited[:, 0], visited[:, 1]]
        maj = np.bincount(vals.astype(int)).argmax()
        share = float((vals == maj).mean())
        if best is None or share > best[1]:
            best = (oi, share, int(maj), int((LT == maj).sum()))
    orient, orient_share, open_val, open_cells = best

    # scored re-entry steps
    last = {}
    runs_in_cell = {}
    scored, gaps, back, prior, s_last = [], [], [], [], []
    for t in range(T):
        p = int(pose[t])
        if t >= 1 and p != pose[t - 1] and p in last:
            s = last[p]
            scored.append(t); gaps.append(t - s); s_last.append(s)
            r = rle(list(cid[s:t + 1]))
            back.append(r == r[::-1])
            prior.append(runs_in_cell.get(int(cid[t]), 0))
        if t == 0 or cid[t] != cid[t - 1]:
            runs_in_cell[int(cid[t])] = runs_in_cell.get(int(cid[t]), 0) + 1
        last[p] = t
    scored = np.array(scored, int); gaps = np.array(gaps, int); back = np.array(back, bool)
    s_last = np.array(s_last, int)

    rec = dict(T=T, N=N, orient=orient, orient_share=orient_share, open_val=open_val, open_cells=open_cells,
               distinct_cells=int(len(visited)), distinct_pose=int(len(np.unique(pose))),
               n_scored=int(len(scored)), gaps=gaps.tolist(), backtrack=back.tolist(),
               prior_runs=prior)

    # action statistics
    d = np.diff(pos, axis=0)
    dn = np.linalg.norm(d, axis=1)
    th = np.arctan2(dr[:, 1], dr[:, 0])
    dth = (np.diff(th) + np.pi) % (2 * np.pi) - np.pi
    at = a[1:]
    ok = at >= 0
    act = {}
    for k in range(6):
        m = ok & (at == k)
        act[ACTIONS[k]] = dict(n=int(m.sum()), disp_sum=float(dn[m].sum()), dth_abs_sum=float(np.abs(dth[m]).sum()),
                              n_still=int((dn[m] < 0.05).sum()))
    fw = ok & np.isin(at, FORWARD_TYPE) & (dn >= 0.05)
    cosp = (d[fw] * dr[:-1][fw]).sum(1) / dn[fw]
    cosc = (d[fw] * dr[1:][fw]).sum(1) / dn[fw]
    rec["actions"] = act
    rec["cos_prev_sum"] = float(cosp.sum()); rec["cos_cur_sum"] = float(cosc.sum()); rec["n_fw_moving"] = int(fw.sum())

    # dead reckoning in P_joint's class (table d(action, heading bin)); relative error between s_last and t
    tab = fitd["tables"][fitd["align"]]
    hb_use = hb[:-1] if fitd["align"] == "prev" else hb[1:]
    dhat = np.zeros_like(d)
    dhat[ok] = tab[at[ok], hb_use[ok]]
    ss_res = float(((d[ok] - dhat[ok]) ** 2).sum()); ss_tot = float(((d[ok] - d[ok].mean(0)) ** 2).sum())
    cum = np.vstack([np.zeros((1, 2)), np.cumsum(dhat, axis=0)])
    # variant B: heading dead-reckoned from a per-action turn table, starting from the true initial heading
    th_hat = th[0] + np.concatenate([[0.0], np.cumsum(np.where(ok, fitd["turn_tab"][np.maximum(at, 0)], 0.0))])
    hb_hat = np.round(np.arctan2(np.sin(th_hat), np.cos(th_hat)) / (np.pi / 4)).astype(int) % N_HB
    hb_hat_use = hb_hat[:-1] if fitd["align"] == "prev" else hb_hat[1:]
    dhatB = np.zeros_like(d)
    dhatB[ok] = tab[at[ok], hb_hat_use[ok]]
    cumB = np.vstack([np.zeros((1, 2)), np.cumsum(dhatB, axis=0)])
    Fh = np.hstack([dr[:-1], dr[1:]])
    dhatC = np.zeros_like(d); dhatE = np.zeros_like(d)
    dhatC[ok] = np.einsum("ni,nij->nj", Fh[ok], fitd["MC"][at[ok]])
    aprev = np.where(a[:-1] >= 0, a[:-1], 6)
    dhatE[ok] = np.einsum("ni,nij->nj", Fh[ok], fitd["ME"][aprev[ok], at[ok]])
    cumC = np.vstack([np.zeros((1, 2)), np.cumsum(dhatC, axis=0)])
    cumE = np.vstack([np.zeros((1, 2)), np.cumsum(dhatE, axis=0)])
    true_rel = pos[scored] - pos[s_last]
    errA = np.linalg.norm((cum[scored] - cum[s_last]) - true_rel, axis=1)
    errB = np.linalg.norm((cumB[scored] - cumB[s_last]) - true_rel, axis=1)
    errC = np.linalg.norm((cumC[scored] - cumC[s_last]) - true_rel, axis=1)
    errE = np.linalg.norm((cumE[scored] - cumE[s_last]) - true_rel, axis=1)
    rec.update(table_ss_res=ss_res, table_ss_tot=ss_tot, dr_errA=errA.tolist(), dr_errB=errB.tolist(),
               dr_errC=errC.tolist(), dr_errE=errE.tolist(),
               ssC=float(((d[ok] - dhatC[ok]) ** 2).sum()), ssE=float(((d[ok] - dhatE[ok]) ** 2).sum()))

    # floors and readouts
    out = {}
    for sp, Xs in (("pix16", X16), ("pcaw64", XP)):
        D = sqdist(Xs, Xs)
        np.fill_diagonal(D, 0.0)
        D4 = np.full_like(D, np.inf)
        D4[3:, 3:] = D[3:, 3:] + D[2:-1, 2:-1] + D[1:-2, 1:-2] + D[:-3, :-3]
        res = {}

        def score_vec(pv, t):  # prediction as a vector
            dv = ((Xs - pv) ** 2).sum(1)
            r1 = pose[int(np.argmin(dv))] == pose[t]
            other = pose != pose[t]
            r2 = dv[t] < dv[other].min() if other.any() else True
            return bool(r1), bool(r2)

        def score_idx(u, t):  # prediction = an existing frame u
            dv = D[u]
            r1 = pose[int(np.argmin(dv))] == pose[t]
            other = pose != pose[t]
            r2 = dv[t] < dv[other].min() if other.any() else True
            return bool(r1), bool(r2)

        keys = ["copy", "cont", "oracle", "ident"] + [f"vr_m{m}_k{k}" for m, k in VR_VARIANTS]
        for kk in keys:
            res[kk] = ([], [])
        for i, t in enumerate(scored):
            def put(kk, r):
                res[kk][0].append(r[0]); res[kk][1].append(r[1])
            put("copy", score_idx(t - 1, t))
            put("oracle", score_idx(int(s_last[i]), t))
            if t >= 2 and a[t] >= 0:
                F = np.concatenate([X16[t - 1], X16[t - 2], [1.0]]).astype(np.float32)
                p16 = F @ fitd["B"][int(a[t])]
                pv = p16 if sp == "pix16" else (p16 - fitd["mu"]) @ fitd["W"]
                put("cont", score_vec(pv, t))
            else:
                put("cont", score_idx(t - 1, t))
            # single-frame identification from the true f_t
            if t - 16 >= 0:
                u = int(np.argmin(D[: t - 15, t]))
                put("ident", score_idx(u, t))
            else:
                put("ident", (False, False))
            # view recognition
            for m, k in VR_VARIANTS:
                Dm = D if m == 1 else D4
                hi = t - 2 - k
                lo = m - 1
                if hi >= lo and a[t] >= 0:
                    cand = np.arange(lo, hi + 1)
                    cand = cand[a[cand + 1] == a[t]]
                else:
                    cand = np.array([], int)
                if len(cand):
                    s = int(cand[np.argmin(Dm[cand, t - 1])])
                    put(f"vr_m{m}_k{k}", score_idx(s + 1, t))
                else:
                    put(f"vr_m{m}_k{k}", score_idx(t - 1, t))
        out[sp] = {kk: [np.array(v[0], bool).tolist(), np.array(v[1], bool).tolist()] for kk, v in res.items()}
    rec["floors"] = out

    # cross-maze probes (frames subsample)
    sel = np.arange(0, T, 10)
    PX1 = np.hstack([XP[sel], np.ones((len(sel), 1), np.float32)]).astype(np.float64)
    rec["probe_h_correct"] = int((np.argmax(PX1 @ fitd["Wh"], 1) == hb[sel]).sum())
    # the cell probe's classes are indexed for the fit's maze size; on another size it is not defined
    rec["probe_c_correct"] = (int((np.argmax(PX1 @ fitd["Wc"], 1) == cid[sel]).sum())
                              if N == fitd["N_fit"] else -10 ** 9)
    rec["probe_n"] = int(len(sel))
    rec["probe_h_major"] = int(np.bincount(hb[sel], minlength=N_HB).max())
    return rec


# ---------------------------------------------------------------- aggregation
def boot_ci(num, den, rng, n=1000):
    num = np.asarray(num, float); den = np.asarray(den, float)
    if den.sum() == 0:
        return [float("nan")] * 3
    idx = rng.integers(0, len(num), size=(n, len(num)))
    b = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return [float(num.sum() / den.sum()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def aggregate(recs, fitd, rng):
    g = {}
    oc = np.array([r["open_cells"] for r in recs])
    g["n_episodes"] = len(recs)
    g["maze_N"] = recs[0]["N"]
    g["orient_counts"] = np.bincount([r["orient"] for r in recs], minlength=8).tolist()
    g["orient_share_min"] = float(min(r["orient_share"] for r in recs))
    g["open_val_counts"] = np.bincount([r["open_val"] for r in recs], minlength=2).tolist()
    g["open_cells"] = dict(median=float(np.median(oc)), q25=float(np.quantile(oc, .25)), q75=float(np.quantile(oc, .75)),
                           min=int(oc.min()), max=int(oc.max()), share_ge128=float((oc >= 128).mean()),
                           share_ge512=float((oc >= 512).mean()))
    dc = np.array([r["distinct_cells"] for r in recs])
    g["distinct_cells_visited"] = dict(median=float(np.median(dc)), mean=float(dc.mean()))
    g["visited_over_open"] = float(np.median(dc / oc))
    g["distinct_pose_bins"] = float(np.median([r["distinct_pose"] for r in recs]))
    ns = np.array([r["n_scored"] for r in recs])
    g["scored_per_episode"] = dict(mean=float(ns.mean()), median=float(np.median(ns)))
    gaps = np.concatenate([r["gaps"] for r in recs]) if ns.sum() else np.array([])
    backs = np.concatenate([r["backtrack"] for r in recs]).astype(bool)
    prior = np.concatenate([r["prior_runs"] for r in recs])
    g["gap_quantiles"] = {str(q): float(np.quantile(gaps, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    g["gap_share_ge"] = {str(w): float((gaps >= w).mean()) for w in (16, 32, 64, 128, 256, 512)}
    g["backtrack_share"] = float(backs.mean())
    g["backtrack_share_longgap"] = float(backs[gaps >= LONG_GAP].mean()) if (gaps >= LONG_GAP).any() else float("nan")
    g["prior_cell_entries_at_scored"] = dict(mean=float(prior.mean()), median=float(np.median(prior)))
    nl_num = [int(((np.array(r["gaps"]) >= LONG_GAP) & ~np.array(r["backtrack"], bool)).sum()) for r in recs]
    lg_num = [int((np.array(r["gaps"]) >= LONG_GAP).sum()) for r in recs]
    g["nonlocal_share"] = boot_ci(nl_num, ns, rng)
    g["longgap_share"] = boot_ci(lg_num, ns, rng)
    g["nonlocal_per_episode_mean"] = float(np.mean(nl_num))
    g["episodes_with_ge20_nonlocal"] = float(np.mean(np.array(nl_num) >= 20))

    # floors: pooled by stratum, with episode bootstrap CI
    floors = {}
    for sp in SPACES:
        floors[sp] = {}
        for kk in recs[0]["floors"][sp]:
            for ri, rname in ((0, "R1"), (1, "R2")):
                row = {}
                for stratum in ("all", "nonlocal", "longgap", "short"):
                    num, den = [], []
                    for r in recs:
                        ok = np.array(r["floors"][sp][kk][ri], bool)
                        ga = np.array(r["gaps"]); bk = np.array(r["backtrack"], bool)
                        if len(ok) == 0:
                            num.append(0); den.append(0); continue
                        m = {"all": np.ones(len(ga), bool), "nonlocal": (ga >= LONG_GAP) & ~bk,
                             "longgap": ga >= LONG_GAP, "short": ga < LONG_GAP}[stratum]
                        num.append(int(ok[m].sum())); den.append(int(m.sum()))
                    row[stratum] = boot_ci(num, den, rng)
                floors[sp][f"{kk}_{rname}"] = row
    g["floors"] = floors

    # actions
    A = {k: dict(n=0, disp=0.0, dth=0.0, still=0) for k in ACTIONS}
    for r in recs:
        for k, v in r["actions"].items():
            A[k]["n"] += v["n"]; A[k]["disp"] += v["disp_sum"]; A[k]["dth"] += v["dth_abs_sum"]; A[k]["still"] += v["n_still"]
    tot = sum(v["n"] for v in A.values())
    g["actions"] = {k: dict(share=v["n"] / tot, mean_disp=v["disp"] / max(v["n"], 1),
                            mean_abs_dturn_deg=float(np.degrees(v["dth"] / max(v["n"], 1))),
                            share_still=v["still"] / max(v["n"], 1)) for k, v in A.items()}
    fwn = sum(A[ACTIONS[k]]["n"] for k in FORWARD_TYPE)
    g["collision_share_forward_type"] = sum(A[ACTIONS[k]]["still"] for k in FORWARD_TYPE) / max(fwn, 1)
    nfw = sum(r["n_fw_moving"] for r in recs)
    g["cos_disp_heading"] = dict(prev=sum(r["cos_prev_sum"] for r in recs) / nfw, cur=sum(r["cos_cur_sum"] for r in recs) / nfw)
    g["table_align"] = fitd["align"]
    g["table_r2_train"] = fitd["table_r2_train"]
    g["table_r2_eval"] = 1 - sum(r["table_ss_res"] for r in recs) / sum(r["table_ss_tot"] for r in recs)
    g["turn_table_deg"] = dict(zip(ACTIONS, np.degrees(fitd["turn_tab"]).round(2).tolist()))
    st = sum(r["table_ss_tot"] for r in recs)
    g["r2_eval_C_continuous_heading"] = 1 - sum(r["ssC"] for r in recs) / st
    g["r2_eval_E_prev_action"] = 1 - sum(r["ssE"] for r in recs) / st
    for var in ("A", "B", "C", "E"):
        e_nl, e_all = [], []
        for r in recs:
            e = np.array(r[f"dr_err{var}"]); ga = np.array(r["gaps"]); bk = np.array(r["backtrack"], bool)
            e_all.append(e); e_nl.append(e[(ga >= LONG_GAP) & ~bk] if len(e) else e)
        e_nl = np.concatenate(e_nl); e_all = np.concatenate(e_all)
        g[f"deadreckon_{var}"] = dict(
            nonlocal_share_err_lt_0p5=float((e_nl < 0.5).mean()) if len(e_nl) else float("nan"),
            nonlocal_share_err_lt_1=float((e_nl < 1.0).mean()) if len(e_nl) else float("nan"),
            nonlocal_median_err=float(np.median(e_nl)) if len(e_nl) else float("nan"),
            all_share_err_lt_0p5=float((e_all < 0.5).mean()), all_median_err=float(np.median(e_all)))
    pn = sum(r["probe_n"] for r in recs)
    g["probe"] = dict(heading_acc=sum(r["probe_h_correct"] for r in recs) / pn,
                      heading_majority=sum(r["probe_h_major"] for r in recs) / pn,
                      heading_chance=1 / N_HB,
                      cell_acc=(sum(r["probe_c_correct"] for r in recs) / pn
                                if all(r["probe_c_correct"] >= 0 for r in recs) else None),
                      cell_chance_uniform=1 / fitd["ncell"])
    g["pca_var_frac"] = fitd["pca_var_frac"]
    g["n_train_eps"] = fitd["n_train_eps"]
    return g


def stop_rule(g):
    v = {}
    oc = g["open_cells"]
    s1a = oc["share_ge128"] < 0.5
    s1b = g["nonlocal_share"][0] < 0.10
    v["S1"] = dict(fires=bool(s1a or s1b), a_share_mazes_ge128=oc["share_ge128"], a_fires=bool(s1a),
                   b_nonlocal_share=g["nonlocal_share"], b_fires=bool(s1b))
    names = ["cont"] + [f"vr_m{m}_k{k}" for m, k in VR_VARIANTS]
    best = max(((g["floors"][sp][f"{n}_R1"]["nonlocal"][0], sp, n) for sp in SPACES for n in names),
               key=lambda x: -1 if np.isnan(x[0]) else x[0])
    v["S2"] = dict(fires=bool(best[0] >= 0.85), best_floor_nonlocal_R1=best[0], space=best[1], floor=best[2],
                   ceiling_R1=1.0, Y=0.15)
    e = g["deadreckon_A"]["nonlocal_share_err_lt_0p5"]
    v["S3"] = dict(fires=bool(not (e >= 0.5)), table_share_err_lt_0p5=e)
    v["STOP_15x15_primary"] = bool(v["S1"]["fires"] or v["S2"]["fires"])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-zip", required=True)
    ap.add_argument("--train-zip", required=True)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    meta = dict(label=args.label, eval_members=n_members(args.eval_zip), train_members=n_members(args.train_zip),
                numpy=np.__version__, python=sys.version.split()[0])
    print(meta, flush=True)
    fitd = fit(args.train_zip, args.n_train, rng=rng)
    recs = []
    for i, ep in enumerate(iter_episodes(args.eval_zip, args.n_eval)):
        recs.append(episode(ep, fitd))
        if (i + 1) % 100 == 0:
            print(f"  {i+1} episodes, {time.time()-t0:.0f}s", flush=True)
    g = aggregate(recs, fitd, rng)
    g["meta"] = meta
    g["stop_rule"] = stop_rule(g) if g["maze_N"] == 15 else "descriptive only (not 15x15)"
    g["elapsed_s"] = time.time() - t0
    with open(args.out, "w") as f:
        json.dump(g, f, indent=1)
    print(json.dumps({k: g[k] for k in ("open_cells", "nonlocal_share", "gap_share_ge", "stop_rule")}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
