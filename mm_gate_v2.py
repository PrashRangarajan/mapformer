"""Memory Maze CPU gate v2: blind stretches (registration: MEMORY_MAZE_GATE_V2.md). numpy only; imports v1's helpers.

Usage (CPU job; no GPU, no torch):
  python3 -I mm_gate_v2.py --eval-zip <15x15>/eval.zip --train-zip <15x15>/train-part0.zip --out gate_v2_15x15.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import mm_gate as g1  # noqa: E402

CONFIGS = [(16, 16), (16, 32), (32, 16), (32, 32), (64, 16), (64, 32)]  # (B blind, S sighted)
PRIMARY = (32, 32)
F3_VARIANTS = [(1, 0), (1, 16), (4, 0), (4, 16)]
CLASSES = ["A8", "A32", "C", "E8", "E"]  # registered order of increasing parameters: A8 < A32 < C < E8 < E
CLASS_ORDER = ["A8", "A32", "C", "E8", "E"]
STRATA = ("all", "nonlocal", "longgap")


def hbin(dr, n):
    ang = np.arctan2(dr[:, 1], dr[:, 0])
    return np.round(ang / (2 * np.pi / n)).astype(int) % n


# ---------------------------------------------------------------- extra fits
def fit_v2(train_zip, n_train, base, ridge=1.0):
    t0 = time.time()
    align = base["align"]
    sums = {"A32": np.zeros((6, 32, 3)), "E8": np.zeros((7, 6, 8, 3))}
    XX = {B: 0 for B, _ in CONFIGS}
    XY = {B: 0 for B, _ in CONFIGS}
    for ep in g1.iter_episodes(train_zip, n_train):
        X = g1.pix16(ep["image"])
        a = g1.actions_of(ep)
        pos, dr = ep["agent_pos"], ep["agent_dir"]
        d = np.diff(pos, axis=0)
        h8, h32 = g1.heading_bin(dr), hbin(dr, 32)
        h8u = h8[:-1] if align == "prev" else h8[1:]
        h32u = h32[:-1] if align == "prev" else h32[1:]
        at = a[1:]
        ok = at >= 0
        ap = np.where(a[:-1] >= 0, a[:-1], 6)
        np.add.at(sums["A32"], (at[ok], h32u[ok]), np.hstack([d[ok], np.ones((ok.sum(), 1))]))
        np.add.at(sums["E8"], (ap[ok], at[ok], h8u[ok]), np.hstack([d[ok], np.ones((ok.sum(), 1))]))
        # blind continuation: [x_{t-B-1}, x_{t-B-2}, counts of a_{t-B..t}, 1] -> x_t
        oh = np.zeros((len(a), 6)); oh[a >= 0, a[a >= 0]] = 1
        csum = np.vstack([np.zeros((1, 6)), np.cumsum(oh, axis=0)])
        for B in XX:
            ts = np.arange(B + 2, len(a))
            hist = csum[ts + 1] - csum[ts - B]
            F = np.hstack([X[ts - B - 1], X[ts - B - 2], hist, np.ones((len(ts), 1))]).astype(np.float64)
            XX[B] = XX[B] + F.T @ F
            XY[B] = XY[B] + F.T @ X[ts].astype(np.float64)
    tabs = {k: v[..., :2] / np.maximum(v[..., 2:], 1) for k, v in sums.items()}
    Bc = {B: np.linalg.solve(XX[B] + ridge * np.eye(XX[B].shape[0]), XY[B]).astype(np.float32) for B in XX}
    print(f"fit_v2: {time.time()-t0:.0f}s", flush=True)
    return dict(tabs=tabs, Bc=Bc)


def step_preds(a, dr, d, base, ext):
    """Predicted displacement per step class; dhat[t-1] is the step caused by a[t]."""
    align = base["align"]
    h8, h32 = g1.heading_bin(dr), hbin(dr, 32)
    h8u = h8[:-1] if align == "prev" else h8[1:]
    h32u = h32[:-1] if align == "prev" else h32[1:]
    at = a[1:]
    ok = at >= 0
    ap = np.where(a[:-1] >= 0, a[:-1], 6)
    Fh = np.hstack([dr[:-1], dr[1:]])
    out = {k: np.zeros_like(d) for k in CLASSES}
    out["A8"][ok] = base["tables"][align][at[ok], h8u[ok]]
    out["A32"][ok] = ext["tabs"]["A32"][at[ok], h32u[ok]]
    out["E8"][ok] = ext["tabs"]["E8"][ap[ok], at[ok], h8u[ok]]
    out["C"][ok] = np.einsum("ni,nij->nj", Fh[ok], base["MC"][at[ok]])
    out["E"][ok] = np.einsum("ni,nij->nj", Fh[ok], base["ME"][ap[ok], at[ok]])
    out["odo"] = d.copy()
    return {k: np.vstack([np.zeros((1, 2)), np.cumsum(v, axis=0)]) for k, v in out.items()}


def sighted_mask(T, B, S, ep_idx):
    off = int(np.random.default_rng([ep_idx, B, S]).integers(0, S + B))
    sm = ((np.arange(T) + off) % (S + B)) < S
    sm[0] = True
    return sm


# ---------------------------------------------------------------- per episode
def episode(ep, ep_idx, base, ext):
    X16 = g1.pix16(ep["image"])
    XP = (X16 - base["mu"]) @ base["W"]
    a = g1.actions_of(ep)
    pos, dr = ep["agent_pos"], ep["agent_dir"]
    N = ep["maze_layout"].shape[-1]
    T = len(a)
    cell = np.clip(np.floor(pos).astype(int), 0, N - 1)
    cid = cell[:, 0] * N + cell[:, 1]
    hb = g1.heading_bin(dr)
    pose = cid * g1.N_HB + hb
    d = np.diff(pos, axis=0)
    P = step_preds(a, dr, d, base, ext)
    oh = np.zeros((T, 6)); oh[a >= 0, a[a >= 0]] = 1
    csum = np.vstack([np.zeros((1, 6)), np.cumsum(oh, axis=0)])

    Ds = {}
    for sp, Xs in (("pix16", X16), ("pcaw64", XP)):
        D = g1.sqdist(Xs, Xs)
        np.fill_diagonal(D, 0.0)
        D4 = np.full_like(D, np.inf)
        D4[3:, 3:] = D[3:, 3:] + D[2:-1, 2:-1] + D[1:-2, 1:-2] + D[:-3, :-3]
        Ds[sp] = (Xs, D, D4)

    def score_idx(D, u, t):
        dv = D[u]
        other = pose != pose[t]
        return bool(pose[int(np.argmin(dv))] == pose[t]), bool(dv[t] < dv[other].min())

    def score_vec(Xs, pv, t):
        dv = ((Xs - pv) ** 2).sum(1)
        other = pose != pose[t]
        return bool(pose[int(np.argmin(dv))] == pose[t]), bool(dv[t] < dv[other].min())

    recs = {}
    for B, S in CONFIGS:
        sm = sighted_mask(T, B, S, ep_idx)
        rows = []
        n_blind_runs = 0
        for t in range(1, T):
            if not (sm[t] and not sm[t - 1]):
                continue
            b0 = t
            while b0 > 0 and not sm[b0 - 1]:
                b0 -= 1
            if t - b0 != B or b0 - 2 < 0:
                continue
            n_blind_runs += 1
            p = pose[t]
            if pose[b0 - 1] == p:
                continue
            prev = np.where((pose[:b0] == p) & sm[:b0])[0]
            if len(prev) == 0:
                continue
            s = int(prev[-1])
            r = g1.rle(list(cid[s:t + 1]))
            row = dict(t=t, gap=t - s, back=r == r[::-1])
            for k, Pk in P.items():
                row[f"err_full_{k}"] = float(np.linalg.norm((Pk[t] - Pk[s]) - (pos[t] - pos[s])))
                row[f"err_blind_{k}"] = float(np.linalg.norm((Pk[t] - Pk[b0 - 1]) - (pos[t] - pos[b0 - 1])))
            delta = t - b0 + 1
            for sp, (Xs, D, D4) in Ds.items():
                row[f"{sp}|F1"] = score_idx(D, b0 - 1, t)
                hist = csum[t + 1] - csum[b0]
                F = np.concatenate([X16[b0 - 1], X16[b0 - 2], hist, [1.0]]).astype(np.float32)
                p16 = F @ ext["Bc"][B]
                pv = p16 if sp == "pix16" else (p16 - base["mu"]) @ base["W"]
                row[f"{sp}|F2"] = score_vec(Xs, pv, t)
                for m, k in F3_VARIANTS:
                    Dm = D if m == 1 else D4
                    cand = np.arange(m - 1, b0 - k)  # s <= b0 - 1 - k
                    if len(cand):
                        okc = (cand + delta < b0)
                        cand = cand[okc]
                    if len(cand):
                        okc = sm[cand + delta] & sm[cand]
                        if m == 4:
                            okc &= sm[cand - 1] & sm[cand - 2] & sm[cand - 3]
                        cand = cand[okc]
                    if len(cand):
                        sb = int(cand[np.argmin(Dm[cand, b0 - 1])])
                        row[f"{sp}|F3_m{m}_k{k}"] = score_idx(D, sb + delta, t)
                    else:
                        row[f"{sp}|F3_m{m}_k{k}"] = score_idx(D, b0 - 1, t)
                row[f"{sp}|oracle"] = score_idx(D, s, t)
                cu = np.where(sm[:b0] & (hb[:b0] == hb[t]))[0]
                for kname, Pk in P.items():
                    if len(cu):
                        u = int(cu[np.argmin(((Pk[cu] - Pk[t]) ** 2).sum(1))])
                        row[f"{sp}|I_{kname}"] = score_idx(D, u, t)
                    else:
                        row[f"{sp}|I_{kname}"] = (False, False)
            rows.append(row)
        recs[f"B{B}_S{S}"] = dict(rows=rows, n_blind_runs=n_blind_runs, sighted_share=float(sm.mean()))
    return recs


# ---------------------------------------------------------------- aggregation
def stratum_mask(rows, st):
    gap = np.array([r["gap"] for r in rows]); bk = np.array([r["back"] for r in rows], bool)
    if st == "all":
        return np.ones(len(rows), bool)
    if st == "longgap":
        return gap >= g1.LONG_GAP
    return (gap >= g1.LONG_GAP) & ~bk


def aggregate(all_recs, rng):
    out = {}
    for cfg in all_recs[0]:
        eps = [r[cfg] for r in all_recs]
        c = {}
        ns = np.array([len(e["rows"]) for e in eps])
        nl = np.array([int(stratum_mask(e["rows"], "nonlocal").sum()) if e["rows"] else 0 for e in eps])
        c["sighted_share"] = float(np.mean([e["sighted_share"] for e in eps]))
        c["blind_runs_per_episode"] = float(np.mean([e["n_blind_runs"] for e in eps]))
        c["scored_per_episode"] = float(ns.mean())
        c["nonlocal_per_episode"] = float(nl.mean())
        c["episodes_with_0_nonlocal"] = float((nl == 0).mean())
        c["n_scored_total"] = int(ns.sum()); c["n_nonlocal_total"] = int(nl.sum())
        rows_all = [r for e in eps for r in e["rows"]]
        if not rows_all:
            out[cfg] = c
            continue
        gaps = np.array([r["gap"] for r in rows_all])
        c["gap_quantiles"] = {str(q): float(np.quantile(gaps, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        c["nonlocal_share_of_scored"] = float(nl.sum() / max(ns.sum(), 1))
        keys = [k for k in rows_all[0] if "|" in k]
        acc = {}
        for k in keys:
            for ri, rn in ((0, "R1"), (1, "R2")):
                row = {}
                for st in STRATA:
                    num, den = [], []
                    for e in eps:
                        if not e["rows"]:
                            num.append(0); den.append(0); continue
                        m = stratum_mask(e["rows"], st)
                        v = np.array([r[k][ri] for r in e["rows"]], bool)
                        num.append(int(v[m].sum())); den.append(int(m.sum()))
                    row[st] = g1.boot_ci(num, den, rng)
                acc[f"{k}|{rn}"] = row
        c["acc"] = acc
        dr = {}
        for kname in CLASSES + ["odo"]:
            dr[kname] = {}
            for which in ("full", "blind"):
                for st in STRATA:
                    m = stratum_mask(rows_all, st)
                    e = np.array([r[f"err_{which}_{kname}"] for r in rows_all])[m]
                    dr[kname][f"{which}_{st}"] = dict(
                        share_lt_0p5=float((e < 0.5).mean()) if len(e) else float("nan"),
                        share_lt_1=float((e < 1.0).mean()) if len(e) else float("nan"),
                        median=float(np.median(e)) if len(e) else float("nan"))
        c["deadreckon"] = dr
        out[cfg] = c
    return out


def stop_rule(agg):
    c = agg[f"B{PRIMARY[0]}_S{PRIMARY[1]}"]
    v = {}
    floors = ["F2"] + [f"F3_m{m}_k{k}" for m, k in F3_VARIANTS]
    best = max(((c["acc"][f"{sp}|{f}|R1"]["nonlocal"][0], sp, f) for sp in g1.SPACES for f in floors),
               key=lambda x: -1 if np.isnan(x[0]) else x[0])
    v["S2v2"] = dict(fires=bool(best[0] >= 0.85), best_floor_nonlocal_R1=best[0], space=best[1], floor=best[2],
                     ceiling_R1=1.0, Y=0.15)
    shares = {k: c["deadreckon"][k]["full_nonlocal"]["share_lt_0p5"] for k in CLASS_ORDER}
    passing = [k for k in CLASS_ORDER if shares[k] >= 0.5]
    v["S3v2"] = dict(fires=bool(not passing), full_gap_share_lt_0p5=shares, passing=passing,
                     primary_step_input=passing[0] if passing else None)
    v["S4v2"] = dict(fires=bool(c["nonlocal_per_episode"] < 3), nonlocal_per_episode=c["nonlocal_per_episode"],
                     minimum=3)
    v["PASS"] = bool(not (v["S2v2"]["fires"] or v["S3v2"]["fires"] or v["S4v2"]["fires"]))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-zip", required=True)
    ap.add_argument("--train-zip", required=True)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    meta = dict(eval_members=g1.n_members(args.eval_zip), train_members=g1.n_members(args.train_zip),
                numpy=np.__version__, python=sys.version.split()[0], configs=CONFIGS, primary=PRIMARY)
    print(meta, flush=True)
    base = g1.fit(args.train_zip, args.n_train, rng=rng)
    ext = fit_v2(args.train_zip, args.n_train, base)
    recs = []
    for i, ep in enumerate(g1.iter_episodes(args.eval_zip, args.n_eval)):
        recs.append(episode(ep, i, base, ext))
        if (i + 1) % 100 == 0:
            print(f"  {i+1} episodes, {time.time()-t0:.0f}s", flush=True)
    agg = aggregate(recs, rng)
    res = dict(meta=meta, table_align=base["align"], n_eval=len(recs), n_train_eps=base["n_train_eps"],
               configs=agg, stop_rule=stop_rule(agg), elapsed_s=time.time() - t0)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["stop_rule"], indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
