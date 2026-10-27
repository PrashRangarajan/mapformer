"""Memory Maze CPU gate v3 (registration: MEMORY_MAZE_GATE_V3.md). Wraps mm_gate.py / mm_gate_v2.py unchanged.

Cell (B 16 blind, S 16 sighted); fits on train-part0 sorted indices 300-599, scores indices 600-1599 (fresh: v1/v2
fitted on 0-299 and scored only the eval archive). Checks S1v3-S4v3 with the one-SE conservative rule.
Usage: python3 -I mm_gate_v3.py --zip <15x15>/train-part0.zip --out gate_v3_15x15.json
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import mm_gate as g1  # noqa: E402
import mm_gate_v2 as g2  # noqa: E402

FIT = (300, 600)
SCORE = (600, 1600)
CELL = (16, 16)
KEYS = ("image", "action", "agent_pos", "agent_dir", "maze_layout")


def iter_range(path, lo, hi, keys=KEYS):
    z = zipfile.ZipFile(path)
    names = sorted(x for x in z.namelist() if x.endswith(".npz"))
    assert hi <= len(names), (hi, len(names))
    for i in range(lo, hi):
        ep = np.load(io.BytesIO(z.read(names[i])))
        out = {k: ep[k] for k in keys}
        out["name"] = names[i]
        out["idx"] = i
        yield out


def _fit_iter(path, n, keys=KEYS):  # replaces g1.iter_episodes for the fits (g1.fit and g2.fit_v2 call it)
    assert FIT[0] + n <= FIT[1], "fit set would overlap the score set"
    return iter_range(path, FIT[0], FIT[0] + n, keys)


def boot_se(num, den, rng, n=1000):
    num = np.asarray(num, float); den = np.asarray(den, float)
    idx = rng.integers(0, len(num), size=(n, len(num)))
    b = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return float(b.std())


def prop_check(num, den, rng, threshold, passes_above):
    p = float(np.sum(num) / max(np.sum(den), 1))
    n = float(np.sum(den))
    se = max(float(np.sqrt(p * (1 - p) / max(n, 1))), boot_se(num, den, rng))
    margin = (p - se - threshold) if passes_above else (threshold - (p + se))
    ok = (margin >= 0) if passes_above else (margin > 0)
    return dict(value=p, n=int(n), se=se, threshold=threshold, margin=margin, passes=bool(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="15x15 train-part0.zip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-zip", default=None,
                    help="POST HOC diagnostic only: score episodes 0-999 of this archive instead of train-part0 600-1599")
    ap.add_argument("--smoke", action="store_true", help="tiny test zip: fit on member 0, score member 1")
    args = ap.parse_args()
    global FIT, SCORE
    if args.smoke:
        FIT, SCORE = (0, 1), (1, 2)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    g1.iter_episodes = _fit_iter
    g2.CONFIGS = [CELL]
    g2.PRIMARY = CELL
    n_fit = FIT[1] - FIT[0]
    meta = dict(members=g1.n_members(args.zip), fit_idx=list(FIT), score_idx=list(SCORE), cell=list(CELL),
                numpy=np.__version__, python=sys.version.split()[0])
    print(meta, flush=True)
    base = g1.fit(args.zip, n_fit, rng=rng)
    ext = g2.fit_v2(args.zip, n_fit, base)
    recs1, recs2, idxs = [], [], []
    score_zip = args.zip
    if args.score_zip:
        score_zip, SCORE = args.score_zip, (0, 1000)
        meta.update(score_zip="OTHER ARCHIVE (post hoc diagnostic)", score_idx=list(SCORE))
    for ep in iter_range(score_zip, SCORE[0], SCORE[1]):
        recs2.append(g2.episode(ep, ep["idx"], base, ext))
        recs1.append(g1.episode(ep, base))
        idxs.append(ep["idx"])
        if len(recs2) % 100 == 0:
            print(f"  {len(recs2)} episodes, {time.time()-t0:.0f}s", flush=True)
    agg1 = g1.aggregate(recs1, base, rng)
    agg2 = g2.aggregate(recs2, rng)
    cfg = f"B{CELL[0]}_S{CELL[1]}"
    eps = [r[cfg]["rows"] for r in recs2]

    # S1v3 (all frames sighted, v1 definitions)
    oc = np.array([r["open_cells"] for r in recs1])
    s1a = prop_check((oc >= 128).astype(int), np.ones(len(oc)), rng, 0.5, True)
    nl1 = [int(((np.array(r["gaps"]) >= g1.LONG_GAP) & ~np.array(r["backtrack"], bool)).sum()) for r in recs1]
    ns1 = [r["n_scored"] for r in recs1]
    s1b = prop_check(nl1, ns1, rng, 0.10, True)

    def nl_mask(rows):
        return g2.stratum_mask(rows, "nonlocal") if rows else np.zeros(0, bool)

    # S2v3: best map-free floor under R1, non-local
    floors = ["F2"] + [f"F3_m{m}_k{k}" for m, k in g2.F3_VARIANTS]
    s2_all = {}
    for sp in g1.SPACES:
        for f in floors:
            key = f"{sp}|{f}"
            num = [int(np.array([r[key][0] for r in rows], bool)[nl_mask(rows)].sum()) if rows else 0 for rows in eps]
            den = [int(nl_mask(rows).sum()) for rows in eps]
            s2_all[key] = prop_check(num, den, rng, 0.85, False)
    best_key = max(s2_all, key=lambda k: s2_all[k]["value"])
    s2 = dict(best=best_key, **s2_all[best_key])

    # S3v3: full-gap dead reckoning within 0.5 cell, non-local
    s3_all = {}
    for c in g2.CLASS_ORDER:
        num = [int((np.array([r[f"err_full_{c}"] for r in rows])[nl_mask(rows)] < 0.5).sum()) if rows else 0
               for rows in eps]
        den = [int(nl_mask(rows).sum()) for rows in eps]
        s3_all[c] = prop_check(num, den, rng, 0.50, True)
    passing = [c for c in g2.CLASS_ORDER if s3_all[c]["passes"]]
    s3 = dict(classes=s3_all, passing=passing, primary_step_input=passing[0] if passing else None,
              passes=bool(passing))

    # S4v3: non-local blind re-entries per episode
    cnt = np.array([int(nl_mask(rows).sum()) for rows in eps], float)
    se4 = float(cnt.std(ddof=1) / np.sqrt(len(cnt)))
    s4 = dict(value=float(cnt.mean()), se=se4, threshold=3.0, margin=float(cnt.mean() - se4 - 3.0),
              passes=bool(cnt.mean() - se4 >= 3.0), episodes_with_0=float((cnt == 0).mean()))

    verdict = dict(S1v3=dict(a=s1a, b=s1b, passes=bool(s1a["passes"] and s1b["passes"])), S2v3=s2, S3v3=s3, S4v3=s4)
    verdict["PASS"] = bool(all(verdict[k]["passes"] for k in ("S1v3", "S2v3", "S3v3", "S4v3")))
    res = dict(meta=meta, score_episode_idx=[min(idxs), max(idxs)], n_score=len(idxs), n_fit_eps=base["n_train_eps"],
               table_align=base["align"], s2_all=s2_all, v1_style=agg1, v2_style=agg2, verdict=verdict,
               elapsed_s=time.time() - t0)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(verdict, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
