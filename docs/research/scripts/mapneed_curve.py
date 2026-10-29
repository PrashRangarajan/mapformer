"""CPU, post hoc, model-free. (A) A 'map-need' curve over task parameters: for directed torus walks (runs of 1..R
repeated moves, as environment.py's default with R = 10) on an N x N torus of length T, every revisit is labelled by
its gap in moves since the last visit to the cell, the number of direction runs it spans, whether it is reachable by the
retrace rule (score_rank_floors / analyze_factorial_n8.floors_walk logic), and whether it is wrap-only (the cell was
seen before only from another unwrapped position, as eval_rank_strata.kinds_vec defines it). From these shares the
accuracy a map-free reader of reach R* can attain is predicted: hit iff reachable within R* (retrace, moves or runs),
else the blank guess. (B) Every stored *_STRATA.json at T = 1024 is read and each arm is scored against the local-rule
model: pred = (n_short * acc_short + (n_long + n_wrap) * floor) / n_all, and a 'map index' = mean(acc_long, acc_wrap) -
floor. Written 2026-10-29 for docs/research/SCOPE_EXPANSION_2026-10-29.md. Not registered. Nothing in the repo is edited.
"""
import collections, glob, json, os, sys
import numpy as np

REPO = "/home/prashr/mapformer"
OUT = os.path.join(REPO, "docs/research/scripts/out")
DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
BLANK_P = 0.5                                  # p_empty of the default torus (blank observations)


def walk(N, T, R, rng, dims=2):
    """Directed walk: segments of a random move repeated k ~ U{1..R}. Returns moves (T,), run id per step (T,)."""
    acts, runs = [], []
    r = 0
    while len(acts) < T:
        a = rng.integers(0, 2 * dims); k = int(rng.integers(1, R + 1))
        acts += [a] * k; runs += [r] * k; r += 1
    return np.array(acts[:T]), np.array(runs[:T])


def label_revisits(N, T, R, rng, dims=2):
    a, run = walk(N, T, R, rng, dims)
    d = DELTA[a][:, :dims] if dims == 2 else np.where(a == 0, -1, 1)[:, None]
    u = np.cumsum(d, 0)                                               # unwrapped position after each move
    w = u % N
    seen_u, last_w = set(), {}
    out = []
    # retrace bookkeeping as floors_walk: cur = length of the current run so far, prev_run = length of the previous
    # run if it is the inverse move of the current one (pairs 0/1, 2/3), else 0
    prev_run = cur = 0
    for t in range(T):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
            cur = 1
        ku, kw = tuple(u[t]), tuple(w[t])
        if kw in last_w:
            s = last_w[kw]
            gap = t - s
            runs_spanned = run[t] - run[s] + 1
            retrace_ok = prev_run > 0 and cur <= prev_run and t - 2 * cur >= 0
            out.append(dict(gap=gap, runs=runs_spanned, retrace=retrace_ok, wrap=(ku not in seen_u)))
        seen_u.add(ku); last_w[kw] = t
    return out


def summarise(revs):
    n = len(revs)
    if n == 0:
        return None
    gap = np.array([r["gap"] for r in revs]); runs = np.array([r["runs"] for r in revs])
    ret = np.array([r["retrace"] for r in revs]); wrap = np.array([r["wrap"] for r in revs])
    s = dict(n=n, retrace_share=ret.mean(), wrap_share=wrap.mean(), gap_median=float(np.median(gap)),
             gap_le20=float((gap <= 20).mean()), gap_le64=float((gap <= 64).mean()), gap_le128=float((gap < 128).mean()),
             runs_le2=float((runs <= 2).mean()), runs_le4=float((runs <= 4).mean()), runs_le8=float((runs <= 8).mean()),
             runs_ge17=float((runs >= 17).mean()))
    # predicted accuracy of a map-free reader: hit on reachable revisits, blank guess elsewhere
    for k, m in (("retrace", ret), ("gap_le64", gap <= 64), ("gap_le128", gap < 128), ("runs_le4", runs <= 4)):
        s[f"pred_{k}"] = float(m.mean() + (1 - m.mean()) * BLANK_P)
    return s


def part_a():
    rows = []
    rng = np.random.default_rng(0)
    configs = [(N, T, R) for N in (8, 16, 32, 64, 128) for T in (128, 512, 1024, 4096) for R in (10,)]
    configs += [(64, 1024, R) for R in (1, 2, 5, 20, 40)]
    configs += [("ring32", 128, 10), ("ring32", 1024, 10), ("ring256", 1024, 10)]
    for N, T, R in configs:
        dims = 1 if isinstance(N, str) else 2
        n = int(N[4:]) if dims == 1 else N
        revs = []
        for _ in range(40):
            revs += label_revisits(n, T, R, rng, dims)
        s = summarise(revs)
        s.update(N=N, T=T, R=R, revisits_per_walk=len(revs) / 40)
        rows.append(s)
    with open(os.path.join(OUT, "mapneed_curve.json"), "w") as f:
        json.dump(rows, f, indent=1)
    hdr = ("N T R | rev/walk | retrace wrap | gap<=20 <=64 <128 | runs<=2 <=4 <=8 >=17 | pred: retrace gap64 gap128 runs4")
    print("PART A: map-need curve (40 walks per cell; shares of revisits)")
    print(hdr)
    for s in rows:
        print(f"{s['N']!s:>7} {s['T']:5d} {s['R']:2d} | {s['revisits_per_walk']:7.1f} | {s['retrace_share']:.3f} "
              f"{s['wrap_share']:.3f} | {s['gap_le20']:.3f} {s['gap_le64']:.3f} {s['gap_le128']:.3f} | {s['runs_le2']:.3f} "
              f"{s['runs_le4']:.3f} {s['runs_le8']:.3f} {s['runs_ge17']:.3f} | {s['pred_retrace']:.3f} {s['pred_gap_le64']:.3f} "
              f"{s['pred_gap_le128']:.3f} {s['pred_runs_le4']:.3f}")


def part_b():
    print("\nPART B: every stored *_STRATA.json at T = 1024, arm means over seeds")
    print("file | arm | n | acc all | short long wrap | floor | local-rule pred | acc - pred | map index | local index")
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, "*_STRATA.json"))):
        d = json.load(open(f))
        by_arm = collections.defaultdict(list)
        for k, v in d.items():
            parts = k.split("|")
            if len(parts) != 3 or parts[2] != "1024" or "all" not in v:
                continue
            by_arm[parts[0]].append(v)
        for arm, vs in sorted(by_arm.items()):
            def m(st, key):
                return float(np.mean([v[st][key] for v in vs]))
            n_all, n_s, n_l, n_w = (m("all", "n"), m("plain_lag<128", "n"), m("plain_lag>=128", "n"), m("wrap", "n"))
            acc, a_s, a_l, a_w = m("all", "acc"), m("plain_lag<128", "acc"), m("plain_lag>=128", "acc"), m("wrap", "acc")
            fl = m("all", "floor"); fl_l = m("plain_lag>=128", "floor"); fl_w = m("wrap", "floor")
            pred = (n_s * a_s + n_l * fl_l + n_w * fl_w) / n_all
            map_idx = 0.5 * ((a_l - fl_l) + (a_w - fl_w))
            local_idx = a_s - 0.911                                    # retrace on short gaps, torus T = 1024 (registered floors)
            r = dict(file=os.path.basename(f), arm=arm, n=len(vs), acc=acc, short=a_s, long=a_l, wrap=a_w, floor=fl,
                     pred=pred, resid=acc - pred, map_index=map_idx, local_index=local_idx)
            rows.append(r)
            print(f"{r['file']:36s} | {arm:16s} | {len(vs)} | {acc:.3f} | {a_s:.3f} {a_l:.3f} {a_w:.3f} | {fl:.3f} | "
                  f"{pred:.3f} | {acc - pred:+.3f} | {map_idx:+.3f} | {local_idx:+.3f}")
    with open(os.path.join(OUT, "strata_crossarm.json"), "w") as f:
        json.dump(rows, f, indent=1)
    # fit check: across arms, is acc - pred explained by the map index?  (resid should equal share_beyond_short * map_idx)
    a = np.array([[r["resid"], r["map_index"]] for r in rows])
    c = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    print(f"\nacross {len(rows)} arm means: corr(acc - local-rule pred, map index) = {c:.3f}; "
          f"arms with map index < 0.05: {sum(r['map_index'] < 0.05 for r in rows)}, in 0.05-0.4: "
          f"{sum(0.05 <= r['map_index'] < 0.4 for r in rows)}, >= 0.4: {sum(r['map_index'] >= 0.4 for r in rows)}")
    print("max |acc - pred| among arms with map index < 0.05: "
          f"{max(abs(r['resid']) for r in rows if r['map_index'] < 0.05):.3f}")


if __name__ == "__main__":
    part_a(); part_b()
