"""Registered readouts for RANK_NOWRAP_PREREG.md: is per-head rank 2's failure on the 2D torus about the exactly
periodic code that wrap-around revisits demand?

Cells (one batch, seeds SEEDS, recipe of run_rank_nd.sh; every arm's omega initialised at the grid-32 values, so at a
seed the initial weights are identical across grids -- train_rank_nowrap.py):
  A32 = per-head rank 2 on the 32-torus (wrap-only share of revisits 0.361)   -- RANK_ND's failing cell
  B32 = per-head rank 3 on the 32-torus                                      -- RANK_ND's solving cell
  AL  = per-head rank 2 on the 256-torus (wrap-only share 0.0000 in 1000 walks)
  BL  = per-head rank 3 on the 256-torus
Inputs: runs/rank_nowrap/N{32,256}/EVAL_D2.json (eval_nd via eval_rank_nowrap, held-out map env seed 10000, walk seed
10000 + s, 100 trajectories), the checkpoints' per-epoch losses (stats_core.classify_run: SOLVED = final-5% loss < 0.05)
and, for the floors, the SAME held-out walks regenerated here on the CPU.

A contrast "X over Y" FIRES if two-sided Fisher on SOLVED has p < .05 with X's rate higher, OR the exact/MC permutation
on accuracy (stats_core.perm2_p) has p < .05 with mean(X) - mean(Y) >= the materiality floor. It is REVERSED if
"Y over X" fires. Within a grid accuracy is raw (floor 0.02); across grids it is floor-relative,
(acc - retrace) / (1 - retrace) with the retrace floor of that seed's own eval stream (floor 0.10), because the
retrace floor is 0.750 on the 32-torus and 0.872 on the 256-torus. Branches: decide().
`python3 -m mapformer.analyze_rank_nowrap [--no-gpu-secondaries]`
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from mapformer import train_rank_nowrap  # noqa: F401  (registers the *_om32 arms in VARIANT_MAP)
from mapformer.environment_nd import GridWorldND
from mapformer.stats_core import classify_run, fisher_solved, perm2_p
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/rank_nowrap"
GS, GL = 32, 256
N_SEEDS = 12
SEEDS = list(range(60, 60 + N_SEEDS))
R2, R3 = "Vanilla_r2ph_om32", "Vanilla_r3ph_om32"
CELLS = {"A32": (GS, R2), "B32": (GS, R3), "AL": (GL, R2), "BL": (GL, R3)}
MIN_RAW, MIN_REL = 0.02, 0.10          # materiality floors: raw accuracy (within grid), floor-relative (across grids)
NEAR = 0.75                            # "solves about as well as rank 3": AL SOLVED >= ceil(0.75 n)
LOW = 0.25                             # "fails as on the 32-torus": AL SOLVED <= floor(0.25 n)
T_REG, N_TRIALS, ENV_SEED = 1024, 100, 10000


# ------------------------------------------------------------------------------------------------ streams and floors
def traj_info(env, tok, T):
    """Per step t: revisit, wrap-only (cell seen before only at another UNWRAPPED position: strat_acc's and
    nd_floor_wrap's definition), lag since the last visit to the wrapped cell, and the retrace predictor's token
    (docs/audits/2026-09-27/nd_floor_wrap.py, same logic)."""
    a = tok[0::2].numpy(); o = tok[1::2].numpy()
    U = np.cumsum(env.action_deltas[a], axis=0); P = [tuple(x) for x in env.visited_locations]
    blank = env.unified_blank
    wrap = np.zeros(T, bool); lag = np.zeros(T, np.int64); ret = np.full(T, blank, np.int64)
    last_cell, seen_unw = {}, set()
    prev_run_len = 0; cur_len = 0
    for t in range(T):
        if t > 0 and a[t] == a[t - 1]:
            cur_len += 1
        else:
            prev_run_len = cur_len if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
            cur_len = 1
        j = cur_len
        if prev_run_len > 0 and j <= prev_run_len and t - 2 * j >= 0 and (a[t - j] == (a[t] ^ 1) if t - j >= 0 else False):
            ret[t] = o[t - 2 * j]
        wrap[t] = tuple(U[t]) not in seen_unw
        lag[t] = t - last_cell.get(P[t], -10 ** 9)
        seen_unw.add(tuple(U[t])); last_cell[P[t]] = t
    return o, wrap, lag, ret


def stream(N, s, map_seed=ENV_SEED, n=N_TRIALS, T=T_REG):
    """eval_nd.evaluate's trajectories for run seed s (np.random.seed(env_seed + s), held-out map env_seed), with
    per-step info. map_seed = s gives the run's own TRAINING map instead (secondary)."""
    env = GridWorldND(dims=2, size=N, seed=map_seed); np.random.seed(ENV_SEED + s if map_seed == ENV_SEED else 10 ** 6 + s)
    out = []
    for _ in range(n):
        tok, _o, rev = env.generate_trajectory(T)
        out.append((tok, rev[1::2].numpy()) + traj_info(env, tok, T))
    return env, out


def floors(st, blank):
    tot = c_blank = c_ret = 0
    for tok, r, o, wrap, lag, ret in st:
        tot += int(r.sum()); c_blank += int((o[r] == blank).sum()); c_ret += int((ret[r] == o[r]).sum())
    return {"blank": c_blank / tot, "retrace": c_ret / tot, "n": tot}


# ------------------------------------------------------------------------------------------------ registered logic
def fires(sx, ax, sy, ay, n, min_d, pfun=perm2_p):
    """X over Y. sx, sy: SOLVED counts of n; ax, ay: per-seed accuracies (raw or floor-relative)."""
    pf = fisher_solved(sy, n, sx, n)
    d = float(np.mean(ax) - np.mean(ay)); pp = pfun(ay, ax)["p"]
    f_fisher = pf < 0.05 and sx > sy
    f_acc = pp < 0.05 and d >= min_d
    return {"fires": f_fisher or f_acc, "fisher": f_fisher, "acc": f_acc, "p_fisher": pf, "p_perm": pp, "d": d}


def decide(sol, n, raw, rel, pfun=perm2_p):
    """sol: SOLVED counts per cell; n: seeds per cell; raw / rel: per-seed T=1024 accuracy, raw and floor-relative.
    Returns (branch, qualifiers, contrasts)."""
    c = {"K": fires(sol["B32"], raw["B32"], sol["A32"], raw["A32"], n, MIN_RAW, pfun),       # control, grid 32
         "G": fires(sol["AL"], rel["AL"], sol["A32"], rel["A32"], n, MIN_REL, pfun),         # rank 2: large over 32
         "Grev": fires(sol["A32"], rel["A32"], sol["AL"], rel["AL"], n, MIN_REL, pfun),
         "R": fires(sol["BL"], raw["BL"], sol["AL"], raw["AL"], n, MIN_RAW, pfun),           # large grid: rank 3 over 2
         "Rrev": fires(sol["AL"], raw["AL"], sol["BL"], raw["BL"], n, MIN_RAW, pfun),
         "G3": fires(sol["BL"], rel["BL"], sol["B32"], rel["B32"], n, MIN_REL, pfun),        # secondary: rank 3 across grids
         "G3rev": fires(sol["B32"], rel["B32"], sol["BL"], rel["BL"], n, MIN_REL, pfun)}
    need = math.ceil(NEAR * n); low = math.floor(LOW * n); half = math.ceil(n / 2); q = []
    if all(min(raw[k]) >= 0.999 for k in CELLS):
        return "CEILING", q + ["every run of every cell >= 0.999: nothing can be read"], c
    if not c["K"]["fires"]:
        return "CONTROL FAILED (VOID)", q + [f"rank 3 not detectably above rank 2 on the 32-torus (B32 {sol['B32']}/{n} vs "
                                             f"A32 {sol['A32']}/{n}): no deficit to explain in this batch"], c
    if sol["BL"] < half or c["G3rev"]["fires"]:
        why = f"solves on {sol['BL']}/{n} < {half}" if sol["BL"] < half else "is detectably worse there than on the 32-torus"
        return "LARGE-GRID CONTROL FAILED", q + [f"rank 3 {why}: the large torus is not a working reference; reported "
                                                 f"as it falls"], c
    if c["Rrev"]["fires"]:
        q.append("REVERSAL: rank 2 detectably ABOVE rank 3 on the 256-torus")
    if c["Grev"]["fires"]:
        return "LARGE GRID HURTS RANK 2", q + [f"rank 2 detectably worse on the 256-torus than on the 32-torus; R "
                                               f"{'fires' if c['R']['fires'] else 'does not fire'}"], c
    if c["G"]["fires"] and not c["R"]["fires"] and sol["AL"] >= need:
        return "PERIODIC CODE IS THE LIMIT", q + [f"AL {sol['AL']}/{n} >= {need}; the rank-3 - rank-2 gap at 256 is "
                                                  f"unmeasured (does not fire), not shown to be zero"], c
    if c["G"]["fires"]:
        why = "rank 3 still detectably above rank 2 at 256" if c["R"]["fires"] else f"AL {sol['AL']}/{n} < {need}"
        return "PARTIAL", q + [f"rank 2 improves when wrap-around revisits vanish, but {why}"], c
    if c["R"]["fires"] and sol["AL"] <= low:
        return "RANK LIMIT IS GENERAL", q + [f"rank 2 does not detectably improve on the 256-torus (AL {sol['AL']}/{n} <= "
                                             f"{low}) and stays detectably below rank 3 there"], c
    if c["R"]["fires"]:
        return "INTERMEDIATE", q + [f"rank 2 below rank 3 at 256 but AL {sol['AL']}/{n} > {low} and not detectably above "
                                    f"its 32-torus self: neither the 32-torus failure rate nor rank 3's"], c
    return "UNMEASURED", q + ["neither G nor R fires: rank 2 on the 256-torus is distinguishable from neither its "
                              "32-torus self nor rank 3"], c


# ------------------------------------------------------------------------------------------------ secondaries (GPU)
def load(N, v, s, dev):
    b = torch.load(f"{R}/N{N}/D2/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False); c = b["config"]
    m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                       grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); return m.to(dev).eval(), b


@torch.no_grad()
def strat(m, st, dev):
    """Accuracy by stratum on a precomputed stream: all / wrap-only / plain lag<128 / plain lag>=128 / retrace-predictable
    / not retrace-predictable. 'all' reproduces eval_nd.evaluate on the same stream."""
    keys = ("all", "wrap", "plain<128", "plain>=128", "retrace_ok", "retrace_miss")
    ok = dict.fromkeys(keys, 0); tot = dict.fromkeys(keys, 0)
    for tok, r, o, wrap, lag, ret in st:
        pred = F.log_softmax(m(tok[None, :-1].to(dev)).float(), -1).argmax(-1)[0].cpu().numpy()
        for t in np.nonzero(r)[0]:
            hit = int(pred[2 * t] == o[t])
            ks = ["all", "wrap" if wrap[t] else ("plain<128" if lag[t] < 128 else "plain>=128"),
                  "retrace_ok" if ret[t] == o[t] else "retrace_miss"]
            for k in ks:
                ok[k] += hit; tot[k] += 1
    return {k: (ok[k] / tot[k] if tot[k] else None) for k in keys}, tot


@torch.no_grad()
def head_stats(m, c, weighted=True):
    """docs/theory/2026-10-04/scripts/basins.py heads(), unchanged in substance (D from the config; the per-head bottleneck
    returns the same (V, H, nb) angle table)."""
    V = c["vocab_size"]; D = c.get("n_dims", 2); K = c.get("n_obs_types", 16); pe = c.get("p_empty", 0.5); nA = 2 * D
    m = m.cpu()
    e = m.token_emb(torch.arange(V)); a = (m.action_to_lie(e[None])[0] * m.path_integrator.omega[None]).numpy()
    L = m.layers[0]; H = c["n_heads"]; dh = c["d_model"] // H; h = L.norm1(e)
    Q = L.q_proj(h).view(V, H, dh); Kk = L.k_proj(h).view(V, H, dh)
    qa = torch.sqrt(Q[..., 0::2] ** 2 + Q[..., 1::2] ** 2); ka = torch.sqrt(Kk[..., 0::2] ** 2 + Kk[..., 1::2] ** 2)
    w = (qa[:nA].mean(0) * ka[nA:].mean(0)).numpy() if weighted else np.ones_like(a[0])
    A = a[:nA]; O = a[nA:]
    U = np.stack([(A[2 * d] - A[2 * d + 1]) / 2 for d in range(D)])
    mm = np.angle(np.exp(1j * (A.mean(0) + pe * O[K] + (1 - pe) * O[:K].mean(0))))
    out = []
    for hh in range(H):
        sw = np.sqrt(w[hh] / w[hh].sum()); Uw = U[:, hh] * sw
        sv = np.linalg.svd(Uw, compute_uv=False)
        out.append((float(np.linalg.norm(mm[hh] * sw) / np.linalg.norm(Uw, axis=1).mean()), float(sv[-1] / sv[0])))
    return out


def basin(o):
    ok = [ind for k, ind in o if k <= 0.01]
    if not ok:
        return "CLOCK"
    return "CLEAN" if max(ok) >= 0.2 else "COLLAPSE"


def boot_did(x, n_boot=20000, seed=0):
    """(BL - AL) - (B32 - A32) on per-seed values, unpaired bootstrap within cells: mean and 95% percentile CI."""
    rng = np.random.default_rng(seed); arr = {k: np.asarray(v, float) for k, v in x.items()}
    f = lambda d: (d["BL"].mean() - d["AL"].mean()) - (d["B32"].mean() - d["A32"].mean())
    bs = [f({k: v[rng.integers(0, len(v), len(v))] for k, v in arr.items()}) for _ in range(n_boot)]
    return f(arr), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu-secondaries", action="store_true", help="registered readout and CPU secondaries only")
    ap.add_argument("--runs-dir", default=None, help="SMOKE TESTS ONLY (pilot dirs); the registered run uses R")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="SMOKE TESTS ONLY; the registered run uses SEEDS")
    ap.add_argument("--rescore-fmt", default=f"{REPO}/RANK_NOWRAP_RESCORE_N{{N}}.json")
    ap.add_argument("--json-out", default=f"{REPO}/RANK_NOWRAP.json")
    a = ap.parse_args()
    global R, SEEDS
    if a.runs_dir or a.seeds:
        R = a.runs_dir or R; SEEDS = a.seeds or SEEDS
        print(f"!! SMOKE TEST: runs-dir {R}, seeds {SEEDS} -- not the registered batch\n")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    n = len(SEEDS)
    J = {N: json.load(open(f"{R}/N{N}/EVAL_D2.json"))["D2"] for N in (GS, GL)}
    for N in (GS, GL):
        assert J[N]["grid"] == N, (N, J[N]["grid"])
    raw, raw2048, cls, sol, flo, rel = {}, {}, {}, {}, {}, {}
    streams = {}
    for k, (N, v) in CELLS.items():
        raw[k] = np.array([J[N]["acc"][v][str(T_REG)][str(s)] for s in SEEDS])
        raw2048[k] = np.array([J[N]["acc"][v]["2048"][str(s)] for s in SEEDS])
        cls[k] = [classify_run(torch.load(f"{R}/N{N}/D2/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)["losses"])
                  for s in SEEDS]
        sol[k] = sum(c["registered"] == "SOLVED" for c in cls[k])
    for N in (GS, GL):
        for s in SEEDS:
            env, st = stream(N, s); streams[(N, s)] = st; flo[(N, s)] = floors(st, env.unified_blank)
    for k, (N, v) in CELLS.items():
        f = np.array([flo[(N, s)]["retrace"] for s in SEEDS]); rel[k] = (raw[k] - f) / (1 - f)

    print(f"== cells: T={T_REG} held-out accuracy (raw | floor-relative) | SOLVED | [T=2048] | run classes ==")
    for N in (GS, GL):
        fr = [flo[(N, s)] for s in SEEDS]
        print(f"  grid {N}: eval-stream floors (mean over seeds): blank {np.mean([x['blank'] for x in fr]):.3f}, "
              f"retrace {np.mean([x['retrace'] for x in fr]):.3f}; scored targets per stream {np.mean([x['n'] for x in fr]):.0f}")
    for k, (N, v) in CELLS.items():
        cc = {c_: sum(c["cls"] == c_ for c in cls[k]) for c_ in ("SOLVED", "STALLED", "DESCENDING", "RISING")}
        print(f"  {k:4s} grid {N:3d} {v:18s} {raw[k].mean():.4f} +/- {raw[k].std(ddof=1):.4f} | rel {rel[k].mean():+.3f} | "
              f"SOLVED {sol[k]}/{n} | [{raw2048[k].mean():.4f}] | {cc}")
        print("        " + " ".join(f"s{s}:{x:.3f}/{c['registered'][:4]}({c['tail']:.3f})" for s, x, c in zip(SEEDS, raw[k], cls[k])))

    br, qual, c = decide(sol, n, raw, rel)
    print("\n== registered contrasts (X over Y; Fisher two-sided on SOLVED, permutation on accuracy) ==")
    lab = {"K": "K   control, 32-torus: B32 over A32 (raw)", "G": "G   rank 2: AL over A32 (floor-relative)",
           "Grev": "G'  rank 2 reversed: A32 over AL", "R": "R   256-torus: BL over AL (raw)",
           "Rrev": "R'  256-torus reversed: AL over BL", "G3": "    (secondary) rank 3: BL over B32 (floor-rel.)",
           "G3rev": "    (secondary) rank 3 reversed: B32 over BL"}
    for key, x in c.items():
        print(f"  {lab[key]:48s} Fisher p {x['p_fisher']:.4f}{' FIRES' if x['fisher'] else ''} | d {x['d']:+.4f} perm p "
              f"{x['p_perm']:.4f}{' FIRES' if x['acc'] else ''} | {'FIRES' if x['fires'] else 'does not fire'}")
    print(f"\n  REGISTERED: {br}" + "".join(f"\n    - {x}" for x in qual))

    print("\n== secondaries (no verdict) ==")
    for nm, vals in (("SOLVED fraction", {k: [float(cl["registered"] == "SOLVED") for cl in cls[k]] for k in CELLS}),
                     ("floor-relative accuracy", rel), ("raw accuracy", raw)):
        d, lo, hi = boot_did(vals)
        print(f"  interaction (BL - AL) - (B32 - A32), {nm}: {d:+.3f}  bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}] "
              f"(negative = the rank gap shrinks on the large torus)")
    tails = np.concatenate([[cl["tail"] for cl in cls[k]] for k in CELLS]); accs = np.concatenate([raw[k] for k in CELLS])
    print(f"  r(final-5% loss, acc@1024) over {len(tails)} runs: {np.corrcoef(tails, accs)[0, 1]:+.3f}; per grid: "
          + "  ".join(f"{N}: {np.corrcoef(np.concatenate([[cl['tail'] for cl in cls[k]] for k in CELLS if CELLS[k][0] == N]), np.concatenate([raw[k] for k in CELLS if CELLS[k][0] == N]))[0, 1]:+.3f}" for N in (GS, GL)))
    for k in CELLS:
        print(f"  T=2048 (rule 10: past training length) {k}: {raw2048[k].mean():.4f}")
    if a.no_gpu_secondaries:
        return
    out = {"solved": sol, "raw": {k: v.tolist() for k, v in raw.items()}, "rel": {k: v.tolist() for k, v in rel.items()},
           "branch": br, "strata": {}, "own": {}, "basins": {}}
    print("\n  accuracy by revisit stratum, held-out map, T=1024 (mean over seeds; target counts per stream in brackets):")
    for k, (N, v) in CELLS.items():
        rows = []
        for i, s in enumerate(SEEDS):
            m, b = load(N, v, s, dev)
            h, ht = strat(m, streams[(N, s)], dev)
            assert abs(h["all"] - raw[k][i]) < 1e-4, (k, s, h["all"], raw[k][i])   # same stream as eval_nd
            _, own_st = stream(N, s, map_seed=s, n=40)
            o, _ot = strat(m, own_st, dev)
            ws = head_stats(m, b["config"]); bs = basin(ws)
            sv = cls[k][i]["registered"] == "SOLVED"
            rows.append({"seed": s, **{f"h_{kk}": vv for kk, vv in h.items()}, **{f"n_{kk}": vv for kk, vv in ht.items()},
                         "own_all": o["all"], "basin": bs, "solved": sv, "flag": sv and h["all"] < 0.95})
            del m
        out["strata"][k] = rows
        mean = lambda f: np.mean([r[f] for r in rows if r[f] is not None]) if any(r[f] is not None for r in rows) else float("nan")
        print(f"    {k:4s} " + "  ".join(f"{kk} {mean('h_' + kk):.3f} [{mean('n_' + kk):.0f}]" for kk in
                                       ("wrap", "plain<128", "plain>=128", "retrace_ok", "retrace_miss")))
        print(f"         own training map {mean('own_all'):.3f} vs held-out {mean('h_all'):.3f} (gap {mean('own_all') - mean('h_all'):+.3f}); "
              f"SOLVED-but-held-out<0.95: {[r['seed'] for r in rows if r['flag']]}")
        cnt = {}
        for r in rows:
            cnt[(r["basin"], r["solved"])] = cnt.get((r["basin"], r["solved"]), 0) + 1
        conc = cnt.get(("CLEAN", True), 0) + sum(cnt.get((b_, False), 0) for b_ in ("COLLAPSE", "CLOCK"))
        print(f"         basins (descriptive): " + " ".join(f"{b_}/{'S' if v_ else 'U'} {cnt[(b_, v_)]}" for (b_, v_) in sorted(cnt))
              + f"; 'SOLVED iff CLEAN' on {conc}/{n}")
    for kk in ("wrap", "plain<128", "plain>=128", "retrace_miss"):
        x = {k: [r[f"h_{kk}"] for r in out["strata"][k]] for k in CELLS}
        for a_, b_ in (("A32", "B32"), ("AL", "BL"), ("A32", "AL")):
            xa = [v for v in x[a_] if v is not None]; xb = [v for v in x[b_] if v is not None]
            if len(xa) >= 2 and len(xb) >= 2:
                print(f"    stratum {kk:12s} {b_} - {a_}: {np.mean(xb) - np.mean(xa):+.3f} perm p {perm2_p(xa, xb)['p']:.4f} (n {len(xa)}/{len(xb)})")
    hm = {k: [r["h_retrace_miss"] for r in out["strata"][k] if r["h_retrace_miss"] is not None] for k in ("AL", "BL")}
    if len(hm["AL"]) >= 2 and len(hm["BL"]) >= 2:
        d_h = np.mean(hm["AL"]) - np.mean(hm["BL"]); p_h = perm2_p(hm["BL"], hm["AL"])["p"]
        trails = d_h <= -MIN_RAW and p_h < 0.05
        print(f"  hard-target qualifier (registered, attaches to PERIODIC CODE IS THE LIMIT): non-retrace targets on the 256-torus, "
              f"AL {np.mean(hm['AL']):.3f} vs BL {np.mean(hm['BL']):.3f}, AL - BL {d_h:+.3f} perm p {p_h:.4f} -> "
              + ("ON THE HARD TARGETS RANK 2 STILL TRAILS RANK 3" if trails else "no detectable shortfall")
              + (f"; AL below 0.95 there ({np.mean(hm['AL']):.3f})" if np.mean(hm["AL"]) < 0.95 else "")
              + ("" if br == "PERIODIC CODE IS THE LIMIT" else f" [branch is {br}: printed for the record]"))
        out["hard_qualifier"] = {"d": d_h, "p": p_h, "trails": bool(trails)}
    rs = a.rescore_fmt
    if all(os.path.exists(rs.format(N=N)) for N in (GS, GL)):
        JR = {N: json.load(open(rs.format(N=N)))["D2"] for N in (GS, GL)}
        rraw = {k: np.array([JR[N]["acc"][v][str(T_REG)][str(s)] for s in SEEDS]) for k, (N, v) in CELLS.items()}
        rrel = {k: (rraw[k] - np.array([flo[(CELLS[k][0], s)]["retrace"] for s in SEEDS])) /
                (1 - np.array([flo[(CELLS[k][0], s)]["retrace"] for s in SEEDS])) for k in CELLS}
        br2, _, c2 = decide(sol, n, rraw, rrel)
        flag = [key for key in c if c[key]["acc"] != c2[key]["acc"]]
        print(f"  dropout-scale re-score (attention x 1/(1-p)): " + "  ".join(f"{k} {rraw[k].mean():.4f}" for k in CELLS)
              + f"; branch under re-score: {br2}" + (f"; FLAG accuracy firing differs on {flag} (verdict unchanged)" if flag else ""))
    json.dump(out, open(a.json_out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
