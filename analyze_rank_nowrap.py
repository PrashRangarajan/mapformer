"""Registered readouts for RANK_NOWRAP_PREREG.md (+ Amendment 1): is per-head rank 2's failure on the 2D torus about
the exactly periodic code that wrap-around revisits demand, or about memorising a small fixed map?

Cells (one batch, seeds SEEDS, recipe of run_rank_nd.sh; every arm's omega initialised at the grid-32 values, so at a
seed the initial weights are identical across cells of the same rank -- train_rank_nowrap.py):
  A32 = per-head rank 2 on the 32-torus, fixed training map (wrap-only share of revisits 0.361) -- RANK_ND's failing cell
  B32 = per-head rank 3 on the 32-torus                                                       -- RANK_ND's solving cell
  M32 = per-head rank 2 on the 32-torus with the map REDRAWN every trajectory (environment_nd_redraw; same walks as A32,
        no map to memorise; Amendment 1)
  AL  = per-head rank 2 on the 256-torus (no wrap-only revisits in 2000 walks)
  BL  = per-head rank 3 on the 256-torus
Every cell is evaluated on the stock fixed held-out map (env seed 10000, walk seed 10000 + s, 100 walks, eval_nd via
eval_rank_nowrap). Inputs: runs/rank_nowrap/N{32,256}/EVAL_D2.json, the checkpoints, and the same held-out walks
regenerated here (floors and strata).

Per run: floor-relative accuracy rel = (acc - f) / (1 - f), f = the retrace-or-blank floor of that seed's eval stream
(predict the retraced observation inside a run that reverses the previous one, blank otherwise); rel is the share of
the floor-missed ("hard") targets recovered. A run HITS if rel >= HIT_REL (0.90). HIT replaces training-loss SOLVED in
every registered count (Amendment 1, D1: the 0.05 loss cut is ~2x more lenient at 256, where hard targets are 0.128 of
targets vs 0.25 at 32); loss-SOLVED is reported as a secondary.
Contrasts "X over Y":
  within a grid: FIRES if Fisher (two-sided) on HIT counts p < .05 with X higher, OR permutation (stats_core.perm2_p)
                 on raw accuracy p < .05 with mean(X) - mean(Y) >= 0.02;
  across grids:  FIRES only if the permutation on rel has p < .05 with mean(X) - mean(Y) >= 0.10.
Branches and qualifiers: decide(). `python3 -m mapformer.analyze_rank_nowrap [--no-gpu-secondaries]`
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from mapformer import train_rank_nowrap  # noqa: F401  (registers the *_om32 arms in VARIANT_MAP)
from mapformer.environment_nd import GridWorldND
from mapformer.stats_core import classify_run, fisher_solved, perm2_p, signflip_p
from mapformer.train_variant import VARIANT_MAP

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/rank_nowrap"
GS, GL = 32, 256
N_SEEDS = 10
SEEDS = list(range(60, 60 + N_SEEDS))
R2, R3, RD = "Vanilla_r2ph_om32", "Vanilla_r3ph_om32", "Vanilla_r2ph_om32_redraw"
CELLS = {"A32": (GS, R2), "B32": (GS, R3), "M32": (GS, RD), "AL": (GL, R2), "BL": (GL, R3)}
MIN_RAW, MIN_REL = 0.02, 0.10          # materiality floors: raw accuracy (within grid), floor-relative (across grids)
HIT_REL = 0.90                         # a run HITS if it recovers >= 90% of the floor-missed targets
NEAR = 0.75                            # "solves about as well as rank 3": AL HIT >= ceil(0.75 n)
LOW = 0.25                             # "fails as on the 32-torus": AL HIT <= floor(0.25 n)
T_REG, N_TRIALS, ENV_SEED = 1024, 100, 10000
RESCUE = ("PERIODIC CODE IS THE LIMIT", "MAP MEMORISATION WAS THE LIMIT", "LARGE TORUS RESCUES RANK 2, CAUSE UNRESOLVED")

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
def within(hx, ax, hy, ay, n, pfun=perm2_p):
    """X over Y on one grid. hx, hy: HIT counts of n; ax, ay: per-seed raw accuracies."""
    pf = fisher_solved(hy, n, hx, n)
    d = float(np.mean(ax) - np.mean(ay)); pp = pfun(ay, ax)["p"]
    f_fisher = pf < 0.05 and hx > hy
    f_acc = pp < 0.05 and d >= MIN_RAW
    return {"fires": f_fisher or f_acc, "fisher": f_fisher, "acc": f_acc, "p_fisher": pf, "p_perm": pp, "d": d}


def across(rx, ry, pfun=perm2_p):
    """X over Y across grids: floor-relative accuracy only (Amendment 1, D1)."""
    d = float(np.mean(rx) - np.mean(ry)); pp = pfun(ry, rx)["p"]
    f = pp < 0.05 and d >= MIN_REL
    return {"fires": f, "fisher": False, "acc": f, "p_fisher": float("nan"), "p_perm": pp, "d": d}


def hard_qualifier(hm, pfun=perm2_p):
    """Registered (Amendment 1, D4): on the 256-torus's hard targets (stratum retrace_miss: non-blank targets outside a
    retrace run) does rank 2 still trail rank 3? hm: {'AL': [...], 'BL': [...]} per-seed accuracies, or None."""
    if hm is None:
        return None, "hard-target qualifier: not computed"
    a, b = [x for x in hm["AL"] if x is not None], [x for x in hm["BL"] if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None, "hard-target qualifier: too few seeds with hard targets"
    d = float(np.mean(a) - np.mean(b)); p = pfun(b, a)["p"]
    trails = d <= -MIN_RAW and p < 0.05
    txt = (f"hard targets on the 256-torus: AL {np.mean(a):.3f} vs BL {np.mean(b):.3f}, AL - BL {d:+.3f} perm p {p:.4f} -> "
           + ("ON THE HARD TARGETS RANK 2 STILL TRAILS RANK 3" if trails else "no detectable shortfall"))
    return trails, txt


def decide(hit, n, raw, rel, hm=None, pfun=perm2_p):
    """The registered verdict: _decide(), with the hard-target qualifier (Amendment 1, D4) always on the qualifier list --
    attached to the verdict on the three rescue branches, '(record)' on the others."""
    br, q, c = _decide(hit, n, raw, rel, hm, pfun)
    _, htxt = hard_qualifier(hm, pfun)
    if not any(htxt in x for x in q):
        q = q + ["(record) " + htxt]
    return br, q, c


def _decide(hit, n, raw, rel, hm=None, pfun=perm2_p):
    """hit: HIT counts per cell; n: seeds per cell; raw / rel: per-seed T=1024 accuracy, raw and floor-relative;
    hm: per-seed hard-target accuracies of AL and BL (registered qualifier). Returns (branch, qualifiers, contrasts)."""
    c = {"K": within(hit["B32"], raw["B32"], hit["A32"], raw["A32"], n, pfun),      # control, 32-torus
         "G": across(rel["AL"], rel["A32"], pfun),                                   # rank 2: 256 over 32
         "Grev": across(rel["A32"], rel["AL"], pfun),
         "R": within(hit["BL"], raw["BL"], hit["AL"], raw["AL"], n, pfun),          # 256-torus: rank 3 over rank 2
         "Rrev": within(hit["AL"], raw["AL"], hit["BL"], raw["BL"], n, pfun),
         "G3rev": across(rel["B32"], rel["BL"], pfun),                               # rank 3 worse on the 256-torus
         "Mem": within(hit["M32"], raw["M32"], hit["A32"], raw["A32"], n, pfun),    # redrawn map helps rank 2 at 32
         "Km": within(hit["B32"], raw["B32"], hit["M32"], raw["M32"], n, pfun)}     # redrawn rank 2 still below rank 3
    need = math.ceil(NEAR * n); low = math.floor(LOW * n); half = math.ceil(n / 2); q = []
    mem_txt = (f"memorisation control M32 (rank 2, 32-torus, map redrawn): HIT {hit['M32']}/{n}; over A32 "
               f"{'FIRES' if c['Mem']['fires'] else 'does not fire'}; B32 over M32 {'FIRES' if c['Km']['fires'] else 'does not fire'}")
    trails, htxt = hard_qualifier(hm, pfun)
    if all(min(raw[k]) >= 0.999 for k in CELLS):
        return "CEILING", ["every run of every cell >= 0.999: nothing can be read"], c
    if not c["K"]["fires"]:
        return "CONTROL FAILED (VOID)", [f"rank 3 not detectably above rank 2 on the 32-torus (B32 HIT {hit['B32']}/{n} vs "
                                         f"A32 {hit['A32']}/{n}): no deficit to explain in this batch", mem_txt], c
    if hit["BL"] < half or c["G3rev"]["fires"]:
        why = f"hits on {hit['BL']}/{n} < {half}" if hit["BL"] < half else "is detectably worse there than on the 32-torus"
        return "LARGE-GRID CONTROL FAILED", [f"rank 3 {why}: the 256-torus is not a working reference; reported as it "
                                             f"falls", mem_txt], c
    if c["Rrev"]["fires"]:
        q.append("REVERSAL: rank 2 detectably ABOVE rank 3 on the 256-torus")
    if c["Grev"]["fires"]:
        return "LARGE GRID HURTS RANK 2", q + [f"R {'fires' if c['R']['fires'] else 'does not fire'}", mem_txt, "(record) " + htxt], c
    if c["G"]["fires"] and not c["R"]["fires"] and hit["AL"] >= need:
        base = [f"AL HIT {hit['AL']}/{n} >= {need}; the rank-3 - rank-2 gap at 256 is unmeasured (R does not fire), not "
                f"shown to be zero", mem_txt, htxt + (" [attaches to the verdict]" if trails else "")]
        if c["Km"]["fires"] and not c["Mem"]["fires"]:
            return RESCUE[0], q + base, c
        if c["Mem"]["fires"] and not c["Km"]["fires"]:
            return RESCUE[1], q + base + ["rank 2 also solves the 32-torus, wrap-around included, once the map cannot be "
                                          "memorised: the 32-torus failure was the fixed map, not the periodic code"], c
        return RESCUE[2], q + base + ["the redrawn-map control separates neither reading"], c
    if c["G"]["fires"]:
        why = "rank 3 still detectably above rank 2 at 256" if c["R"]["fires"] else f"AL HIT {hit['AL']}/{n} < {need}"
        return "PARTIAL", q + [f"rank 2 improves when wrap-around revisits vanish, but {why}", mem_txt, "(record) " + htxt], c
    if c["R"]["fires"]:
        if c["Mem"]["fires"] and not c["Km"]["fires"]:
            return "MEMORISATION AT 32, LARGE TORUS FAILS", q + [
                "rank 2 solves the 32-torus (wraps included) with a redrawn map, yet fails the 256-torus: neither the periodic "
                "code nor a general rank limit; the 256 failure has another cause (e.g. one third of the hard-target "
                "supervision)", mem_txt], c
        if hit["AL"] <= low and c["Km"]["fires"]:
            return "RANK LIMIT IS GENERAL", q + [f"rank 2 fails the 256-torus (AL HIT {hit['AL']}/{n} <= {low}), stays below "
                                                 f"rank 3 there, and fails the 32-torus even with a redrawn map", mem_txt], c
        return "INTERMEDIATE", q + [f"rank 2 below rank 3 at 256 (AL HIT {hit['AL']}/{n}) but neither at the 32-torus "
                                    f"failure rate with a failing redrawn-map control, nor rescued", mem_txt], c
    return "UNMEASURED", q + ["neither G nor R fires: rank 2 on the 256-torus is distinguishable from neither its 32-torus "
                              "self nor rank 3", mem_txt], c


# ------------------------------------------------------------------------------------------------ model loading, strata (registered), secondaries
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
    ap.add_argument("--no-gpu-secondaries", action="store_true",
                    help="skip own-map and basin secondaries (the strata, which carry the registered qualifier, always run)")
    ap.add_argument("--runs-dir", default=None, help="SMOKE TESTS ONLY (pilot dirs); the registered run uses R")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="SMOKE TESTS ONLY; the registered run uses SEEDS")
    ap.add_argument("--cells", nargs="+", default=None, help="SMOKE TESTS ONLY: subset of cells present")
    ap.add_argument("--rescore-fmt", default=f"{REPO}/RANK_NOWRAP_RESCORE_N{{N}}.json")
    ap.add_argument("--json-out", default=f"{REPO}/RANK_NOWRAP.json")
    a = ap.parse_args()
    global R, SEEDS, CELLS
    if a.runs_dir or a.seeds or a.cells:
        R = a.runs_dir or R; SEEDS = a.seeds or SEEDS
        if a.cells:
            CELLS = {k: v for k, v in CELLS.items() if k in a.cells}
        print(f"!! SMOKE TEST: runs-dir {R}, seeds {SEEDS}, cells {list(CELLS)} -- not the registered batch\n")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    n = len(SEEDS)
    J = {N: json.load(open(f"{R}/N{N}/EVAL_D2.json"))["D2"] for N in (GS, GL)}
    for N in (GS, GL):
        assert J[N]["grid"] == N, (N, J[N]["grid"])
    raw, raw2048, cls, sol, flo, rel, hit, streams = {}, {}, {}, {}, {}, {}, {}, {}
    for N in (GS, GL):
        for s in SEEDS:
            env, st = stream(N, s); streams[(N, s)] = st; flo[(N, s)] = floors(st, env.unified_blank)
    for k, (N, v) in CELLS.items():
        raw[k] = np.array([J[N]["acc"][v][str(T_REG)][str(s)] for s in SEEDS])
        raw2048[k] = np.array([J[N]["acc"][v]["2048"][str(s)] for s in SEEDS])
        cls[k] = [classify_run(torch.load(f"{R}/N{N}/D2/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)["losses"])
                  for s in SEEDS]
        sol[k] = sum(c["registered"] == "SOLVED" for c in cls[k])
        f = np.array([flo[(N, s)]["retrace"] for s in SEEDS]); rel[k] = (raw[k] - f) / (1 - f)
        hit[k] = int((rel[k] >= HIT_REL).sum())

    # strata (registered: the hard-target qualifier reads them). Same walks as eval_nd; on the GPU 'all' must reproduce
    # eval_nd's accuracy; on a CPU fallback argmax ties may differ, so the check is reported, not asserted (Amendment 1, N9)
    strata = {}
    for k, (N, v) in CELLS.items():
        rows = []
        for i, s in enumerate(SEEDS):
            m, b = load(N, v, s, dev)
            h, ht = strat(m, streams[(N, s)], dev)
            diff = abs(h["all"] - raw[k][i])
            if dev.startswith("cuda"):
                assert diff < 1e-4, (k, s, h["all"], raw[k][i])
            elif diff >= 1e-4:
                print(f"  WARN (CPU fallback): strata 'all' differs from eval_nd by {diff:.2e} for {k} s{s}")
            rows.append({"seed": s, "h": h, "n": ht, "model": (m, b)})
        strata[k] = rows
    hm = {k: [r["h"]["retrace_miss"] for r in strata[k]] for k in ("AL", "BL")} if {"AL", "BL"} <= set(CELLS) else None

    print(f"== cells: T={T_REG} held-out accuracy (raw | floor-relative) | HIT (rel >= {HIT_REL}) | loss-SOLVED | [T=2048] | classes ==")
    for N in (GS, GL):
        fr = [flo[(N, s)] for s in SEEDS]
        print(f"  grid {N}: eval-stream floors (mean over seeds): blank {np.mean([x['blank'] for x in fr]):.3f}, "
              f"retrace-or-blank {np.mean([x['retrace'] for x in fr]):.3f}; scored targets per stream {np.mean([x['n'] for x in fr]):.0f}")
    for k, (N, v) in CELLS.items():
        cc = {c_: sum(c["cls"] == c_ for c in cls[k]) for c_ in ("SOLVED", "STALLED", "DESCENDING", "RISING")}
        print(f"  {k:4s} grid {N:3d} {v:25s} {raw[k].mean():.4f} +/- {raw[k].std(ddof=1):.4f} | rel {rel[k].mean():+.3f} | "
              f"HIT {hit[k]}/{n} | SOLVED {sol[k]}/{n} | [{raw2048[k].mean():.4f}] | {cc}")
        print("        " + " ".join(f"s{s}:{x:.3f}/{r_:+.2f}/{c['registered'][:4]}({c['tail']:.3f})"
                                    for s, x, r_, c in zip(SEEDS, raw[k], rel[k], cls[k])))

    full = set(CELLS) == {"A32", "B32", "M32", "AL", "BL"}
    if full:
        br, qual, c = decide(hit, n, raw, rel, hm)
        lab = {"K": "K   control, 32-torus: B32 over A32", "G": "G   rank 2: AL over A32 (floor-relative only)",
               "Grev": "G'  A32 over AL (floor-relative only)", "R": "R   256-torus: BL over AL",
               "Rrev": "R'  256-torus: AL over BL", "G3rev": "G3' rank 3: B32 over BL (floor-relative only)",
               "Mem": "Mem 32-torus: M32 (redrawn) over A32", "Km": "Km  32-torus: B32 over M32 (redrawn)"}
        print("\n== registered contrasts (within grid: Fisher on HIT or permutation on raw accuracy; across: permutation on rel) ==")
        for key, x in c.items():
            print(f"  {lab[key]:46s} Fisher p {x['p_fisher']:.4f}{' FIRES' if x['fisher'] else ''} | d {x['d']:+.4f} perm p "
                  f"{x['p_perm']:.4f}{' FIRES' if x['acc'] else ''} | {'FIRES' if x['fires'] else 'does not fire'}")
        trails, _ = hard_qualifier(hm)
        tag = " [ON THE HARD TARGETS RANK 2 STILL TRAILS RANK 3]" if (trails and br in RESCUE) else ""
        print(f"\n  REGISTERED: {br}{tag}" + "".join(f"\n    - {x}" for x in qual))
    else:
        br = "n/a (smoke subset)"; print("\n  (smoke subset: decide() not run)")

    print("\n== secondaries (no verdict) ==")
    if full:
        sv = {k: sol[k] for k in CELLS}
        br_s, _, _ = decide(sv, n, raw, rel, None)
        print(f"  pre-amendment readout (training-loss SOLVED in place of HIT, no hard qualifier): {br_s}; loss-SOLVED "
              + " ".join(f"{k} {sol[k]}/{n}" for k in CELLS))
        print(f"  HIT vs loss-SOLVED agreement per run: {sum((r_ >= HIT_REL) == (c['registered'] == 'SOLVED') for k in CELLS for r_, c in zip(rel[k], cls[k]))}/{n * len(CELLS)}")
        for nm, vals in (("HIT fraction", {k: (rel[k] >= HIT_REL).astype(float) for k in CELLS}), ("floor-relative accuracy", rel),
                         ("raw accuracy", raw)):
            d, lo, hi = boot_did(vals)
            print(f"  interaction (BL - AL) - (B32 - A32), {nm}: {d:+.3f}  bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}] "
                  f"(negative = the rank gap shrinks on the large torus)")
        dp = rel["AL"] - rel["A32"]
        print(f"  paired G (N6): per seed rel(AL_s) - rel(A32_s) (identical init and action streams per seed): mean {dp.mean():+.3f}, "
              f"{int((dp > 0).sum())}/{n} positive, sign-flip p {signflip_p(dp)['p']:.4f}")
    ks = list(CELLS)
    tails = np.concatenate([[cl["tail"] for cl in cls[k]] for k in ks]); accs = np.concatenate([raw[k] for k in ks])
    print(f"  r(final-5% loss, acc@1024) over {len(tails)} runs: {np.corrcoef(tails, accs)[0, 1]:+.3f}; per grid: "
          + "  ".join(f"{N}: {np.corrcoef(np.concatenate([[cl['tail'] for cl in cls[k]] for k in ks if CELLS[k][0] == N]), np.concatenate([raw[k] for k in ks if CELLS[k][0] == N]))[0, 1]:+.3f}" for N in (GS, GL)))
    for k in ks:
        print(f"  T=2048 (rule 10: past training length) {k}: {raw2048[k].mean():.4f}")

    out = {"hit": hit, "solved": sol, "raw": {k: v.tolist() for k, v in raw.items()}, "rel": {k: v.tolist() for k, v in rel.items()},
           "branch": br, "strata": {}, "own": {}, "basins": {}}
    print("\n  accuracy by revisit stratum, held-out map, T=1024 (mean over seeds; target counts per stream in brackets).")
    print("  retrace_ok = targets the retrace-or-blank floor predicts (inside a retrace run, or blank); retrace_miss = non-blank")
    print("  targets outside a retrace run (the hard targets):")
    for k, (N, v) in CELLS.items():
        rows = strata[k]
        mean = lambda kk: (np.mean([r["h"][kk] for r in rows if r["h"][kk] is not None]) if any(r["h"][kk] is not None for r in rows) else float("nan"))
        cnt_ = lambda kk: np.mean([r["n"][kk] for r in rows])
        print(f"    {k:4s} " + "  ".join(f"{kk} {mean(kk):.3f} [{cnt_(kk):.0f}]" for kk in
                                       ("wrap", "plain<128", "plain>=128", "retrace_ok", "retrace_miss")))
        out["strata"][k] = [{"seed": r["seed"], **{f"h_{kk}": vv for kk, vv in r["h"].items()}, **{f"n_{kk}": vv for kk, vv in r["n"].items()}}
                            for r in rows]
        if a.no_gpu_secondaries:
            continue
        own, bas, cnt = [], [], {}
        for i, r in enumerate(rows):
            m, b = r["model"]
            if k != "M32":                                       # M32 has no training map of its own
                _, own_st = stream(N, r["seed"], map_seed=r["seed"], n=40)
                own.append(strat(m, own_st, dev)[0]["all"])
            bs = basin(head_stats(m, b["config"])); hv = bool(rel[k][i] >= HIT_REL); bas.append(bs)
            cnt[(bs, hv)] = cnt.get((bs, hv), 0) + 1
        out["own"][k] = own; out["basins"][k] = bas
        if own:
            print(f"         own training map {np.mean(own):.3f} vs held-out {raw[k].mean():.3f} (gap {np.mean(own) - raw[k].mean():+.3f}); "
                  f"HIT-but-raw<0.95: {[s for s, x, r_ in zip(SEEDS, raw[k], rel[k]) if r_ >= HIT_REL and x < 0.95]}")
        conc = cnt.get(("CLEAN", True), 0) + sum(cnt.get((b_, False), 0) for b_ in ("COLLAPSE", "CLOCK"))
        print(f"         basins (descriptive): " + " ".join(f"{b_}/{'H' if v_ else 'U'} {cnt[(b_, v_)]}" for (b_, v_) in sorted(cnt))
              + f"; 'HIT iff CLEAN' on {conc}/{n}")
    for kk in ("wrap", "plain<128", "plain>=128", "retrace_miss"):
        for a_, b_ in (("A32", "B32"), ("A32", "M32"), ("AL", "BL"), ("A32", "AL")):
            if a_ not in strata or b_ not in strata:
                continue
            xa = [r["h"][kk] for r in strata[a_] if r["h"][kk] is not None]; xb = [r["h"][kk] for r in strata[b_] if r["h"][kk] is not None]
            if len(xa) >= 2 and len(xb) >= 2:
                print(f"    stratum {kk:12s} {b_} - {a_}: {np.mean(xb) - np.mean(xa):+.3f} perm p {perm2_p(xa, xb)['p']:.4f} (n {len(xa)}/{len(xb)})")
    rs = a.rescore_fmt
    if full and all(os.path.exists(rs.format(N=N)) for N in (GS, GL)):
        JR = {N: json.load(open(rs.format(N=N)))["D2"] for N in (GS, GL)}
        rraw = {k: np.array([JR[N]["acc"][v][str(T_REG)][str(s)] for s in SEEDS]) for k, (N, v) in CELLS.items()}
        fl = {k: np.array([flo[(CELLS[k][0], s)]["retrace"] for s in SEEDS]) for k in CELLS}
        rrel = {k: (rraw[k] - fl[k]) / (1 - fl[k]) for k in CELLS}
        rhit = {k: int((rrel[k] >= HIT_REL).sum()) for k in CELLS}
        br2, _, c2 = decide(rhit, n, rraw, rrel, hm)
        flag = [key for key in c if c[key]["fires"] != c2[key]["fires"]]
        print(f"  dropout-scale re-score (attention x 1/(1-p)): " + "  ".join(f"{k} {rraw[k].mean():.4f} HIT {rhit[k]}" for k in CELLS)
              + f"; branch under re-score: {br2}" + (f"; FLAG firing differs on {flag} (verdict unchanged)" if flag else ""))
    json.dump(out, open(a.json_out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
