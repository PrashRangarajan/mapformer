"""Registered readouts for RANK_NOWRAP_PREREG.md (+ Amendments 1-4): is per-head rank 2's failure on the 2D torus
about the exactly periodic code that wrap-around revisits demand, or about the small fixed map of the 32-torus?

Cells (one batch, seeds SEEDS, recipe of run_rank_nd.sh; omega initialised at the grid-32 values in every cell, so at a
seed the initial weights are identical across cells of the same rank -- train_rank_nowrap.py):
  A32 = per-head rank 2 on the 32-torus, fixed training map (wrap-only share of revisits 0.361) -- RANK_ND's failing cell
  B32 = per-head rank 3 on the 32-torus                                                       -- RANK_ND's solving cell
  M32 = per-head rank 2 on the 32-torus, map REDRAWN every trajectory (environment_nd_redraw; same walks as A32)
  AL  = per-head rank 2 on the 256-torus (no wrap-only revisits in 2000 walks)
  BL  = per-head rank 3 on the 256-torus
Every cell is evaluated on the stock fixed held-out map (env seed 10000, walk seed 10000 + s, 100 walks): eval_nd via
eval_rank_nowrap for raw accuracy; the SAME walks are regenerated here for the strata.

THE REGISTERED QUANTITY (Amendment 3) is p = held-out accuracy on the PLAIN HARD targets: revisit targets that the
retrace-or-blank floor gets wrong (stratum retrace_miss: non-blank targets outside a retrace run; the floor predicts the
retraced observation inside a run that reverses the previous one and blank otherwise, and inside a retrace run it is
always right, checked) AND that are not wrap-only (the cell was seen before at the same unwrapped position). p is
kinematically matched across cells: on the 256-torus every hard target is plain (p = h); on the 32-torus the hard set is
69% wrap-only, and p keeps only its plain part, which matches the 256 hard set (lag median 80, ~36 vs ~39 per
sequence). So the
question p asks is: does training without wrap-around make rank 2 better on the SAME kind of targets? h (all hard,
Amendment 2) and wrap-only accuracy are declared secondaries. p and h count only non-blank labels (a lost model that
defaults to blank scores 0 there), on both grids alike. Amendment 1's rel was not grid-comparable either.
A run HITS if p >= HIT_P. Every contrast "X over Y":
  within a grid: FIRES if Fisher (two-sided) on HIT counts p < .05 with X higher, OR permutation (stats_core.perm2_p)
                 on p has p < .05 with mean(X) - mean(Y) >= MIN_P;
  across grids:  FIRES only if the permutation on p has p < .05 with mean(X) - mean(Y) >= MIN_P.
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
N_SEEDS = 10                           # THE seed-count switch (Amendment 4): the driver reads it from here; 10 or 12
SEEDS = list(range(60, 60 + N_SEEDS))
R2, R3, RD = "Vanilla_r2ph_om32", "Vanilla_r3ph_om32", "Vanilla_r2ph_om32_redraw"
CELLS = {"A32": (GS, R2), "B32": (GS, R3), "M32": (GS, RD), "AL": (GL, R2), "BL": (GL, R3)}
HIT_P = 0.98                           # a run HITS if p >= HIT_P (validated on stored runs, Amendment 3; narrow gap)
MIN_P = 0.10                           # materiality of every accuracy arm, in plain-hard-target units (grid-neutral)
MIN_FREE_MB = 3000                     # analysis device: a GPU needs this much free memory, else wait, else CPU
QUAL_D = 0.02                          # hard-target qualifier: AL - BL <= -0.02 with p < .05
NEAR = 0.75                            # "solves like rank 3": HIT >= ceil(0.75 n)
LOW = 0.25                             # "fails like rank 2 at 32": HIT <= floor(0.25 n)
T_REG, N_TRIALS, ENV_SEED = 1024, 100, 10000     # N_TRIALS: eval_nd's walks per seed (raw accuracy)
N_STRATA = 400                         # walks per seed for the strata / p (Amendment 4); the first N_TRIALS are eval_nd's
HIT_SENS = (0.97, 0.99)                # registered sensitivity FLAG: the branch recomputed at these HIT cuts
H_CUT_A2 = 0.90                        # Amendment 2's cut for h (secondary only)
RESCUE = ("PERIODIC CODE IS THE LIMIT", "FIXED MAP WAS THE LIMIT", "LARGE TORUS RESCUES RANK 2, CAUSE UNRESOLVED")
KEYS = ("all", "wrap", "plain<128", "plain>=128", "copy", "blank_out", "retrace_ok", "retrace_miss", "hard_wrap", "hard_plain")


# ------------------------------------------------------------------------------------------------ streams and floors
def deltas_of(env):
    """GridWorldND.action_deltas, or GridWorld's ACTION_DELTAS as an array (actions 2i / 2i+1 are opposite in both)."""
    if hasattr(env, "action_deltas"):
        return env.action_deltas
    return np.array([env.ACTION_DELTAS[i] for i in range(env.N_ACTIONS)], dtype=np.int64)


def traj_info(env, tok, T):
    """Per step t: observation, wrap-only (cell seen before only at another UNWRAPPED position: strat_acc's and
    nd_floor_wrap's definition), lag since the last visit to the wrapped cell, the retrace-or-blank floor's token
    (docs/audits/2026-09-27/nd_floor_wrap.py, same logic) and whether t is inside a retrace run."""
    a = tok[0::2].numpy() - getattr(env, "action_offset", 0); o = tok[1::2].numpy()
    U = np.cumsum(deltas_of(env)[a], axis=0); P = [tuple(x) for x in env.visited_locations]
    blank = env.unified_blank
    wrap = np.zeros(T, bool); lag = np.zeros(T, np.int64); ret = np.full(T, blank, np.int64); inr = np.zeros(T, bool)
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
            ret[t] = o[t - 2 * j]; inr[t] = True
        wrap[t] = tuple(U[t]) not in seen_unw
        lag[t] = t - last_cell.get(P[t], -10 ** 9)
        seen_unw.add(tuple(U[t])); last_cell[P[t]] = t
    return o, wrap, lag, ret, inr


def stream(N, s, map_seed=ENV_SEED, n=N_STRATA, T=T_REG):
    """eval_nd.evaluate's trajectories for run seed s (np.random.seed(env_seed + s), held-out map env_seed), with
    per-step info. map_seed = s gives the run's own TRAINING map instead (secondary)."""
    env = GridWorldND(dims=2, size=N, seed=map_seed); np.random.seed(ENV_SEED + s if map_seed == ENV_SEED else 10 ** 6 + s)
    out = []
    for _ in range(n):
        tok, _o, rev = env.generate_trajectory(T)
        out.append((tok, rev[1::2].numpy()) + traj_info(env, tok, T))
    return env, out


def floors(st, blank):
    tot = c_blank = c_ret = bad_copy = 0; share = dict.fromkeys(("copy", "blank_out", "retrace_miss"), 0)
    for tok, r, o, wrap, lag, ret, inr in st:
        rr = r.astype(bool)
        tot += int(rr.sum()); c_blank += int((o[rr] == blank).sum()); c_ret += int((ret[rr] == o[rr]).sum())
        bad_copy += int((inr & rr & (ret != o)).sum())
        share["copy"] += int((inr & rr & (ret == o)).sum()); share["blank_out"] += int((~inr & rr & (o == blank)).sum())
        share["retrace_miss"] += int((rr & (ret != o)).sum())
    return {"blank": c_blank / tot, "retrace": c_ret / tot, "n": tot, "bad_copy": bad_copy,
            "share": {k: v / tot for k, v in share.items()}}


# ------------------------------------------------------------------------------------------------ registered logic
def within(hx, ax, hy, ay, n, pfun=perm2_p):
    """X over Y on one grid. hx, hy: HIT counts of n; ax, ay: per-seed plain-hard accuracies p."""
    pf = fisher_solved(hy, n, hx, n)
    d = float(np.mean(ax) - np.mean(ay)); pp = pfun(ay, ax)["p"]
    f_fisher = pf < 0.05 and hx > hy
    f_acc = pp < 0.05 and d >= MIN_P
    return {"fires": f_fisher or f_acc, "fisher": f_fisher, "acc": f_acc, "p_fisher": pf, "p_perm": pp, "d": d}


def across(ax, ay, pfun=perm2_p):
    """X over Y across grids: permutation on p only (Amendments 1-3)."""
    d = float(np.mean(ax) - np.mean(ay)); pp = pfun(ay, ax)["p"]
    f = pp < 0.05 and d >= MIN_P
    return {"fires": f, "fisher": False, "acc": f, "p_fisher": float("nan"), "p_perm": pp, "d": d}


def hard_qualifier(h, pfun=perm2_p):
    """Registered: on the 256-torus's (plain) hard targets does rank 2 still trail rank 3 by >= QUAL_D (finer than R)?
    Takes the per-seed p of AL and BL (on the 256-torus p = h)."""
    a, b = np.asarray(h["AL"], float), np.asarray(h["BL"], float)
    d = float(a.mean() - b.mean()); p = pfun(b, a)["p"]
    trails = d <= -QUAL_D and p < 0.05
    return trails, (f"hard targets on the 256-torus: AL {a.mean():.3f} vs BL {b.mean():.3f}, AL - BL {d:+.3f} perm p {p:.4f} -> "
                    + ("ON THE HARD TARGETS RANK 2 STILL TRAILS RANK 3" if trails else "no detectable shortfall"))


def decide(hit, n, h, raw, pfun=perm2_p):
    """The registered verdict. hit: HIT counts per cell; h: per-seed REGISTERED accuracy per cell (p since Amendment 3,
    the plain-hard-target accuracy; the parameter keeps its name); raw: per-seed raw
    accuracy (CEILING only). Returns (branch, qualifiers, contrasts); the hard-target qualifier is always listed --
    attached to the verdict on the three rescue branches, '(record)' otherwise."""
    br, q, c = _decide(hit, n, h, raw, pfun)
    trails, htxt = hard_qualifier(h, pfun)
    q = q + [(htxt + " [attaches to the verdict]") if br in RESCUE else "(record) " + htxt]
    return br, q, c


def _decide(hit, n, h, raw, pfun=perm2_p):
    W = lambda x, y: within(hit[x], h[x], hit[y], h[y], n, pfun)
    A = lambda x, y: across(h[x], h[y], pfun)
    c = {"K": W("B32", "A32"),                      # control, 32-torus
         "G": A("AL", "A32"), "Grev": A("A32", "AL"),  # rank 2: 256 vs 32
         "R": W("BL", "AL"), "Rrev": W("AL", "BL"),    # 256-torus: rank 3 vs rank 2
         "G3rev": A("B32", "BL"),                      # rank 3 worse on the 256-torus
         "Mem": W("M32", "A32"), "MemRev": W("A32", "M32"),   # redrawn vs fixed map, rank 2, 32-torus
         "Km": W("B32", "M32")}                        # redrawn rank 2 below rank 3
    need = math.ceil(NEAR * n); low = math.floor(LOW * n); half = math.ceil(n / 2); q = []
    m32_fails = c["Km"]["fires"] and not c["Mem"]["fires"] and (hit["M32"] <= low or c["MemRev"]["fires"])
    m32_solves = c["Mem"]["fires"] and not c["Km"]["fires"] and hit["M32"] >= need
    mem_txt = (f"fixed-map control M32 (rank 2, 32-torus, map redrawn): HIT {hit['M32']}/{n}; M32 over A32 "
               f"{'FIRES' if c['Mem']['fires'] else 'does not fire'}; A32 over M32 {'FIRES' if c['MemRev']['fires'] else 'does not fire'}; "
               f"B32 over M32 {'FIRES' if c['Km']['fires'] else 'does not fire'} -> "
               + ("fails like rank 2" if m32_fails else "solves like rank 3" if m32_solves else "neither"))
    if c["MemRev"]["fires"]:
        q.append("redrawing the map made the 32-torus HARDER for rank 2 (A32 over M32 fires): the fixed map may be a "
                 "stepping stone (or the redraw adds map diversity / removes train-test overlap)")
    if all(min(raw[k]) >= 0.999 for k in CELLS):
        return "CEILING", ["every run of every cell >= 0.999: nothing can be read"], c
    if not c["K"]["fires"]:
        return "CONTROL FAILED (VOID)", [f"rank 3 not detectably above rank 2 on the 32-torus (B32 HIT {hit['B32']}/{n} vs "
                                         f"A32 {hit['A32']}/{n}): no deficit to explain in this batch", mem_txt] + q, c
    if hit["BL"] < half or c["G3rev"]["fires"]:
        why = f"hits on {hit['BL']}/{n} < {half}" if hit["BL"] < half else "is detectably worse there than on the 32-torus"
        return "LARGE-GRID CONTROL FAILED", [f"rank 3 {why}: the 256-torus is not a working reference; reported as it "
                                             f"falls", mem_txt] + q, c
    if c["Rrev"]["fires"]:
        q.append("REVERSAL: rank 2 detectably ABOVE rank 3 on the 256-torus")
    if c["Grev"]["fires"]:
        return "LARGE GRID HURTS RANK 2", q + [f"R {'fires' if c['R']['fires'] else 'does not fire'}", mem_txt], c
    if c["G"]["fires"] and not c["R"]["fires"] and hit["AL"] >= need:
        base = [f"AL HIT {hit['AL']}/{n} >= {need}; the rank-3 - rank-2 gap at 256 is unmeasured (R does not fire), not "
                f"shown to be zero", mem_txt]
        if m32_fails:
            return RESCUE[0], q + base + ["rank 2 also fails the 32-torus with nothing to memorise, and solves the torus "
                                          "without wrap-around"], c
        if m32_solves:
            return RESCUE[1], q + base + ["with a redrawn map rank 2 solves the 32-torus, wrap-around included: the 32-torus "
                                          "failure was the fixed map (memorisation, train/test shift, or map diversity), "
                                          "not the periodic code"], c
        return RESCUE[2], q + base + ["the redrawn-map control is neither a clear failure nor a clear success"], c
    if c["G"]["fires"]:
        why = "rank 3 still detectably above rank 2 at 256" if c["R"]["fires"] else f"AL HIT {hit['AL']}/{n} < {need}"
        return "PARTIAL", q + [f"rank 2 improves when wrap-around revisits vanish, but {why}", mem_txt], c
    if c["R"]["fires"]:
        if m32_solves:
            return "FIXED MAP AT 32, LARGE TORUS FAILS", q + [
                "rank 2 solves the 32-torus (wraps included) with a redrawn map, yet fails the 256-torus: neither the "
                "periodic code nor a general rank limit; the 32-torus failure was the fixed map (memorisation, train/test "
                "shift, or map diversity) and the 256 failure has another cause", mem_txt], c
        if hit["AL"] <= low and m32_fails:
            return "RANK LIMIT IS GENERAL", q + [f"rank 2 fails the 256-torus (AL HIT {hit['AL']}/{n} <= {low}), stays below "
                                                 f"rank 3 there, and fails the 32-torus with a redrawn map too", mem_txt], c
        return "INTERMEDIATE", q + [f"rank 2 below rank 3 at 256 (AL HIT {hit['AL']}/{n}) without the clear failure pattern "
                                    f"(AL HIT <= {low} and the redrawn-map control failing)", mem_txt], c
    return "UNMEASURED", q + ["neither G nor R fires: rank 2 on the 256-torus is distinguishable from neither its 32-torus "
                              "self nor rank 3", mem_txt], c


# ------------------------------------------------------------------------------------------------ models, strata
def pick_device(wait_s=1800, poll_s=30):
    """The CUDA device with the most free memory if it has >= MIN_FREE_MB (Amendment 3); polls up to wait_s for one;
    then falls back to the CPU (the strata check is then a WARN, Amendment 1 N9). Every decision is printed."""
    import time
    if not torch.cuda.is_available():
        print("pick_device: no CUDA -> cpu"); return "cpu"
    t0 = time.time()
    while True:
        try:
            free = [torch.cuda.mem_get_info(i)[0] / 2 ** 20 for i in range(torch.cuda.device_count())]
        except Exception as e:                                     # a wedged driver must not hang or kill the verdict
            print(f"pick_device: mem_get_info failed ({e}) -> cpu"); return "cpu"
        if free and max(free) >= MIN_FREE_MB:
            g = int(np.argmax(free)); print(f"pick_device: free MiB {[round(x) for x in free]} -> cuda:{g}"); return f"cuda:{g}"
        if time.time() - t0 > wait_s:
            print(f"pick_device: no GPU with >= {MIN_FREE_MB} MiB free after {wait_s} s (free {[round(x) for x in free]}) -> cpu")
            return "cpu"
        print(f"pick_device: free MiB {[round(x) for x in free]} < {MIN_FREE_MB}; waiting"); time.sleep(poll_s)


def load(N, v, s, dev):
    b = torch.load(f"{R}/N{N}/D2/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False); c = b["config"]
    m = VARIANT_MAP[v](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                       grid_size=c["grid_size"])
    m.load_state_dict(b["model_state_dict"]); return m.to(dev).eval(), b


@torch.no_grad()
def strat(m, st, dev, want_nll=False):
    """Accuracy (and optionally NLL) by stratum on a precomputed stream. Strata: all; wrap-only / plain lag < 128 /
    plain lag >= 128; copy (inside a retrace run, floor right), blank_out (blank target outside a retrace run),
    retrace_ok = copy + blank_out, retrace_miss = targets the retrace-or-blank floor gets wrong (the hard targets),
    split into hard_wrap (wrap-only) and hard_plain (the registered stratum, Amendment 3).
    'all' reproduces eval_nd.evaluate on the same stream."""
    ok = dict.fromkeys(KEYS, 0); tot = dict.fromkeys(KEYS, 0); nl = dict.fromkeys(KEYS, 0.0)
    for tok, r, o, wrap, lag, ret, inr in st:
        lp = F.log_softmax(m(tok[None, :-1].to(dev)).float(), -1)[0].cpu()
        pred = lp.argmax(-1).numpy()
        for t in np.nonzero(r)[0]:
            hit = int(pred[2 * t] == o[t]); nll = float(-lp[2 * t, int(o[t])]) if want_nll else 0.0
            miss = ret[t] != o[t]
            ks = ["all", "wrap" if wrap[t] else ("plain<128" if lag[t] < 128 else "plain>=128"),
                  "retrace_miss" if miss else "retrace_ok"]
            if not miss:
                ks.append("copy" if inr[t] else "blank_out")
            else:
                ks.append("hard_wrap" if wrap[t] else "hard_plain")
            for k in ks:
                ok[k] += hit; tot[k] += 1; nl[k] += nll
    acc = {k: (ok[k] / tot[k] if tot[k] else None) for k in KEYS}
    if want_nll:
        return acc, tot, {k: (nl[k] / tot[k] if tot[k] else None) for k in KEYS}
    return acc, tot


@torch.no_grad()
def head_stats(m, c, weighted=True):
    """docs/theory/2026-10-04/scripts/basins.py heads(), unchanged in substance (D from the config; the per-head bottleneck
    returns the same (V, H, nb) angle table)."""
    V = c["vocab_size"]; D = c.get("n_dims", 2); K = c.get("n_obs_types", 16); pe = c.get("p_empty", 0.5); nA = 2 * D
    m = m.cpu()
    e = m.token_emb(torch.arange(V)); a = (m.action_to_lie(e[None])[0] * m.path_integrator.omega[None]).numpy()
    L = m.layers[0]; H = c["n_heads"]; dh = c["d_model"] // H; hh_ = L.norm1(e)
    Q = L.q_proj(hh_).view(V, H, dh); Kk = L.k_proj(hh_).view(V, H, dh)
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


def all_strata(streams, dev, raw=None):
    """p (registered), HIT and strata for every cell and seed. On a GPU 'all' must reproduce eval_nd's accuracy (same walks); on a CPU
    fallback a difference (argmax ties) is printed, not asserted (Amendment 1, N9)."""
    out = {}
    for k, (N, v) in CELLS.items():
        rows = []
        for i, s in enumerate(SEEDS):
            m, b = load(N, v, s, dev)
            st = streams[(N, s)]
            a1, c1 = strat(m, st[:N_TRIALS], dev)                  # eval_nd's walks (checked against eval_nd)
            a2, c2 = strat(m, st[N_TRIALS:], dev)                  # the further walks (Amendment 4)
            cnt = {kk: c1[kk] + c2[kk] for kk in KEYS}
            acc = {kk: (((a1[kk] or 0) * c1[kk] + (a2[kk] or 0) * c2[kk]) / cnt[kk] if cnt[kk] else None) for kk in KEYS}
            if raw is not None:
                diff = abs(a1["all"] - raw[k][i])
                if dev.startswith("cuda"):
                    assert diff < 1e-4, (k, s, a1["all"], raw[k][i])
                elif diff >= 1e-4:
                    print(f"  WARN (CPU fallback): strata 'all' differs from eval_nd by {diff:.2e} for {k} s{s}")
            rows.append({"seed": s, "acc": acc, "n": cnt})
            del m
        out[k] = rows
    p = {k: np.array([r["acc"]["hard_plain"] for r in out[k]], float) for k in CELLS}
    return out, p, {k: int((p[k] >= HIT_P).sum()) for k in CELLS}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu-secondaries", action="store_true",
                    help="skip own-map and basin secondaries (the strata carry the registered quantity and always run)")
    ap.add_argument("--runs-dir", default=None, help="SMOKE TESTS ONLY (pilot dirs); the registered run uses R")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="SMOKE TESTS ONLY; the registered run uses SEEDS")
    ap.add_argument("--rescore-fmt", default=f"{REPO}/RANK_NOWRAP_RESCORE_N{{N}}.json")
    ap.add_argument("--json-out", default=f"{REPO}/RANK_NOWRAP.json")
    a = ap.parse_args()
    global R, SEEDS
    smoke = bool(a.runs_dir or a.seeds)
    if smoke:
        R = a.runs_dir or R; SEEDS = a.seeds or SEEDS
        print(f"!! SMOKE TEST: runs-dir {R}, seeds {SEEDS} -- not the registered batch\n")
    dev = pick_device(); print(f"device {dev}")
    n = len(SEEDS)
    J = {N: json.load(open(f"{R}/N{N}/EVAL_D2.json"))["D2"] for N in (GS, GL)}
    for N in (GS, GL):
        assert J[N]["grid"] == N, (N, J[N]["grid"])
    raw, raw2048, cls, sol, flo, streams = {}, {}, {}, {}, {}, {}
    for N in (GS, GL):
        for s in SEEDS:
            env, st = stream(N, s); streams[(N, s)] = st; flo[(N, s)] = floors(st, env.unified_blank)
            assert flo[(N, s)]["bad_copy"] == 0, ("retrace copy wrong", N, s)
    for k, (N, v) in CELLS.items():
        raw[k] = np.array([J[N]["acc"][v][str(T_REG)][str(s)] for s in SEEDS])
        raw2048[k] = np.array([J[N]["acc"][v]["2048"][str(s)] for s in SEEDS])
        cls[k] = []
        for s in SEEDS:
            b = torch.load(f"{R}/N{N}/D2/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)
            mr = b["config"].get("map_redrawn")                        # recorded by train_rank_nowrap (Amendment 2)
            if mr is None:
                assert smoke, f"{k} s{s}: checkpoint config lacks map_redrawn"
            else:
                assert mr == (k == "M32"), (k, s, mr)
            cls[k].append(classify_run(b["losses"]))
        sol[k] = sum(c["registered"] == "SOLVED" for c in cls[k])

    strata, p, hit = all_strata(streams, dev, raw)
    br, qual, c = decide(hit, n, p, raw)
    sec = lambda key: {k: np.array([r["acc"][key] if r["acc"][key] is not None else np.nan for r in strata[k]], float) for k in CELLS}
    h, wo = sec("retrace_miss"), sec("hard_wrap")

    print(f"== cells: T={T_REG} held-out | p = plain-hard accuracy (registered) | HIT (p >= {HIT_P}) | h = all-hard | "
          f"wrap-only hard | raw | loss-SOLVED | [raw T=2048] | classes ==")
    for N in (GS, GL):
        fr = [flo[(N, s)] for s in SEEDS]; sh = {kk: np.mean([x["share"][kk] for x in fr]) for kk in fr[0]["share"]}
        nst = {kk: np.mean([sum(r["n"][kk] for r in strata[k] if r["seed"] == s) / N_STRATA for k in CELLS if CELLS[k][0] == N
                            for s in SEEDS[:1]]) for kk in ("hard_plain", "hard_wrap")}
        print(f"  grid {N}: floors blank {np.mean([x['blank'] for x in fr]):.3f}, retrace-or-blank {np.mean([x['retrace'] for x in fr]):.3f}; "
              f"target shares copy {sh['copy']:.3f} / blank_out {sh['blank_out']:.3f} / hard {sh['retrace_miss']:.3f}; "
              f"hard targets per sequence: plain {nst['hard_plain']:.1f}, wrap-only {nst['hard_wrap']:.1f}")
    for k, (N, v) in CELLS.items():
        cc = {c_: sum(c["cls"] == c_ for c in cls[k]) for c_ in ("SOLVED", "STALLED", "DESCENDING", "RISING")}
        print(f"  {k:4s} grid {N:3d} {v:25s} p {p[k].mean():.4f} +/- {p[k].std(ddof=1):.4f} | HIT {hit[k]}/{n} | h {np.nanmean(h[k]):.4f} | "
              f"wrap-only {np.nanmean(wo[k]) if np.isfinite(wo[k]).any() else float('nan'):.4f} | raw {raw[k].mean():.4f} | "
              f"SOLVED {sol[k]}/{n} | [{raw2048[k].mean():.4f}] | {cc}")
        print("        " + " ".join(f"s{s}:{x:.3f}/{y:.3f}/{cl['registered'][:4]}({cl['tail']:.3f})"
                                    for s, x, y, cl in zip(SEEDS, p[k], raw[k], cls[k])))
    lab = {"K": "K    32-torus: B32 over A32", "G": "G    rank 2: AL over A32 (p only)", "Grev": "G'   A32 over AL (p only)",
           "R": "R    256-torus: BL over AL", "Rrev": "R'   256-torus: AL over BL", "G3rev": "G3'  rank 3: B32 over BL (p only)",
           "Mem": "Mem  32-torus: M32 (redrawn) over A32", "MemRev": "Mem' 32-torus: A32 over M32 (redrawn)",
           "Km": "Km   32-torus: B32 over M32 (redrawn)"}
    print("\n== registered contrasts (within grid: Fisher on HIT or permutation on p; across grids: permutation on p) ==")
    for key, x in c.items():
        print(f"  {lab[key]:42s} Fisher p {x['p_fisher']:.4f}{' FIRES' if x['fisher'] else ''} | d {x['d']:+.4f} perm p "
              f"{x['p_perm']:.4f}{' FIRES' if x['acc'] else ''} | {'FIRES' if x['fires'] else 'does not fire'}")
    trails, _ = hard_qualifier(p)
    tag = " [ON THE HARD TARGETS RANK 2 STILL TRAILS RANK 3]" if (trails and br in RESCUE) else ""
    sens = {cut: decide({k: int((p[k] >= cut).sum()) for k in CELLS}, n, p, raw)[0] for cut in HIT_SENS}
    stag = (" [HIT-cut sensitivity: " + ", ".join(f"{cut} -> {b_}" for cut, b_ in sens.items())
            + ("; FLAG: the branch depends on the cut (verdict unchanged)]" if any(b_ != br for b_ in sens.values()) else "; stable]"))
    print(f"\n  REGISTERED: {br}{tag}{stag}" + "".join(f"\n    - {x}" for x in qual))

    print("\n== secondaries (no verdict) ==")
    br_s, _, _ = decide(sol, n, p, raw)
    print(f"  amended branches with loss-SOLVED in place of HIT: {br_s}; loss-SOLVED " + " ".join(f"{k} {sol[k]}/{n}" for k in CELLS))
    hh = {k: np.nan_to_num(h[k]) for k in CELLS}; hhit = {k: int((hh[k] >= H_CUT_A2).sum()) for k in CELLS}
    print(f"  amended branches on h (all hard targets, Amendment 2's quantity and cut {H_CUT_A2}; not kinematically matched): "
          f"{decide(hhit, n, hh, raw)[0]}")
    print(f"  HIT vs loss-SOLVED agreement per run: " + "  ".join(
        f"{k} {sum((x >= HIT_P) == (cl['registered'] == 'SOLVED') for x, cl in zip(p[k], cls[k]))}/{n}" for k in CELLS))
    for a_, b_ in (("A32", "B32"), ("A32", "M32")):
        xa, xb = wo[a_][np.isfinite(wo[a_])], wo[b_][np.isfinite(wo[b_])]
        if len(xa) >= 2 and len(xb) >= 2:
            print(f"  wrap-only hard targets, 32-torus: {b_} - {a_} {xb.mean() - xa.mean():+.3f} perm p {perm2_p(xa, xb)['p']:.4f}; "
                  f"wrap deficit within cell (p - wrap-only): " + "  ".join(f"{k} {np.nanmean(p[k] - wo[k]):+.3f}" for k in (a_, b_)))
    for nm, vals in (("HIT fraction", {k: (p[k] >= HIT_P).astype(float) for k in CELLS}), ("p", p), ("raw accuracy", raw)):
        d, lo, hi = boot_did(vals)
        print(f"  interaction (BL - AL) - (B32 - A32), {nm}: {d:+.3f}  bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}] "
              f"(negative = the rank gap shrinks on the large torus)")
    for nm, x, y in (("paired G (N6): p(AL_s) - p(A32_s)", "AL", "A32"), ("paired Mem: p(M32_s) - p(A32_s)", "M32", "A32")):
        dp = p[x] - p[y]
        print(f"  {nm} (identical init and action streams per seed): mean {dp.mean():+.3f}, {int((dp > 0).sum())}/{n} "
              f"positive, sign-flip p {signflip_p(dp)['p']:.4f}")
    ks = list(CELLS)
    tails = np.concatenate([[cl["tail"] for cl in cls[k]] for k in ks])
    for nm, v_ in (("raw acc", raw), ("p", p)):
        accs = np.concatenate([v_[k] for k in ks])
        print(f"  r(final-5% loss, {nm}@1024) over {len(tails)} runs: {np.corrcoef(tails, accs)[0, 1]:+.3f}; per grid: "
              + "  ".join(f"{N}: {np.corrcoef(np.concatenate([[cl['tail'] for cl in cls[k]] for k in ks if CELLS[k][0] == N]), np.concatenate([v_[k] for k in ks if CELLS[k][0] == N]))[0, 1]:+.3f}" for N in (GS, GL)))
    print("  T=2048 raw (rule 10: past training length): " + "  ".join(f"{k} {raw2048[k].mean():.4f}" for k in ks))

    out = {"hit": hit, "solved": sol, "p": {k: v.tolist() for k, v in p.items()}, "h": {k: v.tolist() for k, v in h.items()},
           "wrap_only": {k: v.tolist() for k, v in wo.items()}, "raw": {k: v.tolist() for k, v in raw.items()},
           "branch": br, "qualifiers": qual, "strata": {k: [{"seed": r["seed"], **{f"acc_{kk}": vv for kk, vv in r["acc"].items()},
                                                             **{f"n_{kk}": vv for kk, vv in r["n"].items()}} for r in strata[k]] for k in ks},
           "own": {}, "basins": {}}
    print("\n  accuracy by revisit stratum, held-out map, T=1024 (mean over seeds; targets per stream in brackets; hard_wrap /")
    print("  hard_plain = the hard targets crossed with wrap-only / plain):")
    for k, (N, v) in CELLS.items():
        rows = strata[k]
        mean = lambda kk: (np.mean([r["acc"][kk] for r in rows if r["acc"][kk] is not None]) if any(r["acc"][kk] is not None for r in rows) else float("nan"))
        print(f"    {k:4s} " + "  ".join(f"{kk} {mean(kk):.3f} [{np.mean([r['n'][kk] for r in rows]):.0f}]" for kk in KEYS[1:]))
    for kk in ("wrap", "plain<128", "plain>=128", "copy", "blank_out", "hard_wrap", "hard_plain", "retrace_miss"):
        for a_, b_ in (("A32", "B32"), ("A32", "M32"), ("AL", "BL"), ("A32", "AL")):
            xa = [r["acc"][kk] for r in strata[a_] if r["acc"][kk] is not None]; xb = [r["acc"][kk] for r in strata[b_] if r["acc"][kk] is not None]
            if len(xa) >= 2 and len(xb) >= 2:
                print(f"    stratum {kk:12s} {b_} - {a_}: {np.mean(xb) - np.mean(xa):+.3f} perm p {perm2_p(xa, xb)['p']:.4f} (n {len(xa)}/{len(xb)})")
    if not a.no_gpu_secondaries:
        for k, (N, v) in CELLS.items():
            own, cnt = [], {}
            for i, s in enumerate(SEEDS):
                m, b = load(N, v, s, dev)
                if k != "M32":                                    # M32 has no training map of its own
                    _, own_st = stream(N, s, map_seed=s, n=40)
                    own.append(strat(m, own_st, dev)[0]["all"])
                bs = basin(head_stats(m, b["config"])); hv = bool(p[k][i] >= HIT_P)
                cnt[(bs, hv)] = cnt.get((bs, hv), 0) + 1; out["basins"].setdefault(k, []).append(bs)
            out["own"][k] = own
            conc = cnt.get(("CLEAN", True), 0) + sum(cnt.get((b_, False), 0) for b_ in ("COLLAPSE", "CLOCK"))
            print(f"    {k:4s} " + (f"own training map {np.mean(own):.3f} vs held-out {raw[k].mean():.3f} (gap {np.mean(own) - raw[k].mean():+.3f}); "
                                    if own else "no own map (redrawn); ")
                  + "basins " + " ".join(f"{b_}/{'H' if v_ else 'U'} {cnt[(b_, v_)]}" for (b_, v_) in sorted(cnt))
                  + f"; 'HIT iff CLEAN' on {conc}/{n}")

    # dropout-scale re-score (attention x 1/(1-p)): the registered branch recomputed with RE-SCORED p, HIT and hard-target
    # qualifier, plus eval_nd's re-scored raw accuracy (CEILING only) when present. Installed last: it patches nn.Module.eval.
    # Amendment 3: the number of hooked layers must equal (models loaded after install) x (layers per model), else FAIL.
    from mapformer import rescore_hook
    rescore_hook.install("auto")
    _, p_r, hit_r = all_strata(streams, dev)
    n_layers = {k: torch.load(f"{R}/N{CELLS[k][0]}/D2/{CELLS[k][1]}_s{SEEDS[0]}/{CELLS[k][1]}.pt", map_location="cpu",
                              weights_only=False)["config"]["n_layers"] for k in CELLS}
    want = sum(n_layers[k] * n for k in CELLS)
    if rescore_hook.STATS["hooked"] != want or rescore_hook.STATS["skipped"]:
        raise SystemExit(f"re-score FAILED: hooked {rescore_hook.STATS['hooked']} layers, expected {want}; "
                         f"skipped classes {sorted(rescore_hook.STATS['skipped'])}")
    raw_r = raw
    rs = a.rescore_fmt
    if all(os.path.exists(rs.format(N=N)) for N in (GS, GL)):
        JR = {N: json.load(open(rs.format(N=N)))["D2"] for N in (GS, GL)}
        raw_r = {k: np.array([JR[N]["acc"][v][str(T_REG)][str(s)] for s in SEEDS]) for k, (N, v) in CELLS.items()}
    br2, q2, c2 = decide(hit_r, n, p_r, raw_r)
    flag = [key for key in c if c[key]["fires"] != c2[key]["fires"]]
    print(f"  dropout-scale re-score (hooked {rescore_hook.STATS['hooked']} = {want} layers): "
          + "  ".join(f"{k} p {p_r[k].mean():.4f} HIT {hit_r[k]} raw {raw_r[k].mean():.4f}" for k in CELLS)
          + f"; branch under re-score: {br2} ({q2[-1]})" + (f"; FLAG firing differs on {flag} (verdict unchanged)" if flag else ""))
    out["hit_sensitivity"] = {str(k_): v_ for k_, v_ in sens.items()}
    out["rescore"] = {"branch": br2, "p": {k: v.tolist() for k, v in p_r.items()}, "hit": hit_r, "flag": flag}
    json.dump(out, open(a.json_out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
