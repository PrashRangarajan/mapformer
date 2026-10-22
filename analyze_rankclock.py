"""Registered analysis for RANKCLOCK_PREREG.md (G-1, the rank clock-route batch, torus T = 1024), as amended
(Amendment 1: the independent results-blind code check).

  python3 -m mapformer.analyze_rankclock               # reads runs/rankclock{,_wd0}/p0 + the two eval JSONs
  python3 -m mapformer.analyze_rankclock --selftest    # every label from planted checkpoints AND snapshots on disk

Shared primitives (rule 5): stats_core.classify_run (SOLVED = final-5% training loss < 0.05), fisher_solved, perm2_p.
Arms (n = 8, seeds 1900-1907): r2 = Vanilla, slotxy = Vanilla_r2clockxy (one slot per move axis), slot = Vanilla_r2clock
(one per-token slot), DF = Vanilla_r2DF, AO = Vanilla_r2AO, r4 = Vanilla_r4mi, r4wd0 = Vanilla_r4mi at weight decay 0.

G0  r4 SOLVED >= 6/8, else UNREADABLE.
G1  r2 SOLVED <= 2/8, else RANK-2 DIFFICULTY NOT REPLICATED; P, P2 and E2 are not read.
P   (primary: the per-axis slot, which has both of rank 4's clock exits while the map stays rank 2)
      k >= 6 and Fisher(slotxy, r2) < .05 -> CLOCK EXITS HELP; k <= 2 -> CLOCK EXITS DO NOT HELP; else PARTIAL.
P2  (secondary: the single per-token slot, the common-mode exit only; the plan's original arm)
      same thresholds -> A COMMON-MODE EXIT HELPS / A COMMON-MODE EXIT DOES NOT HELP / PARTIAL.
E2  (G1+; RANK_TESTS' e2 rule with Amendment 1 finding 4's gaps closed), first match:
      k_AO >= 6 and k_DF <= 2                          -> AO SOLVES, DF DOES NOT
      k_AO >= 6 and Fisher(AO, r2) < .05               -> OBSERVATION STEPS ARE THE OBSTACLE
      k_AO <= 2 and k_DF >= 6 and Fisher(DF, AO) < .05 -> THE CLOCK IS THE OBSTACLE
      k_DF <= 2 and k_AO <= 2                          -> NEITHER (a second obstacle at rank 2)
      otherwise                                        -> PARTIAL
X2  (r4 at weight decay 0; Amendment 1 finding 2: the question is whether the final shrinkage of the clock needs decay)
    Per inverse pair (p, q), latents z = W_in e: u = z_p + z_q, delta = (z_p - z_q) / 2; angles WRAPPED to (-pi, pi]
    (rule 8): read clock = |wrap(omega * W_out u)|, map read = |wrap(omega * W_out delta)|, readgain = read / map; all
    averaged over the two pairs. Relative to the run's own initialisation (snapshot e0000):
      shrink = |u_end| / |u_init|,  rgrel = readgain_end / readgain_init,  u_ratio_end = |u| / (|z_p| + |z_q|).
    Over SOLVED r4wd0 runs (>= 4, else X2 UNREADABLE), medians, first match:
      shrink <= 0.2                     -> REMOVAL (the gradient removes the clock without decay)
      shrink >= 0.5 and rgrel <= 0.1    -> PARKING (without decay a large clock is kept, unread)
      u_ratio_end <= 0.15               -> DILUTED (the ratio is small because the map grew, not because u shrank)
      otherwise                         -> UNRESOLVED
X2b (registered secondary, RANK_THEORY P4 from the snapshots; r4wd0 and r4 separately, solved runs): t_read = first snapshot
    epoch with read clock <= 0.5 x its init value, t_lat = first epoch with |u| <= 0.5 x init. Runs with both defined and
    t_read != t_lat enter a two-sided sign test: READ CLOCK FALLS FIRST / LATENT CLOCK FALLS FIRST (p < .05), else NO ORDER.
    Parking-then-removal (P4) predicts READ CLOCK FALLS FIRST.

Accuracy (descriptive + permutation p beside each Fisher): eval_noise_refine on fresh maps at T = 512 / 1024 / 2048.
Also printed: run classes, geometry medians per arm, slot sizes, r(log10 final loss, acc) per length (rule 2).
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from scipy.stats import binomtest

from mapformer.model_rankclock import TORUS_PAIRS
from mapformer.stats_core import classify_run, fisher_solved, perm2_p

REPO = "/home/prashr/mapformer"
ROOT, ROOT_WD0 = f"{REPO}/runs/rankclock", f"{REPO}/runs/rankclock_wd0"
EVAL, EVAL_WD0 = f"{REPO}/RANKCLOCK.json", f"{REPO}/RANKCLOCK_WD0.json"
SEEDS = list(range(1900, 1908))
EPOCHS = 900
ARMS = {"r2": "Vanilla", "slotxy": "Vanilla_r2clockxy", "slot": "Vanilla_r2clock", "DF": "Vanilla_r2DF",
        "AO": "Vanilla_r2AO", "r4": "Vanilla_r4mi", "r4wd0": "Vanilla_r4mi"}
WD = {k: 0.05 for k in ARMS}
WD["r4wd0"] = 0.0
LENGTHS = (512, 1024, 2048)
CODE = {"SOLVED": "S", "STALLED": "T", "DESCENDING": "D", "RISING": "R", "DIVERGED": "X"}
RECIPE = {"n_steps": 1024, "batch_size": 16, "n_batches": 98, "lr": 1e-3, "schedule": "cosine", "n_layers": 1,
          "n_heads": 2, "d_model": 128, "n_landmarks": 0, "data_workers": 3, "grid_size": 64, "n_obs_types": 16,
          "env": "torus", "p_action_noise": 0.0, "p_transition_noise": 0.0, "init_from": None, "fast_attn": False,
          "snapshot_every": 10}


def ckpt_dir(arm, seed, root=None, root_wd0=None):
    r = (root_wd0 or ROOT_WD0) if arm == "r4wd0" else (root or ROOT)
    return os.path.join(r, "p0", f"{ARMS[arm]}_s{seed}")


# --------------------------------------------------------------------------------------------- geometry
def _wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.no_grad()
def geometry_sd(sd, tied=False):
    """Per-pair clock geometry from a state dict (a checkpoint or a step-map snapshot)."""
    E = sd["token_emb.weight"].double()
    Win, Wout = sd["action_to_lie.w_in.weight"].double(), sd["action_to_lie.w_out.weight"].double()
    om = sd["path_integrator.omega"].reshape(-1).double()
    un, ur, rd, rg = [], [], [], []
    for p, q in TORUS_PAIRS:
        if tied:
            a = (E[p] - E[q]) / math.sqrt(2)
            zp, zq = Win @ a, -(Win @ a)
        else:
            zp, zq = Win @ E[p], Win @ E[q]
        u, d = zp + zq, (zp - zq) / 2
        read = float(_wrap(om * (Wout @ u)).norm())
        mapr = float(_wrap(om * (Wout @ d)).norm())
        un.append(float(u.norm()))
        ur.append(float(u.norm() / (zp.norm() + zq.norm()).clamp_min(1e-12)))
        rd.append(read)
        rg.append(read / max(mapr, 1e-12))
    out = {"u_norm": float(np.mean(un)), "u_ratio": float(np.mean(ur)), "read": float(np.mean(rd)),
           "readgain": float(np.mean(rg))}
    for k in ("action_to_lie.clock", "action_to_lie.clock_x", "action_to_lie.clock_y"):
        if k in sd:
            out[k.split(".")[-1] + "_angle_absmax"] = float((om * sd[k].reshape(-1).double()).abs().max())
    return out


# --------------------------------------------------------------------------------------------- loading
def _acc_table(path):
    if not os.path.exists(path):
        return None
    t = {}
    for key, rows in json.load(open(path)).items():
        noise, v, T = key.split("|")
        for s, acc, nll in rows:
            t[(v, int(T), int(s))] = (acc, nll)
    return t


def load(root=None, root_wd0=None, eval_json=None, eval_wd0=None, seeds=SEEDS):
    acc = {"main": _acc_table(eval_json or EVAL), "wd0": _acc_table(eval_wd0 or EVAL_WD0)}
    R, missing = {}, []
    for arm, v in ARMS.items():
        for s in seeds:
            d = ckpt_dir(arm, s, root, root_wd0)
            p = os.path.join(d, f"{v}.pt")
            if not os.path.exists(p):
                missing.append(p)
                continue
            b = torch.load(p, map_location="cpu", weights_only=False)
            A = b["config"]["args"]
            assert (A["variant"], A["seed"]) == (v, s), (p, A["variant"], A["seed"])
            assert A["epochs"] == EPOCHS and len(b["losses"]) == EPOCHS, (p, A["epochs"], len(b["losses"]))
            assert abs(A.get("weight_decay", 0.05) - WD[arm]) < 1e-12, (p, A.get("weight_decay"))
            bad = {k: A.get(k) for k, w in RECIPE.items() if A.get(k) != w}
            assert not bad, (p, bad)
            L = np.asarray(b["losses"], float)
            div = not np.all(np.isfinite(L))
            k = classify_run(np.nan_to_num(L, nan=1e9, posinf=1e9))
            tied = v == "Vanilla_r2DF"
            snaps = sorted(glob.glob(os.path.join(d, "snaps", "e*.pt")))
            assert snaps and os.path.basename(snaps[0]) == "e0000.pt" and os.path.basename(snaps[-1]) == "e0900.pt", \
                (d, [os.path.basename(x) for x in snaps[:2]], [os.path.basename(x) for x in snaps[-1:]])
            traj = [(int(os.path.basename(x)[1:5]), geometry_sd(torch.load(x), tied)) for x in snaps]
            g_end = geometry_sd(b["model_state_dict"], tied)
            tab = acc["wd0" if arm == "r4wd0" else "main"]
            rec = {"solved": k["registered"] == "SOLVED" and not div, "cls": "DIVERGED" if div else k["cls"],
                   "final_loss": float(k["tail"]) if not div else float("nan"), **g_end,
                   "g_init": traj[0][1], "traj": traj}
            for T in LENGTHS:
                assert tab is not None and (v, T, s) in tab, f"no evaluation row for {v} T={T} s{s}"
                a_ = tab[(v, T, s)][0]
                rec[f"acc{T}"] = float(a_) if a_ is not None and np.isfinite(a_) else 0.0   # declared: non-finite = 0
            R[(arm, s)] = rec
    return R, missing


# --------------------------------------------------------------------------------------------- decisions
def slot_label(k_slot, k_r2, names):
    helps, not_helps = names
    p = fisher_solved(k_slot, 8, k_r2, 8)
    if k_slot >= 6 and p < .05:
        return helps, p
    if k_slot <= 2:
        return not_helps, p
    return f"PARTIAL ({k_slot}/8 vs r2 {k_r2}/8, Fisher p {p:.4f})", p


def e2_label(k_r2, k_ao, k_df):
    pAO, pDF = fisher_solved(k_ao, 8, k_r2, 8), fisher_solved(k_df, 8, k_ao, 8)
    tail = f"r2 {k_r2}/8, AO {k_ao}/8, DF {k_df}/8; Fisher AO vs r2 {pAO:.4f}, DF vs AO {pDF:.4f}"
    if k_ao >= 6 and k_df <= 2:
        return f"AO SOLVES, DF DOES NOT ({tail})"
    if k_ao >= 6 and pAO < .05:
        return f"OBSERVATION STEPS ARE THE OBSTACLE ({tail})"
    if k_ao <= 2 and k_df >= 6 and pDF < .05:
        return f"THE CLOCK IS THE OBSTACLE ({tail})"
    if k_df <= 2 and k_ao <= 2:
        return f"NEITHER (a second obstacle at rank 2: {tail})"
    return f"PARTIAL ({tail})"


def _rel(rows, num, den):
    return [r[num] / max(r["g_init"][den], 1e-12) for r in rows]


def x2_label(rows):
    sol = [r for r in rows if r["solved"]]
    if len(sol) < 4:
        return f"X2 UNREADABLE (r4wd0 SOLVED {len(sol)}/8 < 4)"
    shrink = float(np.median(_rel(sol, "u_norm", "u_norm")))
    rgrel = float(np.median(_rel(sol, "readgain", "readgain")))
    ur = float(np.median([r["u_ratio"] for r in sol]))
    tail = f"median shrink {shrink:.3f}, readgain/init {rgrel:.3f}, u_ratio {ur:.3f}; {len(sol)} solved"
    if shrink <= 0.2:
        return f"REMOVAL ({tail})"
    if shrink >= 0.5 and rgrel <= 0.1:
        return f"PARKING ({tail})"
    if ur <= 0.15:
        return f"DILUTED ({tail})"
    return f"UNRESOLVED ({tail})"


def first_below(traj, key, frac):
    v0 = traj[0][1][key]
    for ep, g in traj:
        if g[key] <= frac * v0:
            return ep
    return None


def x2b_label(rows):
    sol = [r for r in rows if r["solved"]]
    pairs = [(first_below(r["traj"], "read", 0.5), first_below(r["traj"], "u_norm", 0.5)) for r in sol]
    usable = [(a, b) for a, b in pairs if a is not None and b is not None and a != b]
    n_read = sum(a < b for a, b in usable)
    if not usable:
        return f"NO ORDER (no usable runs; {len(sol)} solved)"
    p = binomtest(n_read, len(usable), 0.5).pvalue
    tail = f"read first in {n_read}/{len(usable)} usable runs, sign p {p:.4f}; {len(sol)} solved"
    if p < .05 and n_read > len(usable) / 2:
        return f"READ CLOCK FALLS FIRST ({tail})"
    if p < .05:
        return f"LATENT CLOCK FALLS FIRST ({tail})"
    return f"NO ORDER ({tail})"


def analyse(R, seeds=SEEDS, out=print):
    k = {a: sum(R[(a, s)]["solved"] for s in seeds) for a in ARMS}
    accs = {a: [R[(a, s)]["acc1024"] for s in seeds] for a in ARMS}
    v = {"solved": k}
    out("== RANKCLOCK (G-1): run classes (SOLVED = final-5% loss < 0.05; S/T/D/R/X) ==")
    for a in ARMS:
        out(f"  {a:6s} SOLVED {k[a]}/8 | acc@1024 {np.mean(accs[a]):.4f} | "
            + " ".join(f"s{s}:{CODE[R[(a, s)]['cls']]}({R[(a, s)]['final_loss']:.3f})" for s in seeds))
    if k["r4"] < 6:
        v["reading"] = f"UNREADABLE (r4 SOLVED {k['r4']}/8 < 6)"
    elif k["r2"] > 2:
        v["reading"] = f"RANK-2 DIFFICULTY NOT REPLICATED (r2 SOLVED {k['r2']}/8 > 2): P, P2 and E2 not read"
    else:
        for key, arm, names in (("P", "slotxy", ("CLOCK EXITS HELP", "CLOCK EXITS DO NOT HELP")),
                                ("P2", "slot", ("A COMMON-MODE EXIT HELPS", "A COMMON-MODE EXIT DOES NOT HELP"))):
            lab, p = slot_label(k[arm], k["r2"], names)
            pa = perm2_p(np.asarray(accs["r2"]), np.asarray(accs[arm]))["p"]
            v[key] = lab
            out(f"\n{key}: {lab} -- {arm} {k[arm]}/8 vs r2 {k['r2']}/8, Fisher p {p:.4f}; acc@1024 "
                f"{np.mean(accs[arm]):.4f} vs {np.mean(accs['r2']):.4f}, perm p {pa:.4f}")
        v["E2"] = e2_label(k["r2"], k["AO"], k["DF"])
        v["reading"] = v["P"]
        out(f"E2 (G1+): {v['E2']}")
        out(f"   DF vs r4: SOLVED {k['DF']}/8 vs {k['r4']}/8, Fisher p {fisher_solved(k['DF'], 8, k['r4'], 8):.4f} (descriptive)")
    v["X2"] = x2_label([R[("r4wd0", s)] for s in seeds])
    v["X2b_wd0"] = x2b_label([R[("r4wd0", s)] for s in seeds])
    v["X2b_r4"] = x2b_label([R[("r4", s)] for s in seeds])
    out(f"\nX2 (r4 weight decay 0): {v['X2']}")
    ref = [R[("r4", s)] for s in seeds if R[("r4", s)]["solved"]]
    if ref:
        out(f"   reference r4 (wd 0.05), {len(ref)} solved: median shrink {np.median(_rel(ref, 'u_norm', 'u_norm')):.3f}, "
            f"readgain/init {np.median(_rel(ref, 'readgain', 'readgain')):.3f}, u_ratio "
            f"{np.median([r['u_ratio'] for r in ref]):.3f}")
    out(f"X2b (P4 ordering): r4wd0 {v['X2b_wd0']} | r4 {v['X2b_r4']}")
    out(f"\nREADING (registered primary): {v['reading']}")
    out("   scope: torus T = 1024, 900 epochs, shared rank, one host; budget-scoped where runs are still DESCENDING.")

    out("\n== geometry medians (all runs, end of training; descriptive) ==")
    for a in ARMS:
        rows = [R[(a, s)] for s in seeds]
        extra = ""
        for kk in ("clock_angle_absmax", "clock_x_angle_absmax", "clock_y_angle_absmax"):
            vals = [r[kk] for r in rows if kk in r]
            if vals:
                extra += f"  max|omega*{kk.split('_angle')[0]}| {np.median(vals):.3f}"
        out(f"  {a:6s} u_ratio {np.median([r['u_ratio'] for r in rows]):.3f}  shrink "
            f"{np.median(_rel(rows, 'u_norm', 'u_norm')):.3f}  readgain {np.median([r['readgain'] for r in rows]):.3f}"
            + extra)
    out("\n== rule 2: r(log10 final loss, acc) per length, finite runs ==")
    fin = [r for r in R.values() if np.isfinite(r["final_loss"])]
    Ls = np.log10(np.array([r["final_loss"] for r in fin]) + 1e-6)
    for T in LENGTHS:
        A = np.array([r[f"acc{T}"] for r in fin])
        rr = float(np.corrcoef(Ls, A)[0, 1]) if len(A) > 2 and A.std() > 0 and Ls.std() > 0 else float("nan")
        out(f"   T={T}: r = {rr:+.3f} over {len(A)} runs")
    out("\nper seed acc@1024 (class):")
    for a in ARMS:
        out(f"  {a:6s} " + " ".join(f"{R[(a, s)]['acc1024']:.3f}{CODE[R[(a, s)]['cls']]}" for s in seeds))
    return v


# --------------------------------------------------------------------------------------------- selftest
def _geom_sd(model_sd, E, ks, read_clock):
    """A step-map state dict whose r4 latents carry a clock of size k on axes 3-4: z_+x = e1 + k e3, z_-x = -e1 + k e3,
    z_+y = e2 + k e4, z_-y = -e2 + k e4; readout columns 3-4 zero unless read_clock."""
    sd = {kk: vv.clone() for kk, vv in model_sd.items()
          if kk.startswith(("action_to_lie.", "path_integrator.", "token_emb."))}
    Z = torch.zeros(4, 4, dtype=torch.float64)
    (px, mx), (py, my) = TORUS_PAIRS
    Z[px, 0], Z[mx, 0], Z[py, 1], Z[my, 1] = 1, -1, 1, -1
    Z[px, 2] = Z[mx, 2] = ks
    Z[py, 3] = Z[my, 3] = ks
    sd["action_to_lie.w_in.weight"] = (torch.linalg.pinv(E[:4]) @ Z).T.float()
    W = model_sd["action_to_lie.w_out.weight"].clone()
    W[:, :2] = W[:, :2].sign() * 0.05                      # a modest, fixed map readout
    if not read_clock:
        W[:, 2:] = 0
    sd["action_to_lie.w_out.weight"] = W
    return sd


def _plant(arm, seed, solved, root, root_wd0, geo=None):
    """Write a REAL checkpoint and snapshot series in the driver's layout. geo (r4 arms only):
    list of (epoch, clock size k, read_clock) for e0000 ... e0900; the last entry is also the checkpoint's geometry."""
    from mapformer.environment import GridWorld
    from mapformer.train_variant import VARIANT_MAP
    v = ARMS[arm]
    torch.manual_seed(seed)
    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=0)
    m = VARIANT_MAP[v](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=64)
    msd = m.state_dict()
    d = ckpt_dir(arm, seed, root, root_wd0)
    os.makedirs(os.path.join(d, "snaps"), exist_ok=True)
    E = msd["token_emb.weight"].double()
    if geo is None or v != "Vanilla_r4mi":
        geo_sds = [(0, None), (900, None)]
    else:
        geo_sds = [(ep, _geom_sd(msd, E, k, rc)) for ep, k, rc in geo]
    for ep, gsd in geo_sds:
        snap = gsd if gsd is not None else {kk: vv for kk, vv in msd.items()
                                             if kk.startswith(("action_to_lie.", "path_integrator.", "token_emb."))}
        torch.save(snap, os.path.join(d, "snaps", f"e{ep:04d}.pt"))
    final = dict(msd)
    if geo_sds[-1][1] is not None:
        final.update(geo_sds[-1][1])
    L = list(np.linspace(2.0, 0.6, 450)) + ([1e-3] * 450 if solved else [0.6] * 450)
    args = dict(RECIPE, variant=v, seed=seed, epochs=EPOCHS, weight_decay=WD[arm])
    torch.save({"model_state_dict": final, "losses": L, "variant": v, "seed": seed,
                "config": {"vocab_size": env.unified_vocab_size, "d_model": 128, "n_heads": 2, "n_layers": 1,
                           "grid_size": 64, "args": args}}, os.path.join(d, f"{v}.pt"))


def _write_eval(path, spec, arms):
    d = {}
    for arm in arms:
        for T in LENGTHS:
            d[f"0.0|{ARMS[arm]}|{T}"] = [[s, 1.0 if i < spec[arm] else 0.9, 0.0] for i, s in enumerate(SEEDS)]
    json.dump(d, open(path, "w"))


def selftest():
    silent = lambda *a, **k: None
    base = {"r2": 0, "slotxy": 7, "slot": 7, "DF": 7, "AO": 1, "r4": 8, "r4wd0": 8}
    # geometry series (epoch, k, read): REMOVAL: clock shrinks to ~0; PARKING: clock kept, readout stops reading it
    # early; READ_THEN_REMOVE gives the read-first ordering for X2b; DILUTED is checked on hand-made records below
    REMOVE = [(0, 1.0, True), (300, 0.6, True), (600, 0.1, True), (900, 0.05, True)]
    PARK = [(0, 1.0, True), (100, 1.0, False), (500, 0.9, False), (900, 0.9, False)]
    READ_THEN_REMOVE = [(0, 1.0, True), (100, 0.9, False), (500, 0.3, False), (900, 0.05, False)]
    STAY_READ = [(0, 1.0, True), (900, 0.9, True)]
    cases = [
        ("P", "CLOCK EXITS HELP", {}, {}),
        ("P", "CLOCK EXITS DO NOT HELP", {"slotxy": 1}, {}),
        ("P", "PARTIAL", {"slotxy": 4}, {}),
        ("P", "PARTIAL", {"r2": 2, "slotxy": 6}, {}),                    # finding 4: 6 vs 2, Fisher p 0.13
        ("P2", "A COMMON-MODE EXIT DOES NOT HELP", {"slot": 2}, {}),
        ("P2", "A COMMON-MODE EXIT HELPS", {}, {}),
        ("reading", "UNREADABLE", {"r4": 5}, {}),
        ("reading", "RANK-2 DIFFICULTY NOT REPLICATED", {"r2": 6}, {}),
        ("E2", "THE CLOCK IS THE OBSTACLE", {}, {}),
        ("E2", "OBSERVATION STEPS ARE THE OBSTACLE", {"AO": 7}, {}),
        ("E2", "AO SOLVES, DF DOES NOT", {"r2": 2, "AO": 6, "DF": 1}, {}),  # finding 4's counterexample
        ("E2", "NEITHER", {"DF": 1}, {}),
        ("E2", "PARTIAL", {"AO": 4, "DF": 4}, {}),
        ("X2", "REMOVAL", {}, {"r4wd0": REMOVE}),
        ("X2", "PARKING", {}, {"r4wd0": PARK}),
        ("X2", "UNRESOLVED", {}, {"r4wd0": STAY_READ}),
        ("X2", "X2 UNREADABLE", {"r4wd0": 2}, {"r4wd0": PARK}),
        ("X2b_wd0", "READ CLOCK FALLS FIRST", {}, {"r4wd0": READ_THEN_REMOVE}),
        ("X2b_wd0", "NO ORDER", {}, {"r4wd0": STAY_READ}),
    ]
    ok = True
    tmp = tempfile.mkdtemp(prefix="rankclock_selftest_")
    for i, (field, want, over, geos) in enumerate(cases):
        spec = dict(base, **over)
        root, root_wd0 = os.path.join(tmp, f"c{i}"), os.path.join(tmp, f"c{i}w")
        for arm in ARMS:
            for j, s in enumerate(SEEDS):
                _plant(arm, s, j < spec[arm], root, root_wd0, geos.get(arm))
        ej, ew = os.path.join(tmp, f"e{i}.json"), os.path.join(tmp, f"e{i}w.json")
        _write_eval(ej, spec, [a for a in ARMS if a != "r4wd0"])
        _write_eval(ew, spec, ["r4wd0"])
        R, missing = load(root, root_wd0, ej, ew)
        assert not missing, missing[:2]
        v = analyse(R, out=silent)
        assert field in v, f"field {field} missing from the verdict (finding 4: no silent fallback)"
        got = v[field]
        good = got.startswith(want)
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {field:8s} want {want!r:36s} got {got[:96]!r}")
    # DILUTED: construct directly (a ratio made small by a growing map, u kept): check the rule on hand-made records
    rec = lambda ur, un, rg: {"solved": True, "u_ratio": ur, "u_norm": un, "readgain": rg,
                              "g_init": {"u_norm": 1.0, "readgain": 1.0}}
    got = x2_label([rec(0.10, 0.4, 0.5)] * 8)
    good = got.startswith("DILUTED")
    ok &= good
    print(f"  {'ok ' if good else 'BAD'} X2       want 'DILUTED' (rule on records)           got {got[:96]!r}")
    # a missing evaluation row must fail loudly, not read as accuracy 0 (finding 7)
    root, root_wd0 = os.path.join(tmp, "m"), os.path.join(tmp, "mw")
    for arm in ARMS:
        for j, s in enumerate(SEEDS):
            _plant(arm, s, j < base[arm], root, root_wd0)
    ej, ew = os.path.join(tmp, "m.json"), os.path.join(tmp, "mw.json")
    _write_eval(ej, base, [a for a in ARMS if a not in ("r4wd0", "AO")])
    _write_eval(ew, base, ["r4wd0"])
    try:
        load(root, root_wd0, ej, ew)
        print("  BAD a missing evaluation row was not caught"); ok = False
    except AssertionError as e:
        print(f"  ok  missing evaluation row raises: {str(e)[:60]}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    R, missing = load()
    if missing:
        print(f"MISSING {len(missing)} checkpoints, e.g. {missing[:3]}")
        sys.exit(2)
    v = analyse(R)
    json.dump(v, open(f"{REPO}/RANKCLOCK_VERDICTS.json", "w"), indent=2)


if __name__ == "__main__":
    main()
