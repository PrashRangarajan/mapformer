"""Registered analysis for RNN_TORUS_PREREG.md: does a recurrent baseline (LSTM / GRU, parameter-matched) learn the long-walk
torus map that one path-integrated attention layer learns? Torus 64 x 64, T = 1024, 900 epochs, FACTORIAL_N8's recipe.

    python3 -m mapformer.analyze_rnn_torus                       (from /home/prashr, after the batch; CPU)
    python3 -m mapformer.analyze_rnn_torus --context-destruction --device cuda:G   (driver step, writes RNN_TORUS_CD.json)
    python3 -m mapformer.analyze_rnn_torus --floors-only         (model-free floors on the batch's eval walks; CPU)
    python3 -m mapformer.analyze_rnn_torus --pilot               (the lr pilot's readout and the registered pick rule)
    python3 -m mapformer.analyze_rnn_torus --selftest            (synthetic batches through every label; CPU, no run dirs)

REGISTERED PRIMARY (n = 8 per cell, seeds 3300-3307; accuracy at T = 1024 on the held-out map):
  path = Path_L1 (MapWM Vanilla_r4, 1 layer), each RNN cell c (arm at its registered rate).
  beats(c): (acc(path) - acc(c) >= 0.05 and exact two-sample permutation p < .05) or (Fisher on SOLVED p < .05, path ahead).
  - UNREADABLE  if path SOLVED < 6/8 (the anchor did not reproduce FACTORIAL's 8/8).
  - PATH WINS   if beats(c) for EVERY RNN cell (intersection-union: level .05 without correction, and selection-proof --
                whichever cell is 'best', path beats it). Amendment 1 (A2, reporting only): each cell's route
                (accuracy / SOLVED / both) is printed; if any cell is separated on SOLVED only with d < 0.05 the label
                reads 'PATH WINS (on SOLVED only for <cells>; accuracy within 0.05: ...)'.
  - RNN WINS    else if some cell c has (acc(c) - acc(path) >= 0.05 and Holm-adjusted permutation p < .05) or
                (Holm-adjusted Fisher p < .05, c ahead); Holm over the K RNN cells, per route.
  - RNN MATCHES else if the SELECTED best cell (most SOLVED, then lowest mean final-5% training loss, then fewer params,
                then name; training loss only, never held-out accuracy) has acc >= acc(path) - 0.02 and SOLVED >= 6/8.
  - UNMEASURED  otherwise.
  Every label carries "[within 900 epochs; DESCENDING k/8 in <cell>]" for the selected cell and every cell path does not beat.
Declared secondaries (no verdict): S1 every RNN cell vs index RoPE 1L (d, perm p, Fisher; RNN ABOVE / BELOW INDEX 1L /
AS IT FALLS); S2 accuracy minus the retrace floor on the same walks; S3 revisit strata (short-gap / long-gap / wrap-only)
beside the retrace floor per stratum; S4 T = 512 / 2048 (out of distribution); S5 own (training) map vs held-out; S6
r(final-5% loss, acc); S7 run classes; S8 context destruction (rule 11 gate on trained models: observations swapped for
another map's along the same walk; a context-using model falls below the blank floor); S9 capacity (large vs small arm).
Floors and strata are FACTORIAL_N8's own functions (analyze_factorial_n8.floors_stream / STRATA), imported unchanged.
"""
import argparse, glob, json, math, os, re, sys, tempfile

import numpy as np
import torch

from mapformer import analyze_factorial_n8 as AF8
from mapformer.environment import GridWorld
from mapformer.stats_core import perm2_p, fisher_solved, classify_run
from mapformer.analyze_lrsweep import holm

REPO = "/home/prashr/mapformer"; EPOCHS = 900; N = 8
TAG = "RNN_TORUS"
RUNS = f"{REPO}/runs/rnn_torus"
SEEDS = [3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307]
PILOT_SEED = 3390
STRATA, SNAME = AF8.STRATA, AF8.SNAME
D_WIN, D_MATCH, K_MATCH, K_PATH = 0.05, 0.02, 6, 6
CD_ENV_SEED, CD_TRIALS = 20000, 50                    # context destruction: observations from GridWorld(seed 20000)

# cell -> (root, variant, layers, lr, fast_attn). Anchors: FACTORIAL_N8's flags exactly (lr 1e-3, --fast-attn).
ANCHORS = {"Path_L1": ("anchor", "Vanilla_r4", 1, 1e-3, True),
           "RoPE_L1": ("anchor", "RoPE", 1, 1e-3, True)}
# RNN arms at their registered rates (RNN_TORUS_PREREG.md, "Arms" and "Pilot": the lr pick rule applied to the pilot).
# Picked 2026-10-26 by pilot_pick on runs/rnn_torus_pilot/lr (seed 3390; RNN_TORUS_PREREG.md, Pilot). Amendment 1 (A3):
# the 4x LSTM also runs at 3e-3 (its 1e-3 / 3e-3 pilot difference, 0.005, was inside the tie band). (variant, rate) pairs.
RNN_ARMS = [("LSTM_L1_h313", 1e-3), ("LSTM_L1_h313", 3e-3), ("LSTM_L1_h157", 3e-3), ("GRU_L1_h181", 1e-3),
            ("LSTM_L2_h111", 3e-3), ("GRU_L2_h128", 3e-3)]
LARGE_OF = {"LSTM_L1_h313": "LSTM_L1_h157"}            # large arm -> its small counterpart (S9; the 1e-3 cell of the
                                                        # large arm when it runs at two rates, Amendment 1 A3)


def lr_root(lr):
    return f"lr{lr:.0e}"                                # 1e-3 -> lr1e-03 (the driver uses the same python expression)


def rnn_cells(arms=None):
    arms = RNN_ARMS if arms is None else arms
    pairs = list(arms.items()) if isinstance(arms, dict) else list(arms)      # {variant: rate} or [(variant, rate), ...]
    from mapformer.model_rnn import ARM_SPECS
    out = {f"{v}@{lr:g}": (lr_root(lr), v, ARM_SPECS[v][1], lr, False) for v, lr in pairs}
    assert len(out) == len(pairs), "duplicate (variant, rate) cell"
    return out


def all_cells(arms=None):
    return {**ANCHORS, **rnn_cells(arms)}


def n_params_of(var):
    from mapformer.model_rnn import ARM_SPECS, n_params
    if var in ARM_SPECS:
        c, l, h = ARM_SPECS[var]; return n_params(c, h, l)
    return {"Vanilla_r4": 204_757, "RoPE": 203_925}[var]


# ------------------------------------------------------------------------------------------------ recipe
VARYING = {"seed", "device", "output_dir", "variant", "lr", "n_layers", "fast_attn"}
REF_CKPT = f"{REPO}/runs/factorial_n8/L1/p0/Vanilla_r4_s2800/Vanilla_r4.pt"   # FACTORIAL_N8's path run: the recipe source


def recipe(args):
    return {k: v for k, v in args.items() if k not in VARYING}


def ref_recipe():
    if not os.path.exists(REF_CKPT):
        return None
    return recipe(torch.load(REF_CKPT, map_location="cpu", weights_only=False)["config"]["args"])


# ------------------------------------------------------------------------------------------------ loading
def load_batch(cells, runs, out_dir, seeds=SEEDS, ref=None):
    """Per cell: acc (T = 512 / 1024 / 2048), SOLVED, classes, tail loss, strata, own map, context destruction. Asserts the
    recipe of every run: all runs share one recipe dict (args minus seed / device / output dir / variant / lr / layers /
    fast_attn), equal to `ref` (FACTORIAL_N8's) when given; each run's lr, layers, fast_attn, epochs, length are the cell's."""
    D = {k: {} for k in ("acc", "acc512", "acc2048", "solved", "cls", "tail", "strata", "own", "cd", "cd_clean")}
    common = None
    try:
        CD = json.load(open(f"{out_dir}/{TAG}_CD.json"))
    except FileNotFoundError:
        CD = {}
    for cell, (root, var, nl, lr, fa) in cells.items():
        J = json.load(open(f"{out_dir}/{TAG}_{root}.json"))
        by = {L: {int(r[0]): r for r in J[f"0.0|{var}|{L}"]} for L in (512, 1024, 2048)}
        for k in D:
            D[k][cell] = []
        try:
            S = json.load(open(f"{out_dir}/{TAG}_{root}_STRATA.json"))
        except FileNotFoundError:
            S = {}
        for s in seeds:
            b = torch.load(f"{runs}/{root}/p0/{var}_s{s}/{var}.pt", map_location="cpu", weights_only=False)
            c = b["config"]; a = c["args"]
            assert b["variant"] == var and b["seed"] == s and len(b["losses"]) == EPOCHS, (cell, s)
            assert c["n_layers"] == nl and c["n_steps"] == 1024 and c["epochs"] == EPOCHS and c["grid_size"] == 64, (cell, s, c)
            assert bool(c.get("fast_attn")) == fa and not c.get("deterministic"), (cell, s, "numerics path")
            assert abs(a["lr"] - lr) < 1e-12 and a["variant"] == var and a["n_layers"] == nl, (cell, s, "cell flags")
            assert a.get("weight_decay", 0.05) == 0.05 and a.get("snapshot_every", 0) == 0 and a.get("init_from") is None, (cell, s)
            r = recipe(a)
            if common is None:
                common = r
                if ref is not None:          # every FACTORIAL_N8 key must be present and equal; a key train_variant gained
                    bad = {k: (r.get(k, "<absent>"), ref[k]) for k in ref if r.get(k, "<absent>") != ref[k]}
                    assert not bad, ("recipe differs from FACTORIAL_N8's", bad)
                    new = sorted(set(r) - set(ref))           # since FACTORIAL_N8 is reported, not fatal
                    if new:
                        print(f"NOTE: CLI keys absent from FACTORIAL_N8's checkpoint: { {k: r[k] for k in new} }")
            assert r == common, (cell, s, "recipe drift inside the batch",
                                 {k: (r.get(k), common.get(k)) for k in set(r) | set(common) if r.get(k) != common.get(k)})
            D["acc"][cell].append(float(by[1024][s][1])); D["acc512"][cell].append(float(by[512][s][1]))
            D["acc2048"][cell].append(float(by[2048][s][1]))
            k = classify_run(b["losses"]); D["cls"][cell].append(k["registered"])
            D["solved"][cell].append(k["registered"] == "SOLVED"); D["tail"][cell].append(float(k["tail"]))
            v = S.get(f"{var}|{s}|1024")
            if v is not None:
                assert abs(v["all"]["acc"] - D["acc"][cell][-1]) < 1e-3, (cell, s, "strata 'all' != eval JSON")
            D["strata"][cell].append({st: (v[st]["acc"] if v.get(st) else None) for st in STRATA} if v else None)
            try:
                O = json.load(open(f"{out_dir}/{TAG}_{root}_OWN_s{s}.json"))
                D["own"][cell].append(float(O[f"0.0|{var}|1024"][0][1]))
            except (FileNotFoundError, KeyError, IndexError):
                D["own"][cell].append(None)
            x = CD.get(f"{root}|{var}|{s}")
            D["cd"][cell].append(None if x is None else float(x["destroyed"]))
            D["cd_clean"][cell].append(None if x is None else float(x["clean"]))
    return D


# ------------------------------------------------------------------------------------------------ registered readings
def select_best(D, rcells):
    """Most SOLVED, then lowest mean final-5% training loss, then fewer params, then name. Training loss only."""
    def key(c):
        return (-sum(D["solved"][c]), float(np.mean(D["tail"][c])), n_params_of(c.split("@")[0]), c)
    return min(rcells, key=key)


def contrast(D, a, b):
    """b - a on accuracy (perm p) and SOLVED (Fisher p)."""
    d = float(np.mean(D["acc"][b]) - np.mean(D["acc"][a]))
    p = perm2_p(D["acc"][a], D["acc"][b])["p"]
    ka, kb = sum(D["solved"][a]), sum(D["solved"][b])
    pf = fisher_solved(ka, len(D["solved"][a]), kb, len(D["solved"][b]))
    return {"d": d, "p_perm": p, "k_a": ka, "k_b": kb, "p_fisher": pf}


def primary(D, rcells, path="Path_L1"):
    rc = list(rcells)
    kp = sum(D["solved"][path]); n = len(D["solved"][path])
    best = select_best(D, rc)
    C = {c: contrast(D, c, path) for c in rc}                         # path - c
    beats = {c: (C[c]["d"] >= D_WIN and C[c]["p_perm"] < .05) or (C[c]["p_fisher"] < .05 and C[c]["k_b"] > C[c]["k_a"])
             for c in rc}
    hp = dict(zip(rc, holm([C[c]["p_perm"] for c in rc]))); hf = dict(zip(rc, holm([C[c]["p_fisher"] for c in rc])))
    rev = {c: (-C[c]["d"] >= D_WIN and hp[c] < .05) or (hf[c] < .05 and C[c]["k_a"] > C[c]["k_b"]) for c in rc}
    route = {c: route_of(C[c]) for c in rc}                            # Amendment 1 (A2): reporting only
    acc_p, acc_b, kb = float(np.mean(D["acc"][path])), float(np.mean(D["acc"][best])), sum(D["solved"][best])
    not_beaten = [c for c in rc if not beats[c]]
    desc = {c: D["cls"][c].count("DESCENDING") for c in rc}
    scope = sorted(set([best] + not_beaten))
    sfx = ("[within 900 epochs; DESCENDING " + ", ".join(f"{desc[c]}/{len(D['cls'][c])} in {c}" for c in scope) + "]")
    if kp < K_PATH:
        lab = f"UNREADABLE: path anchor SOLVED {kp}/{n} < {K_PATH} (FACTORIAL / FACTORIAL_N8: 8/8)"
    elif all(beats.values()):
        worst = max(rc, key=lambda c: min(C[c]["p_perm"] if C[c]["d"] >= D_WIN else 1.0,
                                          C[c]["p_fisher"] if C[c]["k_b"] > C[c]["k_a"] else 1.0))
        sol_only = [c for c in rc if route[c] == "SOLVED" and C[c]["d"] < D_WIN]
        q = (f" (on SOLVED only for {', '.join(sol_only)}; accuracy within 0.05: a within-budget convergence difference, "
             f"not an accuracy separation)" if sol_only else "")
        lab = (f"PATH WINS{q}: path beats every RNN cell ({len(rc)}/{len(rc)}); selected best {best}: acc {acc_b:.4f} vs path "
               f"{acc_p:.4f} (d {acc_p - acc_b:+.4f}, perm p {C[best]['p_perm']:.5f}), SOLVED {kb}/{n} vs {kp}/{n} (Fisher p "
               f"{C[best]['p_fisher']:.5f}); binding cell {worst} (separated on {route[worst]}) {sfx}")
    elif any(rev.values()):
        w = [c for c in rc if rev[c]]
        lab = (f"RNN WINS: {', '.join(w)} above path (Holm over {len(rc)} cells): " +
               "; ".join(f"{c} d {-C[c]['d']:+.4f} Holm perm p {hp[c]:.5f}, SOLVED {C[c]['k_a']}/{n} vs {kp}/{n} Holm Fisher p "
                         f"{hf[c]:.5f}" for c in w) + f" {sfx}")
    elif acc_b >= acc_p - D_MATCH and kb >= K_MATCH:
        lab = (f"RNN MATCHES: selected best {best} acc {acc_b:.4f} >= path {acc_p:.4f} - {D_MATCH} and SOLVED {kb}/{n} >= "
               f"{K_MATCH} {sfx}")
    else:
        lab = (f"UNMEASURED: path does not separate from {', '.join(not_beaten)}; selected best {best} acc {acc_b:.4f} vs "
               f"path {acc_p:.4f}, SOLVED {kb}/{n} vs {kp}/{n} {sfx}")
    return {"label": lab, "best": best, "contrasts": C, "beats": beats, "holm_perm": hp, "holm_fisher": hf, "reversed": rev,
            "route": route}


def route_of(x):
    """Which of beats()'s two tests separates path from the cell (x = contrast(D, cell, path)): 'accuracy', 'SOLVED',
    'both' or 'none'. Amendment 1 (A2): printed beside every cell; it does not change any verdict."""
    a = x["d"] >= D_WIN and x["p_perm"] < .05
    f = x["p_fisher"] < .05 and x["k_b"] > x["k_a"]
    return {(True, True): "both", (True, False): "accuracy", (False, True): "SOLVED", (False, False): "none"}[(a, f)]


def head(label):
    return label.split(":", 1)[0].strip()


def label_kind(label):
    """The label's verdict word(s) without the Amendment-1 qualifier, e.g. 'PATH WINS'."""
    return head(label).split(" (", 1)[0].strip()


def vs_index(D, c, idx="RoPE_L1"):
    x = contrast(D, idx, c)                                            # c - RoPE_L1
    if x["d"] >= D_WIN and x["p_perm"] < .05:
        lab = "RNN ABOVE INDEX 1L"
    elif x["d"] <= -D_WIN and x["p_perm"] < .05:
        lab = "RNN BELOW INDEX 1L"
    else:
        lab = "AS IT FALLS"
    return lab, x


def cd_gate(D, cell, floors):
    """S8 (rule 11): mean accuracy with another map's observations along the same walk vs the blank floor."""
    xs = [x for x in D["cd"][cell] if x is not None]
    if not xs:
        return None, "NOT RUN"
    bl = float(np.mean([floors[s]["all"]["blank"] for s in SEEDS[:len(D["cd"][cell])]]))
    m = float(np.mean(xs))
    return m, ("CONTEXT USED" if m <= bl - 0.10 else "CONTEXT NOT SHOWN") + f" (destroyed {m:.4f} vs blank {bl:.4f})"


# ------------------------------------------------------------------------------------------------ report
def report(D, floors, cells, out=print, seeds=SEEDS):
    rc = [c for c in cells if c not in ANCHORS]
    n = len(seeds)
    V = {"cells": {}, "primary": None, "secondary": {}}
    out(f"== REGISTERED PRIMARY: path 1L vs every RNN cell, seeds {seeds[0]}-{seeds[-1]} (n = {n} per cell; exact permutation "
        f"min p {2 / math.comb(2 * n, n):.5f}, Fisher 8/8 vs 0/8 p 0.00016) ==")
    for c in cells:
        out(f"  {c:20s} params {n_params_of(cells[c][1]):>8,} | acc T=1024 {np.mean(D['acc'][c]):.4f} (sd "
            f"{np.std(D['acc'][c], ddof=1):.4f}; per seed {[round(x, 3) for x in D['acc'][c]]}) | SOLVED {sum(D['solved'][c])}/{n} "
            f"| tail loss {np.mean(D['tail'][c]):.4f} | classes {D['cls'][c]}")
        V["cells"][c] = {"acc": D["acc"][c], "solved": D["solved"][c], "cls": D["cls"][c], "tail": D["tail"][c]}
    fl = np.array([[floors[s]["all"]["blank"], floors[s]["all"]["retrace"], floors[s]["wrap_share"]] for s in seeds])
    out(f"  floors on these eval walks (measured, model-free): blank {fl[:, 0].mean():.4f}, retrace {fl[:, 1].mean():.4f}, "
        f"wrap-only share {fl[:, 2].mean():.3f}; best n-gram 0.576 (quoted, RANK_MI)")
    P = primary(D, rc)
    for c in rc:
        x = P["contrasts"][c]
        out(f"  path - {c:20s}: d {x['d']:+.4f} (perm p {x['p_perm']:.5f}); SOLVED {x['k_b']}/{n} vs {x['k_a']}/{n} (Fisher p "
            f"{x['p_fisher']:.5f}) -> {'path beats it (route: ' + P['route'][c] + ')' if P['beats'][c] else 'NOT separated'}")
    out(f"  selected best RNN cell (training loss only): {P['best']}")
    out(f"  REGISTERED PRIMARY: {P['label']}")
    V["primary"] = {"label": P["label"], "best": P["best"], "beats": P["beats"],
                    "contrasts": P["contrasts"], "holm_perm": P["holm_perm"], "holm_fisher": P["holm_fisher"]}
    small = [c for c in rc if n_params_of(c.split("@")[0]) < 500_000]
    if 0 < len(small) < len(rc):                  # registered companion reading: the same rule over the matched-parameter cells
        Pm = primary(D, small)
        out(f"  REGISTERED MATCHED-PARAMETER READING ({len(small)} cells at ~205k): {Pm['label']}")
        V["matched"] = {"label": Pm["label"], "best": Pm["best"], "beats": Pm["beats"]}

    out("== SECONDARY (declared, no verdict) ==")
    S = V["secondary"]
    S["S1"] = {}
    for c in rc:
        lab, x = vs_index(D, c)
        S["S1"][c] = {"label": lab, **x}
        out(f"  S1 {c:20s} - RoPE_L1: {x['d']:+.4f} (perm p {x['p_perm']:.5f}); SOLVED {x['k_b']}/{n} vs {x['k_a']}/{n} "
            f"(Fisher p {x['p_fisher']:.5f}) -> {lab}{' [selected best]' if c == P['best'] else ''}")
    S["S2"] = {}
    for c in cells:
        m = [a - floors[s]["all"]["retrace"] for a, s in zip(D["acc"][c], seeds)]
        S["S2"][c] = m
        out(f"  S2 acc - retrace floor (same walks) {c:20s}: {np.mean(m):+.4f} ({sum(x > 0 for x in m)}/{n} above)")
    S["S3"] = {}
    fr = {st: float(np.mean([floors[s][st]["retrace"] for s in seeds])) for st in STRATA}
    for c in cells:
        rows = [x for x in D["strata"][c] if x]
        if rows:
            mean = {st: float(np.mean([x[st] for x in rows if x[st] is not None])) for st in STRATA}
            S["S3"][c] = mean
            out(f"  S3 strata {c:20s}: " + ", ".join(f"{SNAME[st]} {mean[st]:.3f} (retrace {fr[st]:.3f})" for st in STRATA))
    for c in cells:
        out(f"  S4 OOD (rule 10) {c:20s}: T=512 {np.mean(D['acc512'][c]):.4f}, T=2048 {np.mean(D['acc2048'][c]):.4f}")
    S["S5"] = {}
    for c in cells:
        own = [x for x in D["own"][c] if x is not None]
        if own:
            S["S5"][c] = own
            out(f"  S5 own (training) map {c:20s}: {np.mean(own):.4f} vs held-out {np.mean(D['acc'][c]):.4f}")
    x = np.array(sum((D["tail"][c] for c in cells), [])); y = np.array(sum((D["acc"][c] for c in cells), []))
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    S["S6"] = r; out(f"  S6 r(final-5% training loss, acc T=1024) over the {len(x)} runs: {r:+.3f}")
    S["S7"] = {c: {k: D["cls"][c].count(k) for k in ("SOLVED", "STALLED", "DESCENDING")} for c in cells}
    out("  S7 run classes: " + "; ".join(f"{c} {S['S7'][c]}" for c in cells))
    S["S8"] = {}
    for c in cells:
        m, lab = cd_gate(D, c, floors)
        S["S8"][c] = {"destroyed": m, "label": lab}
        cl = [v for v in D["cd_clean"][c] if v is not None]
        out(f"  S8 context destruction {c:20s}: {lab}" + (f"; same {CD_TRIALS} walks intact {np.mean(cl):.4f}" if cl else ""))
    S["S9"] = {}
    for big, small in LARGE_OF.items():
        bc = sorted([c for c in rc if c.startswith(big + "@")], key=lambda c: (c != f"{big}@0.001", c))
        sc = [c for c in rc if c.startswith(small + "@")]
        if bc and sc:
            x = contrast(D, sc[0], bc[0]); S["S9"][bc[0]] = x
            out(f"  S9 capacity {bc[0]} - {sc[0]}: {x['d']:+.4f} (perm p {x['p_perm']:.5f}); SOLVED {x['k_b']} vs {x['k_a']}")
    return V


# ------------------------------------------------------------------------------------------------ context destruction
@torch.no_grad()
def cd_eval(model, seed, dev, T=1024, n_trials=CD_TRIALS, size=64):
    """Rule 11 on a trained model. Walk w is eval_noise_refine's walk (np seed 1234 + seed, held-out map env seed 10000);
    the model reads the SAME actions with the observations of map B (env seed 20000) and is scored on map A's targets at
    A's revisits. Returns accuracy intact and destroyed on the same walks."""
    A = GridWorld(size=size, n_obs_types=16, p_empty=0.5, seed=10000)
    B = GridWorld(size=size, n_obs_types=16, p_empty=0.5, seed=CD_ENV_SEED)
    np.random.seed(1234 + seed)
    ok = {"clean": 0, "destroyed": 0}; tot = 0
    for _ in range(n_trials):
        st = np.random.get_state()
        ta, _oa, ra = A.generate_trajectory(T)
        st_after = np.random.get_state()
        np.random.set_state(st)
        tb, _ob, rb = B.generate_trajectory(T)
        assert torch.equal(ta[0::2], tb[0::2]) and torch.equal(ra, rb), "walks must share actions and revisits"
        assert all(np.array_equal(x, y) for x, y in zip(np.random.get_state()[1:3], st_after[1:3]))
        m = ra[1:].to(dev); tgt = ta[1:].to(dev)
        if m.sum() == 0:
            continue
        for k, tok in (("clean", ta), ("destroyed", tb)):
            pred = model(tok[None, :-1].to(dev)).argmax(-1)[0]
            ok[k] += int((pred[m] == tgt[m]).sum())
        tot += int(m.sum())
    return {k: v / tot for k, v in ok.items()} | {"n": tot}


def run_cd(dev, cells=None, runs=RUNS, seeds=SEEDS, out=f"{REPO}/{TAG}_CD.json"):
    from mapformer.model_rnn import register
    torch.backends.cuda.matmul.allow_tf32 = False                     # Amendment 1 (A4): fp32, as every other RNN forward
    torch.backends.cudnn.allow_tf32 = False
    VM = register()
    cells = all_cells() if cells is None else cells
    res = {}
    for cell, (root, var, nl, lr, fa) in cells.items():
        for s in seeds:
            b = torch.load(f"{runs}/{root}/p0/{var}_s{s}/{var}.pt", map_location="cpu", weights_only=False); c = b["config"]
            m = VM[var](vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"],
                        grid_size=c["grid_size"])
            m.load_state_dict(b["model_state_dict"]); m = m.to(dev).eval()
            res[f"{root}|{var}|{s}"] = r = cd_eval(m, s, dev)
            print(f"CD {cell} s{s}: clean {r['clean']:.4f} destroyed {r['destroyed']:.4f} (n {r['n']})", flush=True)
            del m
    json.dump(res, open(out, "w"), indent=1)


# ------------------------------------------------------------------------------------------------ pilot
PILOT_RATES = (3e-4, 1e-3, 3e-3, 1e-2)


def pilot_pick(curves):
    """The registered lr pick rule (RNN_TORUS_PREREG.md, Pilot). curves: {lr: [(epoch, loss), ...]} for one arm, logged every
    5 epochs of a 900-epoch cosine run stopped at epoch 300. Score = mean logged loss over epochs 275-300 (6 points).
    A rate is UNSTABLE if any logged loss after epoch 50 exceeds its running minimum by > 0.5 or is non-finite; unstable
    rates are excluded. Pick = lowest score; a rate within 0.01 of the lowest that is closer to 1e-3 (the anchors' rate, in
    log distance) wins the tie."""
    sc, unst = {}, {}
    for lr, cv in curves.items():
        e = np.array([x[0] for x in cv]); l = np.array([x[1] for x in cv], float)
        post = l[e > 50]; runmin = np.minimum.accumulate(l)[e > 50]
        unst[lr] = bool((~np.isfinite(l)).any() or (post - runmin > 0.5).any())
        w = l[(e >= 275) & (e <= 300)]
        sc[lr] = float(w.mean()) if len(w) == 6 else float("nan")
    ok = [lr for lr in sc if not unst[lr] and np.isfinite(sc[lr])]
    if not ok:
        return None, sc, unst
    lo = min(sc[lr] for lr in ok)
    near = [lr for lr in ok if sc[lr] <= lo + 0.01]
    return min(near, key=lambda lr: (abs(math.log10(lr) + 3), lr)), sc, unst


def read_curve(log):
    out = []
    for line in open(log):
        m = re.match(r"\s*Epoch\s+(\d+)/900 \| Loss: ([0-9.naninf]+)", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def pilot(root=f"{REPO}/runs/rnn_torus_pilot/lr", out=print):
    arms = {}
    for f in sorted(glob.glob(f"{root}/*_lr*_s{PILOT_SEED}.log")):
        m = re.match(r"(.+)_lr([0-9.e-]+)_s\d+\.log", os.path.basename(f))
        arms.setdefault(m.group(1), {})[float(m.group(2))] = read_curve(f)
    picks = {}
    for a, cv in arms.items():
        pick, sc, un = pilot_pick(cv)
        picks[a] = pick
        out(f"  {a:14s} " + " | ".join(f"lr {lr:g}: loss@275-300 {sc[lr]:.4f}{' UNSTABLE' if un[lr] else ''} "
                                       f"(last epoch {cv[lr][-1][0] if cv[lr] else 0})" for lr in sorted(cv)) +
            f" -> pick {pick}")
    return picks


# ------------------------------------------------------------------------------------------------ selftest
def _fake(out_dir, runs, cells, acc, solved, desc=None, cd=None, ref=None, seeds=SEEDS):
    desc = desc or {}; cd = cd or {}
    base = dict(ref or {"n_steps": 1024, "batch_size": 16, "n_batches": 98, "epochs": EPOCHS, "schedule": "cosine",
                        "data_workers": 3, "weight_decay": 0.05, "snapshot_every": 0, "init_from": None, "deterministic": False,
                        "d_model": 128, "n_heads": 2, "grid_size": 64})
    roots = {}
    CD = {}
    for cell, (root, var, nl, lr, fa) in cells.items():
        J = roots.setdefault(root, ({}, {}))
        for L in (512, 1024, 2048):
            J[0][f"0.0|{var}|{L}"] = []
        nd = 0
        for i, s in enumerate(seeds):
            a = acc[cell][i]
            if solved[cell][i]:
                losses = list(np.linspace(1.0, 0.01, EPOCHS))
            elif nd < desc.get(cell, 0):
                nd += 1; losses = list(np.linspace(1.0, 0.3, EPOCHS))
            else:
                losses = [1.0] * 100 + [0.3 + 0.001 * i] * (EPOCHS - 100)
            args = dict(base, seed=s, device="cuda:0", output_dir="x", variant=var, lr=lr, n_layers=nl, fast_attn=fa)
            cfg = {"n_layers": nl, "n_steps": 1024, "epochs": EPOCHS, "fast_attn": fa, "deterministic": False, "grid_size": 64,
                   "args": args}
            d = f"{runs}/{root}/p0/{var}_s{s}"; os.makedirs(d, exist_ok=True)
            torch.save({"variant": var, "seed": s, "losses": losses, "config": cfg}, f"{d}/{var}.pt")
            for L in (512, 1024, 2048):
                J[0][f"0.0|{var}|{L}"].append([s, a if L == 1024 else a - 0.01, 0.5])
            J[1][f"{var}|{s}|1024"] = {"all": {"acc": a}, "plain_lag<128": {"acc": min(1, a + 0.05)},
                                      "plain_lag>=128": {"acc": a - 0.3}, "wrap": {"acc": a - 0.3}}
            json.dump({f"0.0|{var}|1024": [[s, a + 0.01, 0.1]]}, open(f"{out_dir}/{TAG}_{root}_OWN_s{s}.json", "w"))
            CD[f"{root}|{var}|{s}"] = {"clean": a, "destroyed": cd.get(cell, 0.27), "n": 1}
    for root, (J, S) in roots.items():
        json.dump(J, open(f"{out_dir}/{TAG}_{root}.json", "w")); json.dump(S, open(f"{out_dir}/{TAG}_{root}_STRATA.json", "w"))
    json.dump(CD, open(f"{out_dir}/{TAG}_CD.json", "w"))


def selftest():
    ok = True

    def chk(cond, what):
        nonlocal ok
        ok &= bool(cond); print(f"  {'ok ' if cond else 'BAD'} {what}")

    arms = {"LSTM_L1_h157": 1e-3, "GRU_L1_h181": 3e-3, "GRU_L2_h128": 1e-3}
    cells = all_cells(arms)
    rc = [c for c in cells if c not in ANCHORS]
    # the matched-parameter companion reading: a large cell is added to a copy of the family; path wins over the three
    # small cells but not over the large one -> primary UNMEASURED / RNN MATCHES, matched reading PATH WINS
    cellsL = all_cells(list(arms.items()) + [("LSTM_L1_h313", 3e-3), ("LSTM_L1_h313", 1e-3)])   # the large arm at 2 rates
    BIG, BIG3 = "LSTM_L1_h313@0.001", "LSTM_L1_h313@0.003"
    one = [1.0] * 8; T8, F8 = [True] * 8, [False] * 8
    low = [0.70, 0.72, 0.71, 0.73, 0.69, 0.74, 0.72, 0.70]; mid = [0.86, 0.87, 0.85, 0.88, 0.84, 0.86, 0.87, 0.85]
    near = [0.99, 1.0, 0.995, 0.99, 1.0, 0.985, 0.99, 1.0]
    mixed = [1.0, 0.80, 1.0, 0.75, 1.0, 0.8, 1.0, 0.78]; half = [True, False] * 4
    fl = {s: {"all": {"blank": 0.51, "retrace": 0.843, "n": 1}, "wrap_share": 0.08,
              **{st: {"retrace": 0.9, "blank": 0.5, "n": 1} for st in STRATA}} for s in SEEDS}
    L, M, G = rc  # LSTM_L1_h157@0.001, GRU_L1_h181@0.003, GRU_L2_h128@0.001
    base_acc = {"Path_L1": one, "RoPE_L1": low}; base_sol = {"Path_L1": T8, "RoPE_L1": F8}
    cases = {
        "path_wins": ({L: mid, M: mid, G: low}, {L: F8, M: F8, G: F8}, {}, {},
                      ["REGISTERED PRIMARY: PATH WINS: path beats every RNN cell", f"selected best", "RNN ABOVE INDEX 1L",
                       "AS IT FALLS", "CONTEXT USED", "DESCENDING 0/8", "path beats it (route: both)", "(separated on both)"],
                      ["RNN MATCHES", "RNN WINS", "UNMEASURED", "UNREADABLE", "on SOLVED only"]),
        "fisher_route": ({L: near, M: mid, G: mid}, {L: F8, M: F8, G: F8}, {}, {},
                         ["REGISTERED PRIMARY: PATH WINS (on SOLVED only for " + L + "; accuracy within 0.05",
                          "binding cell " + L + " (separated on SOLVED)", "path beats it (route: SOLVED)"],
                         ["UNMEASURED", "RNN MATCHES", "REGISTERED PRIMARY: PATH WINS: "]),
        "matches": ({L: near, M: mid, G: low}, {L: [True] * 7 + [False], M: F8, G: F8}, {}, {},
                    [f"REGISTERED PRIMARY: RNN MATCHES: selected best {L}"], ["PATH WINS", "RNN WINS", "UNMEASURED"]),
        "unmeasured": ({L: mixed, M: mid, G: mid}, {L: half, M: F8, G: F8}, {L: 2}, {},
                       ["REGISTERED PRIMARY: UNMEASURED: path does not separate from " + L, f"DESCENDING 2/8 in {L}"],
                       ["PATH WINS", "RNN MATCHES", "RNN WINS"]),
        "unreadable": ({L: mid, M: mid, G: mid}, {L: F8, M: F8, G: F8}, {}, {},
                       ["REGISTERED PRIMARY: UNREADABLE: path anchor SOLVED 4/8"], ["PATH WINS", "RNN MATCHES"]),
        "rnn_wins": ({L: one, M: mid, G: mid}, {L: T8, M: F8, G: F8}, {}, {},
                     ["REGISTERED PRIMARY: RNN WINS: " + L], ["PATH WINS", "RNN MATCHES", "UNMEASURED"]),
        "cd_not_shown": ({L: mid, M: mid, G: low}, {L: F8, M: F8, G: F8}, {}, {L: 0.51},
                         ["CONTEXT NOT SHOWN (destroyed 0.5100", "RNN BELOW INDEX 1L"], []),
    }
    reached = set()
    with tempfile.TemporaryDirectory() as tmp:
        for name, (racc, rsol, desc, cd, wants, nots) in cases.items():
            d = f"{tmp}/{name}"; os.makedirs(d)
            acc = dict(base_acc, **racc); sol = dict(base_sol, **rsol)
            if name == "unreadable":
                acc["Path_L1"] = [1.0] * 4 + [0.8] * 4; sol["Path_L1"] = [True] * 4 + [False] * 4
            if name == "rnn_wins":
                acc["Path_L1"] = [0.9] * 8; sol["Path_L1"] = [True] * 6 + [False] * 2
            if name == "cd_not_shown":
                acc[G] = [0.55] * 8
            _fake(d, f"{d}/runs", cells, acc, sol, desc, cd)
            D = load_batch(cells, f"{d}/runs", d)
            lines = []
            V = report(D, fl, cells, out=lines.append)
            txt = "\n".join(lines); reg = "\n".join(l for l in lines if "REGISTERED PRIMARY:" in l)
            for w in wants:
                chk(w in txt, f"[{name}] '{w}'")
            for w in nots:
                chk(w not in reg, f"[{name}] absent from the registered line: '{w}'")
            reached.add(label_kind(V["primary"]["label"]))
            chk(json.dumps(V, default=str) is not None, f"[{name}] verdict dict serialises")
        chk(reached == {"PATH WINS", "RNN WINS", "RNN MATCHES", "UNMEASURED", "UNREADABLE"}, f"every label reached: {sorted(reached)}")
        d = f"{tmp}/matched"; os.makedirs(d)
        _fake(d, f"{d}/runs", cellsL, dict(base_acc, **{L: mid, M: mid, G: low, BIG: near, BIG3: mid}),
              dict(base_sol, **{L: F8, M: F8, G: F8, BIG: [True] * 7 + [False], BIG3: F8}))
        D = load_batch(cellsL, f"{d}/runs", d); lines = []
        V = report(D, fl, cellsL, out=lines.append); txt = "\n".join(lines)
        chk(f"REGISTERED PRIMARY: RNN MATCHES: selected best {BIG}" in txt, "[matched] primary RNN MATCHES via the large cell")
        chk("REGISTERED MATCHED-PARAMETER READING (3 cells at ~205k): PATH WINS" in txt, "[matched] matched-parameter PATH WINS")
        chk(f"S9 capacity {BIG} - {L}" in txt, "[matched] S9 capacity line uses the large arm's 1e-3 cell")
        chk("(4 cells at ~205k)" not in txt and "(3 cells at ~205k)" in txt and f"path - {BIG3}" in txt,
            "[matched] both rates of the large arm enter the primary family, not the matched reading")
        try:
            all_cells([("GRU_L2_h128", 1e-3), ("GRU_L2_h128", 1e-3)]); chk(False, "duplicate cell refused")
        except AssertionError:
            chk(True, "duplicate (variant, rate) cell refused")
        lines = []; report(load_batch(cells, f"{tmp}/path_wins/runs", f"{tmp}/path_wins"), fl, cells, out=lines.append)
        chk(not any("MATCHED-PARAMETER" in l for l in lines), "[matched] no companion reading when every cell is small")
        # the selection uses training loss, never accuracy: a cell with higher accuracy but higher loss is not selected
        d = f"{tmp}/sel"; os.makedirs(d)
        _fake(d, f"{d}/runs", cells, dict(base_acc, **{L: mid, M: [0.95] * 8, G: low}), dict(base_sol, **{L: F8, M: F8, G: F8}))
        D = load_batch(cells, f"{d}/runs", d)
        D["tail"][L] = [0.10] * 8; D["tail"][M] = [0.20] * 8; D["tail"][G] = [0.30] * 8
        chk(select_best(D, rc) == L, f"selection by training loss: {select_best(D, rc)} (acc-best would be {M})")
        D["solved"][G] = [True] + [False] * 7
        chk(select_best(D, rc) == G, "selection: most SOLVED first")
        # recipe drift: a changed shared key, and a mismatch with the reference recipe, are caught
        for kind in ("inside", "ref_value", "ref_absent"):
            d = f"{tmp}/drift_{kind}"; os.makedirs(d)
            _fake(d, f"{d}/runs", cells, dict(base_acc, **{L: mid, M: mid, G: low}), dict(base_sol, **{L: F8, M: F8, G: F8}))
            if kind == "inside":
                p = f"{d}/runs/lr1e-03/p0/GRU_L2_h128_s3303/GRU_L2_h128.pt"; b = torch.load(p, weights_only=False)
                b["config"]["args"]["batch_size"] = 32; torch.save(b, p); ref = None
            elif kind == "ref_value":
                ref = {"n_steps": 512}                                 # a FACTORIAL_N8 value differs
            else:
                ref = {"n_steps": 1024, "aux_coef": 0.0}               # a FACTORIAL_N8 key is missing from the run
            try:
                load_batch(cells, f"{d}/runs", d, ref=ref); chk(False, f"recipe drift ({kind}) caught")
            except AssertionError:
                chk(True, f"recipe drift ({kind}) caught")
        # a wrong lr on an RNN run is caught
        d = f"{tmp}/lr"; os.makedirs(d)
        _fake(d, f"{d}/runs", cells, dict(base_acc, **{L: mid, M: mid, G: low}), dict(base_sol, **{L: F8, M: F8, G: F8}))
        p = f"{d}/runs/lr3e-03/p0/GRU_L1_h181_s3300/GRU_L1_h181.pt"; b = torch.load(p, weights_only=False)
        b["config"]["args"]["lr"] = 1e-3; torch.save(b, p)
        try:
            load_batch(cells, f"{d}/runs", d); chk(False, "wrong lr caught")
        except AssertionError:
            chk(True, "wrong lr on a run is caught")
    # pilot pick rule: lowest score; tie within 0.01 -> closer to 1e-3; unstable excluded
    ep = list(range(5, 305, 5))
    mk = lambda f: [(e, f(e)) for e in ep]
    cv = {3e-4: mk(lambda e: 1.0 - e / 1000), 1e-3: mk(lambda e: 0.9 - e / 1000), 3e-3: mk(lambda e: 0.895 - e / 1000),
          1e-2: mk(lambda e: 0.5 if e < 200 else 1.5)}
    pick, sc, un = pilot_pick(cv)
    chk(pick == 1e-3 and un[1e-2] and not un[3e-3], f"pilot pick tie -> 1e-3, unstable 1e-2 excluded ({pick}, {un})")
    cv[3e-3] = mk(lambda e: 0.8 - e / 1000)
    chk(pilot_pick(cv)[0] == 3e-3, "pilot pick: clearly lower loss wins")
    # every cell's parameter count is within 1% of its target
    from mapformer.model_rnn import ARM_SPECS, TARGET_PATH, TARGET_ROPE4
    for v in ARM_SPECS:
        p = n_params_of(v); t = TARGET_PATH if p < 500_000 else TARGET_ROPE4
        chk(abs(p - t) / t < 0.01, f"{v} params {p:,} within 1% of {t:,}")
    print("SELFTEST", "ALL PASS" if ok else "FAILED"); return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--floors-only", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--context-destruction", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.pilot:
        pilot(); return
    if a.context_destruction:
        run_cd(torch.device(a.device)); return
    floors = {s: AF8.floors_stream(s) for s in SEEDS}
    if a.floors_only:
        for s, f in floors.items():
            print(f"  s{s}: n {f['all']['n']} blank {f['all']['blank']:.4f} retrace {f['all']['retrace']:.4f} wrap-only share "
                  f"{f['wrap_share']:.3f} | retrace by stratum " +
                  ", ".join(f"{SNAME[st]} {f[st]['retrace']:.3f} (n {f[st]['n']})" for st in STRATA))
        return
    cells = all_cells()
    D = load_batch(cells, RUNS, REPO, ref=ref_recipe())
    V = report(D, floors, cells)
    V["floors"] = floors
    json.dump(V, open(f"{REPO}/{TAG}_VERDICTS.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
