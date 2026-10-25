"""Registered analysis for FACTORIAL_N8_PREREG.md: a fresh-seed replication of FACTORIAL (index RoPE at 1 / 4 layers / looped
x4 vs path-integrated MapWM at 1 layer, torus T = 1024, 900 epochs, --fast-attn).

    python3 -m mapformer.analyze_factorial_n8              (from /home/prashr, after the batch; CPU)
    python3 -m mapformer.analyze_factorial_n8 --selftest   (synthetic batches through every reading + floor cross-checks)

A thin wrapper: the registered decision rules are FACTORIAL's own functions (`analyze_factorial.c0 / catch_up / helps`),
imported unchanged, so `analyze_factorial.py` stays byte-identical to the md5 recorded in runs/factorial/code_md5.txt.
REGISTERED PRIMARY: the fresh batch (seeds 2800-2803) ALONE (rule 6). Declared secondaries, no verdict: pooled n = 8 with
FACTORIAL's seeds 500-503 (cross-batch, DESCRIPTIVE ONLY: train.py and train_variant.py changed between the batches, by
additive off-default-path edits); per-cell batch shift; revisit strata; floors measured on the batch's own eval walks
(blank, retrace, wrap-only share; retrace ported from docs/audits/2026-10-05/score_rank_floors.py) with each cell's
accuracy minus the retrace floor on the same walks; own-map accuracy; r(final loss, acc); accuracy at 512 / 2048 (OOD).
"""
import argparse, collections, hashlib, json, os, sys, tempfile

import numpy as np
import torch

from mapformer import analyze_factorial as AF                       # registered rules, unchanged
from mapformer.environment import GridWorld
from mapformer.eval_rank_strata import kinds_vec
from mapformer.stats_core import perm2_p, fisher_solved, classify_run

REPO = "/home/prashr/mapformer"; EPOCHS = 900
FRESH = {"tag": "FACTORIAL_N8", "runs": f"{REPO}/runs/factorial_n8", "seeds": [2800, 2801, 2802, 2803]}
REF = {"tag": "FACTORIAL", "runs": f"{REPO}/runs/factorial", "seeds": [500, 501, 502, 503]}
CELLS = AF.CELLS                                                    # cell -> (root, variant, layers)
STRATA = ("plain_lag<128", "plain_lag>=128", "wrap")
SNAME = {"plain_lag<128": "short-gap", "plain_lag>=128": "long-gap", "wrap": "wrap-only"}
# recipe keys that must be identical between a fresh run and FACTORIAL's run of the same cell (args dict of the checkpoint)
SKIP_ARGS = {"seed", "device", "output_dir", "weight_decay", "snapshot_every"}
N_ACT, BLANK = 4, 20
DELTA = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
DIFF_NOTE = ("train.py (+8 lines: epoch_callback, default None) and train_variant.py (+61 lines: new variants, "
             "--weight-decay default 0.05 = train()'s default, --snapshot-every default 0) changed since FACTORIAL's 4041da9; "
             "every other module in the import closure is unchanged or new")


# ------------------------------------------------------------------------------------------------ floors (model-free)
def floors_walk(tok, rev, size):
    """Per revisit target of one walk: (blank hit, retrace hit, stratum). Retrace = while a run of actions reverses the
    previous run, step j copies the observation 2j STEPS back, o[t - 2j] (score_rank_floors.py, verbatim logic)."""
    a = tok[0::2]; o = tok[1::2]; r = rev[1::2]; T = len(a)
    new_u, lag = kinds_vec(torch.as_tensor(tok), size)
    out = []; prev_run = cur = 0
    for t in range(T):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_run = cur if (t > 0 and a[t] == (a[t - 1] ^ 1)) else 0
            cur = 1
        if r[t]:
            j = cur; pr = BLANK
            if prev_run > 0 and j <= prev_run and t - 2 * j >= 0:
                pr = o[t - 2 * j]
            st = "wrap" if new_u[t] else ("plain_lag<128" if lag[t] < 128 else "plain_lag>=128")
            out.append((o[t] == BLANK, pr == o[t], st))
    return out


def floors_stream(seed, T=1024, n_trials=100, env_seed=10000, size=64):
    """eval_noise_refine's stream exactly: GridWorld(64, 16, 0.5, env_seed), np.random.seed(1234 + seed), n_trials walks."""
    env = GridWorld(size=size, n_obs_types=16, p_empty=0.5, seed=env_seed)
    np.random.seed(1234 + seed)
    c = collections.Counter()
    for _ in range(n_trials):
        tok, _o, rev = env.generate_trajectory(T)
        for b, rt, st in floors_walk(tok.numpy(), rev.numpy(), size):
            for k in ("all", st):
                c[f"n|{k}"] += 1; c[f"blank|{k}"] += int(b); c[f"retrace|{k}"] += int(rt)
    f = {k: {"n": c[f"n|{k}"], "blank": c[f"blank|{k}"] / max(1, c[f"n|{k}"]),
             "retrace": c[f"retrace|{k}"] / max(1, c[f"n|{k}"])} for k in ("all",) + STRATA}
    f["wrap_share"] = float(f["wrap"]["n"] / f["all"]["n"])
    return f


# ------------------------------------------------------------------------------------------------ loading
def load_batch(B, out_dir, ref_args=None):
    """acc (T=512/1024/2048), SOLVED, classes, tail loss, strata, own map per cell; asserts the recipe."""
    D = {"acc": {}, "acc512": {}, "acc2048": {}, "solved": {}, "cls": {}, "tail": {}, "strata": {}, "own": {}, "args": {}}
    for cell, (root, var, nl) in CELLS.items():
        J = json.load(open(f"{out_dir}/{B['tag']}_{root}.json"))
        by = {L: {int(r[0]): r for r in J[f"0.0|{var}|{L}"]} for L in (512, 1024, 2048)}
        for k in D:
            D[k][cell] = []
        try:
            S = json.load(open(f"{out_dir}/{B['tag']}_{root}_STRATA.json"))
        except FileNotFoundError:
            S = {}
        for s in B["seeds"]:
            b = torch.load(f"{B['runs']}/{root}/p0/{var}_s{s}/{var}.pt", map_location="cpu", weights_only=False)
            c = b["config"]
            assert b["variant"] == var and b["seed"] == s and len(b["losses"]) == EPOCHS, (cell, s)
            assert c["n_layers"] == nl and c["n_steps"] == 1024 and c["epochs"] == EPOCHS and c.get("fast_attn"), (cell, s, c)
            assert c["grid_size"] == 64 and not c.get("deterministic"), (cell, s)
            a = c["args"]
            assert a.get("weight_decay", 0.05) == 0.05 and a.get("snapshot_every", 0) == 0, (cell, s, "recipe drift")
            D["args"][cell].append({k: v for k, v in a.items() if k not in SKIP_ARGS})
            if ref_args is not None:                                 # identical flags to FACTORIAL's run of this cell
                assert D["args"][cell][-1] == ref_args[cell][0], (cell, s, "flags differ from FACTORIAL",
                    {k: (D["args"][cell][-1].get(k), ref_args[cell][0].get(k)) for k in
                     set(D["args"][cell][-1]) | set(ref_args[cell][0]) if D["args"][cell][-1].get(k) != ref_args[cell][0].get(k)})
            D["acc"][cell].append(float(by[1024][s][1])); D["acc512"][cell].append(float(by[512][s][1]))
            D["acc2048"][cell].append(float(by[2048][s][1]))
            k = classify_run(b["losses"]); D["cls"][cell].append(k["registered"])
            D["solved"][cell].append(k["registered"] == "SOLVED"); D["tail"][cell].append(float(k["tail"]))
            v = S.get(f"{var}|{s}|1024")
            if v is not None:
                assert abs(v["all"]["acc"] - D["acc"][cell][-1]) < 1e-9, (cell, s, "strata 'all' != eval JSON")
            D["strata"][cell].append({st: (v[st]["acc"] if v and v.get(st) else None) for st in STRATA} if v else None)
            try:
                O = json.load(open(f"{out_dir}/{B['tag']}_{root}_OWN_s{s}.json")); D["own"][cell].append(float(O[f"0.0|{var}|1024"][0][1]))
            except (FileNotFoundError, KeyError, IndexError):
                D["own"][cell].append(None)
    return D


# ------------------------------------------------------------------------------------------------ readings
def registered(D):
    """FACTORIAL's C0-C3 on one batch of n = 4 per cell (AF.catch_up's Fisher uses AF.N = 4)."""
    acc, sol, cls = D["acc"], D["solved"], D["cls"]
    assert all(len(acc[c]) == AF.N for c in CELLS), "the registered rules are written for n = 4 per cell"
    R = {"C0": AF.c0(acc["Path_L1"], acc["RoPE_L1"])}
    for cell, tag in (("RoPE_L4", "C1"), ("RoPE_Loop", "C2")):
        R[tag] = AF.catch_up(acc[cell], acc["Path_L1"], sum(sol[cell]), sum(sol["Path_L1"]), cell,
                             cls[cell].count("DESCENDING"))
        R[f"C3_{cell}"] = AF.helps(acc[cell], acc["RoPE_L1"], cell)
    return R


def head(label):
    """The label head: the text before the first ':' (e.g. 'INDEX (RoPE_L4) DOES NOT CATCH UP', 'UNMEASURED (n = 4)')."""
    return label.split(":", 1)[0].strip()


# FACTORIAL's registered labels (FACTORIAL_ANALYSIS.txt), by head
FACTORIAL_HEADS = {"C0": "PATH WINS AT ONE LAYER", "C1": "INDEX (RoPE_L4) DOES NOT CATCH UP",
                   "C2": "INDEX (RoPE_Loop) DOES NOT CATCH UP", "C3_RoPE_L4": "HELPS", "C3_RoPE_Loop": "HELPS"}


def replication(R):
    """Amendment 1 (A3). REPLICATION SUMMARY (catch-up readings C1, C2) -- a pure function of the registered labels, not a
    new test: REPLICATES iff C1 and C2 both read DOES NOT CATCH UP (FACTORIAL's reading); DOES NOT REPLICATE if either reads
    INDEX WITHIN 0.02 OF PATH; PARTIAL iff exactly one reads DOES NOT CATCH UP (and neither WITHIN); otherwise NOT SHOWN AT
    n = 4. Labels are matched on their head."""
    h = {k: head(R[k]) for k in ("C1", "C2")}
    nc = {k: h[k].startswith("INDEX (") and h[k].endswith("DOES NOT CATCH UP") for k in h}
    wi = {k: h[k].startswith("INDEX (") and "WITHIN 0.02 OF PATH" in h[k] for k in h}
    if all(nc.values()):
        return "REPLICATES (C1 and C2 both DOES NOT CATCH UP, as FACTORIAL)"
    if any(wi.values()):
        return "DOES NOT REPLICATE (index within 0.02 of path in " + " and ".join(k for k in wi if wi[k]) + ")"
    if sum(nc.values()) == 1:
        k1 = "C1" if nc["C1"] else "C2"; k2 = "C2" if k1 == "C1" else "C1"
        return f"PARTIAL ({k1} DOES NOT CATCH UP; {k2} {h[k2]})"
    return f"NOT SHOWN AT n = 4 (C1 {h['C1']}; C2 {h['C2']})"


def same_as_factorial(R):
    """Amendment 1 (A3), descriptive: does each of C0, C3 (RoPE_L4), C3 (RoPE_Loop) carry FACTORIAL's label head?"""
    return ", ".join(f"{k} {'yes' if head(R[k]) == FACTORIAL_HEADS[k] else 'no (' + head(R[k]) + ')'}"
                     for k in ("C0", "C3_RoPE_L4", "C3_RoPE_Loop"))


def pooled_line(a_idx, a_path, s_idx, s_path, name, desc):
    """FACTORIAL's branch rule applied to pooled n = 8 vs 8 -- DESCRIPTIVE (cross-batch), not a verdict."""
    n_i, n_p = len(a_idx), len(a_path)
    d = float(np.mean(a_path) - np.mean(a_idx)); p = perm2_p(a_idx, a_path)["p"]; pf = fisher_solved(s_path, n_p, s_idx, n_i)
    if np.mean(a_idx) >= np.mean(a_path) - 0.02 and s_idx >= 0.75 * n_i:
        lab = "would read WITHIN 0.02"
    elif (d >= 0.05 and p < .05) or (pf < .05 and s_path > s_idx):
        lab = "would read DOES NOT CATCH UP"
    else:
        lab = "would read UNMEASURED"
    return (f"[DESCRIPTIVE, cross-batch n = {n_i} vs {n_p}] {lab}: Path_L1 - {name} {d:+.4f} (perm p {p:.4f}); "
            f"SOLVED {s_idx}/{n_i} vs {s_path}/{n_p} (Fisher p {pf:.4f}); DESCENDING {desc}/{n_i}")


def r_loss_acc(D):
    x = np.array(sum((D["tail"][c] for c in CELLS), [])); y = np.array(sum((D["acc"][c] for c in CELLS), []))
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


def report(F, Rf, floors_f, floors_r=None, out=print):
    """F = fresh batch, Rf = FACTORIAL's batch (or None); floors_* = {seed: floors_stream(seed)}."""
    V = {"fresh": {}, "pooled": {}, "batch_shift": {}, "strata": {}, "retrace_margin": {}, "own_map": {}}
    out(f"== REGISTERED PRIMARY: fresh seeds {FRESH['seeds']} ALONE (n = 4 per cell; exact permutation and Fisher minimum "
        f"p 0.029 = complete separation only; rule 5) ==")
    for cell in CELLS:
        out(f"  {cell:9s} acc T=1024 {np.mean(F['acc'][cell]):.4f} (per seed {[round(x, 3) for x in F['acc'][cell]]}) | SOLVED "
            f"{sum(F['solved'][cell])}/4 | classes {F['cls'][cell]}")
    fl = np.array([[floors_f[s]["all"]["blank"], floors_f[s]["all"]["retrace"], floors_f[s]["wrap_share"]] for s in FRESH["seeds"]])
    out(f"  floors on these eval walks (measured, model-free): blank {fl[:, 0].mean():.4f}, retrace {fl[:, 1].mean():.4f} "
        f"(per seed {[round(x, 3) for x in fl[:, 1]]}), wrap-only share {fl[:, 2].mean():.3f}; best n-gram 0.576 (quoted, RANK_MI)")
    R = registered(F)
    out(f"  REGISTERED C0 (Path_L1 - RoPE_L1): {R['C0']}")
    out(f"  REGISTERED C1 (depth): {R['C1']}")
    out(f"  REGISTERED C3 (RoPE_L4): {R['C3_RoPE_L4']}")
    out(f"  REGISTERED C2 (loop): {R['C2']}")
    out(f"  REGISTERED C3 (RoPE_Loop): {R['C3_RoPE_Loop']}")
    rep = replication(R); out(f"  REPLICATION SUMMARY (catch-up readings C1, C2): {rep}")
    same = same_as_factorial(R); out(f"  same label as FACTORIAL (descriptive): {same}")
    V["fresh"] = {"registered": R, "replication": rep, "same_as_factorial": same, "acc": F["acc"], "solved": F["solved"], "cls": F["cls"]}

    out("== SECONDARY (declared, no verdict) ==")
    for cell in CELLS:                                                # accuracy minus the retrace floor on the SAME walks
        m = [a - floors_f[s]["all"]["retrace"] for a, s in zip(F["acc"][cell], FRESH["seeds"])]
        V["retrace_margin"][cell] = m
        out(f"  acc - retrace floor (same walks) {cell:9s}: {np.mean(m):+.4f} (per seed {[round(x, 3) for x in m]}; "
            f"{sum(x > 0 for x in m)}/4 above)")
    for cell in CELLS:
        rows = [x for x in F["strata"][cell] if x]
        if rows:
            mean = {st: float(np.mean([x[st] for x in rows if x[st] is not None])) for st in STRATA}
            fr = {st: float(np.mean([floors_f[s][st]["retrace"] for s in FRESH["seeds"]])) for st in STRATA}
            V["strata"][cell] = mean
            out(f"  strata {cell:9s}: " + ", ".join(f"{SNAME[st]} {mean[st]:.3f} (retrace {fr[st]:.3f})" for st in STRATA))
    for cell in CELLS:
        own = [x for x in F["own"][cell] if x is not None]
        if own:
            V["own_map"][cell] = own
            out(f"  own (training) map {cell:9s}: {np.mean(own):.4f} vs held-out {np.mean(F['acc'][cell]):.4f}")
    for cell in CELLS:
        out(f"  OOD (rule 10) {cell:9s}: T=512 {np.mean(F['acc512'][cell]):.4f}, T=2048 {np.mean(F['acc2048'][cell]):.4f}")
    out(f"  r(final-5% training loss, acc T=1024) over the 16 fresh runs: {r_loss_acc(F):+.3f}")

    if Rf is not None:
        out(f"== SECONDARY: pooled n = 8 with FACTORIAL's seeds {REF['seeds']} -- CROSS-BATCH, DESCRIPTIVE ONLY ({DIFF_NOTE}) ==")
        P = {k: {c: F[k][c] + Rf[k][c] for c in CELLS} for k in ("acc", "solved", "cls")}
        for cell in CELLS:
            out(f"  pooled {cell:9s} acc {np.mean(P['acc'][cell]):.4f} | SOLVED {sum(P['solved'][cell])}/8 | "
                f"fresh {np.mean(F['acc'][cell]):.4f} vs FACTORIAL {np.mean(Rf['acc'][cell]):.4f}")
        d0 = float(np.mean(P["acc"]["Path_L1"]) - np.mean(P["acc"]["RoPE_L1"])); p0 = perm2_p(P["acc"]["RoPE_L1"], P["acc"]["Path_L1"])["p"]
        out(f"  pooled C0: Path_L1 - RoPE_L1 {d0:+.4f} (perm p {p0:.5f}) [DESCRIPTIVE]")
        V["pooled"]["C0"] = {"d": d0, "p": p0}
        for cell, tag in (("RoPE_L4", "C1"), ("RoPE_Loop", "C2")):
            line = pooled_line(P["acc"][cell], P["acc"]["Path_L1"], sum(P["solved"][cell]), sum(P["solved"]["Path_L1"]),
                               cell, P["cls"][cell].count("DESCENDING"))
            out(f"  pooled {tag}: {line}")
            d3 = float(np.mean(P["acc"][cell]) - np.mean(P["acc"]["RoPE_L1"])); p3 = perm2_p(P["acc"]["RoPE_L1"], P["acc"][cell])["p"]
            out(f"  pooled C3 ({cell}): {cell} - RoPE_L1 {d3:+.4f} (perm p {p3:.5f}) [DESCRIPTIVE]")
            V["pooled"][tag] = line; V["pooled"][f"C3_{cell}"] = {"d": d3, "p": p3}
        for cell in CELLS:                                            # batch shift (rule 6: runs may not reproduce across batches)
            d = float(np.mean(F["acc"][cell]) - np.mean(Rf["acc"][cell])); p = perm2_p(Rf["acc"][cell], F["acc"][cell])["p"]
            V["batch_shift"][cell] = {"d": d, "p": p}
            out(f"  batch shift {cell:9s}: fresh - FACTORIAL {d:+.4f} (perm p {p:.4f}, n = 4 vs 4)")
        if floors_r:
            for cell in CELLS:
                m = [a - floors_r[s]["all"]["retrace"] for a, s in zip(Rf["acc"][cell], REF["seeds"])]
                out(f"  FACTORIAL acc - retrace floor (its own walks) {cell:9s}: {np.mean(m):+.4f} ({sum(x > 0 for x in m)}/4 above)")
    return V


# ------------------------------------------------------------------------------------------------ selftest
def _fake_batch(B, out_dir, acc, solved, desc=None, strata=True, own=True):
    """Synthetic checkpoints + eval / strata / own-map JSONs in the batch's real layout."""
    desc = desc or {}
    for cell, (root, var, nl) in CELLS.items():
        J = {f"0.0|{var}|{L}": [] for L in (512, 1024, 2048)}; S = {}
        n_desc = 0
        for i, s in enumerate(B["seeds"]):
            a = acc[cell][i]
            if solved[cell][i]:
                losses = list(np.linspace(1.0, 0.01, EPOCHS))
            elif n_desc < desc.get(cell, 0):
                n_desc += 1
                losses = list(np.linspace(1.0, 0.3, EPOCHS))         # still falling at the end: DESCENDING
            else:
                losses = [1.0] * 100 + [0.3] * (EPOCHS - 100)        # flat: STALLED
            args = {"variant": var, "seed": s, "n_layers": nl, "epochs": EPOCHS, "n_steps": 1024, "lr": 0.001, "fast_attn": True,
                    "deterministic": False, "device": "cuda:1", "output_dir": "x", "weight_decay": 0.05, "snapshot_every": 0}
            cfg = {"n_layers": nl, "n_steps": 1024, "epochs": EPOCHS, "fast_attn": True, "deterministic": False,
                   "grid_size": 64, "args": args}
            d = f"{B['runs']}/{root}/p0/{var}_s{s}"; os.makedirs(d, exist_ok=True)
            torch.save({"variant": var, "seed": s, "losses": losses, "config": cfg}, f"{d}/{var}.pt")
            for L in (512, 1024, 2048):
                J[f"0.0|{var}|{L}"].append([s, a if L == 1024 else a - 0.01, 0.5])
            S[f"{var}|{s}|1024"] = {"all": {"acc": a}, "plain_lag<128": {"acc": min(1, a + 0.05)},
                                    "plain_lag>=128": {"acc": a - 0.3}, "wrap": {"acc": a - 0.3}}
            if own:
                json.dump({f"0.0|{var}|1024": [[s, a + 0.01, 0.1]]}, open(f"{out_dir}/{B['tag']}_{root}_OWN_s{s}.json", "w"))
        old = json.load(open(f"{out_dir}/{B['tag']}_{root}.json")) if os.path.exists(f"{out_dir}/{B['tag']}_{root}.json") else {}
        old.update(J); json.dump(old, open(f"{out_dir}/{B['tag']}_{root}.json", "w"))
        if strata:
            olds = json.load(open(f"{out_dir}/{B['tag']}_{root}_STRATA.json")) if os.path.exists(f"{out_dir}/{B['tag']}_{root}_STRATA.json") else {}
            olds.update(S); json.dump(olds, open(f"{out_dir}/{B['tag']}_{root}_STRATA.json", "w"))


def selftest():
    ok = True

    def chk(cond, what):
        nonlocal ok
        ok &= bool(cond); print(f"  {'ok ' if cond else 'BAD'} {what}")

    # 1. FACTORIAL's registered analysis file is the one its batch ran (the decision rules are imported from it)
    want = dict(l.split()[::-1] for l in open(f"{REPO}/runs/factorial/code_md5.txt"))["analyze_factorial.py"]
    got = hashlib.md5(open(f"{REPO}/analyze_factorial.py", "rb").read()).hexdigest()
    chk(got == want, f"analyze_factorial.py md5 {got} == FACTORIAL's recorded {want}")
    chk(AF.smoke(), "analyze_factorial's own smoke test passes")

    # 2. the ported floors reproduce score_rank_floors_out.txt (seed 10, T = 1024) and eval_rank_strata's stratum counts
    f10 = floors_stream(10)
    chk(f10["all"]["n"] == 32255 and round(f10["all"]["blank"], 4) == 0.5049 and round(f10["all"]["retrace"], 4) == 0.8411
        and round(f10["wrap_share"], 3) == 0.081, f"floors s10 T=1024: n {f10['all']['n']} blank {f10['all']['blank']:.4f} "
        f"retrace {f10['all']['retrace']:.4f} wrap {f10['wrap_share']:.3f} (want 32255 / 0.5049 / 0.8411 / 0.081)")
    f500 = floors_stream(500); St = json.load(open(f"{REPO}/FACTORIAL_L1_STRATA.json"))["RoPE|500|1024"]
    chk(all(f500[k]["n"] == St[k]["n"] for k in ("all",) + STRATA),
        f"stratum counts s500 match FACTORIAL_L1_STRATA.json ({[f500[k]['n'] for k in ('all',) + STRATA]} vs "
        f"{[St[k]['n'] for k in ('all',) + STRATA]})")
    chk(abs(f500["all"]["blank"] - St["all"]["floor"]) < 1e-12, "blank floor s500 == strata JSON's constant floor")

    # 3. synthetic batches through every reading
    one = [1.0] * 4; low = [0.72, 0.73, 0.70, 0.74]; mid = [0.86, 0.87, 0.85, 0.88]
    T4, F4 = [True] * 4, [False] * 4
    hi = [0.99, 0.995, 0.99, 0.985]; mixed = [1.0, 0.80, 1.0, 0.75]; TFTF = [True, False, True, False]
    ALL_REG = ["PATH WINS AT ONE LAYER", "AS IT FALLS", "WITHIN 0.02 OF PATH", "DOES NOT CATCH UP", "UNMEASURED", "HELPS",
               "REPLICATES", "DOES NOT REPLICATE", "PARTIAL", "NOT SHOWN AT n = 4"]
    # name: (acc, solved, descending, readings that must appear, readings that must NOT appear in the registered lines)
    cases = {
        "replicates": ({"Path_L1": one, "RoPE_L1": low, "RoPE_L4": mid, "RoPE_Loop": mid},
                       {"Path_L1": T4, "RoPE_L1": F4, "RoPE_L4": F4, "RoPE_Loop": F4}, {},
                       ["PATH WINS AT ONE LAYER", "INDEX (RoPE_L4) DOES NOT CATCH UP", "INDEX (RoPE_Loop) DOES NOT CATCH UP",
                        "HELPS: RoPE_L4", "HELPS: RoPE_Loop", "SUMMARY (catch-up readings C1, C2): REPLICATES",
                        "still DESCENDING 0/4", "C0 yes, C3_RoPE_L4 yes, C3_RoPE_Loop yes"],
                       ["AS IT FALLS", "WITHIN 0.02 OF PATH", "UNMEASURED", "DOES NOT REPLICATE", "PARTIAL", "NOT SHOWN"]),
        "within": ({"Path_L1": one, "RoPE_L1": low, "RoPE_L4": hi, "RoPE_Loop": mid},
                   {"Path_L1": T4, "RoPE_L1": F4, "RoPE_L4": [True, True, True, False], "RoPE_Loop": F4}, {},
                   ["INDEX (RoPE_L4) WITHIN 0.02 OF PATH", "INDEX (RoPE_Loop) DOES NOT CATCH UP",
                    "DOES NOT REPLICATE (index within 0.02 of path in C1)",
                    "pooled C1: [DESCRIPTIVE, cross-batch n = 8 vs 8] would read DOES NOT CATCH UP"],
                   ["INDEX (RoPE_L4) DOES NOT CATCH UP", "UNMEASURED", "SUMMARY (catch-up readings C1, C2): REPLICATES",
                    "PARTIAL", "NOT SHOWN"]),
        "dncu_within": ({"Path_L1": one, "RoPE_L1": low, "RoPE_L4": mid, "RoPE_Loop": hi},
                        {"Path_L1": T4, "RoPE_L1": F4, "RoPE_L4": F4, "RoPE_Loop": [True, True, True, False]}, {},
                        ["INDEX (RoPE_L4) DOES NOT CATCH UP", "INDEX (RoPE_Loop) WITHIN 0.02 OF PATH",
                         "DOES NOT REPLICATE (index within 0.02 of path in C2)"],
                        ["SUMMARY (catch-up readings C1, C2): REPLICATES", "PARTIAL", "NOT SHOWN", "UNMEASURED"]),
        "partial": ({"Path_L1": one, "RoPE_L1": low, "RoPE_L4": mid, "RoPE_Loop": mixed},
                    {"Path_L1": T4, "RoPE_L1": F4, "RoPE_L4": F4, "RoPE_Loop": TFTF},
                    {"RoPE_Loop": 2}, ["UNMEASURED (n = 4)", "PARTIAL (C1 DOES NOT CATCH UP; C2 UNMEASURED (n = 4))",
                                       "RoPE_Loop still DESCENDING 2/4"],
                    ["SUMMARY (catch-up readings C1, C2): REPLICATES", "DOES NOT REPLICATE", "NOT SHOWN", "WITHIN 0.02"]),
        "both_unmeasured": ({"Path_L1": one, "RoPE_L1": low, "RoPE_L4": mixed, "RoPE_Loop": mixed},
                            {"Path_L1": T4, "RoPE_L1": F4, "RoPE_L4": TFTF, "RoPE_Loop": TFTF}, {},
                            ["NOT SHOWN AT n = 4 (C1 UNMEASURED (n = 4); C2 UNMEASURED (n = 4))"],
                            ["DOES NOT CATCH UP", "WITHIN 0.02", "SUMMARY (catch-up readings C1, C2): REPLICATES",
                             "DOES NOT REPLICATE", "PARTIAL"]),
        "fisher_only": ({"Path_L1": one, "RoPE_L1": low, "RoPE_L4": [0.97, 1.0, 0.95, 0.96], "RoPE_Loop": mid},
                        {"Path_L1": T4, "RoPE_L1": F4, "RoPE_L4": F4, "RoPE_Loop": F4}, {},
                        ["INDEX (RoPE_L4) DOES NOT CATCH UP: Path_L1 - RoPE_L4 +0.0300 (perm p 0.1429)", "Fisher p 0.0286"],
                        ["WITHIN 0.02", "UNMEASURED"]),
        "perm_only": ({"Path_L1": one, "RoPE_L1": low, "RoPE_L4": [0.90, 0.85, 0.88, 0.95], "RoPE_Loop": mid},
                      {"Path_L1": T4, "RoPE_L1": F4, "RoPE_L4": [False, False, False, True], "RoPE_Loop": F4}, {},
                      ["INDEX (RoPE_L4) DOES NOT CATCH UP: Path_L1 - RoPE_L4 +0.1050 (perm p 0.0286); SOLVED RoPE_L4 1/4 vs "
                       "Path_L1 4/4 (Fisher p 0.1429)"],
                      ["WITHIN 0.02", "UNMEASURED"]),
        "c0_c3_fall": ({"Path_L1": [1.0, 0.70, 1.0, 0.72], "RoPE_L1": low, "RoPE_L4": [0.71, 0.75, 0.73, 0.72],
                        "RoPE_Loop": [0.70, 0.76, 0.72, 0.73]},
                       {"Path_L1": TFTF, "RoPE_L1": F4, "RoPE_L4": F4, "RoPE_Loop": F4}, {},
                       ["REGISTERED C0 (Path_L1 - RoPE_L1): AS IT FALLS", "AS IT FALLS: RoPE_L4", "AS IT FALLS: RoPE_Loop",
                        "C0 no (AS IT FALLS), C3_RoPE_L4 no (AS IT FALLS), C3_RoPE_Loop no (AS IT FALLS)"],
                       ["PATH WINS", "HELPS", "SUMMARY (catch-up readings C1, C2): REPLICATES"]),
    }
    reached = set()
    with tempfile.TemporaryDirectory() as tmp:
        for name, (acc, sol, desc, wants, nots) in cases.items():
            d = f"{tmp}/{name}"; os.makedirs(d)
            fb = dict(FRESH, runs=f"{d}/runs_fresh"); rb = dict(REF, runs=f"{d}/runs_ref")
            _fake_batch(fb, d, acc, sol, desc)
            _fake_batch(rb, d, {"Path_L1": one, "RoPE_L1": low, "RoPE_L4": mid, "RoPE_Loop": mid},
                        {"Path_L1": T4, "RoPE_L1": F4, "RoPE_L4": F4, "RoPE_Loop": F4})
            lines = []
            Rf = load_batch(rb, d); Fr = load_batch(fb, d, ref_args=Rf["args"])
            fake_fl = {s: {"all": {"blank": 0.5, "retrace": 0.843, "n": 1}, "wrap_share": 0.08,
                           **{st: {"retrace": 0.9, "blank": 0.5, "n": 1} for st in STRATA}} for s in FRESH["seeds"] + REF["seeds"]}
            V = report(Fr, Rf, fake_fl, fake_fl, out=lines.append)
            txt = "\n".join(lines)
            reg = "\n".join(l for l in lines if "REGISTERED" in l or "REPLICATION SUMMARY" in l)   # the registered lines only
            for w in wants:
                chk(w in txt, f"[{name}] reading '{w}'")
            for w in nots:
                chk(w not in reg, f"[{name}] absent from the registered lines: '{w}'")
            reached |= {w for w in ALL_REG if w in reg}
            if name == "replicates":
                for w in ("pooled C1: [DESCRIPTIVE, cross-batch n = 8 vs 8] would read DOES NOT CATCH UP", "pooled C0",
                          "batch shift RoPE_L4", "strata RoPE_L4", "short-gap", "long-gap", "wrap-only",
                          "own (training) map RoPE_L4", "acc - retrace floor (same walks) RoPE_L4", "OOD (rule 10)",
                          "r(final-5% training loss", "FACTORIAL acc - retrace floor", "CROSS-BATCH, DESCRIPTIVE ONLY"):
                    chk(w in txt, f"[{name}] secondary '{w}'")
                chk(json.dumps(V, default=str) is not None, "verdict dict serialises")
        chk(reached == set(ALL_REG), f"every registered label reached (missing: {sorted(set(ALL_REG) - reached)})")
        # pooled WITHIN and UNMEASURED branches (both batches within 0.02; overlapping)
        chk("would read WITHIN 0.02" in pooled_line(hi + hi, one + one, 7, 8, "RoPE_L4", 0), "pooled rule reaches WITHIN 0.02")
        chk("would read UNMEASURED" in pooled_line(mixed + mixed, one + one, 4, 8, "RoPE_L4", 0), "pooled rule reaches UNMEASURED")
        # a flag drift against FACTORIAL's run is caught: a changed value, an added key, a removed key
        for kind in ("changed", "added", "removed"):
            d = f"{tmp}/drift_{kind}"; os.makedirs(d)
            fb = dict(FRESH, runs=f"{d}/runs_fresh"); rb = dict(REF, runs=f"{d}/runs_ref")
            base = ({"Path_L1": one, "RoPE_L1": low, "RoPE_L4": mid, "RoPE_Loop": mid},
                    {"Path_L1": T4, "RoPE_L1": F4, "RoPE_L4": F4, "RoPE_Loop": F4})
            _fake_batch(fb, d, *base); _fake_batch(rb, d, *base)
            p = f"{fb['runs']}/L4/p0/RoPE_s2800/RoPE.pt"; b = torch.load(p, weights_only=False); A = b["config"]["args"]
            if kind == "changed":
                A["lr"] = 0.003
            elif kind == "added":
                A["new_flag"] = 1
            else:
                del A["lr"]
            torch.save(b, p)
            try:
                load_batch(fb, d, ref_args=load_batch(rb, d)["args"]); chk(False, f"flag drift ({kind} key) caught")
            except AssertionError:
                chk(True, f"flag drift ({kind} key) against FACTORIAL's run is caught")
    print("SELFTEST", "ALL PASS" if ok else "FAILED"); return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-ref", action="store_true", help="skip the pooled / batch-shift secondaries")
    ap.add_argument("--floors-only", action="store_true", help="print the model-free floors on both batches' eval walks")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    floors_f = {s: floors_stream(s) for s in FRESH["seeds"]}
    floors_r = {s: floors_stream(s) for s in REF["seeds"]}
    if a.floors_only:
        for s, f in list(floors_f.items()) + list(floors_r.items()):
            print(f"  s{s}: n {f['all']['n']} blank {f['all']['blank']:.4f} retrace {f['all']['retrace']:.4f} wrap-only share "
                  f"{f['wrap_share']:.3f} | retrace by stratum " + ", ".join(f"{SNAME[st]} {f[st]['retrace']:.3f} (n {f[st]['n']})" for st in STRATA))
        return
    Rf = None if a.no_ref else load_batch(REF, REPO)
    F = load_batch(FRESH, REPO, ref_args=None if Rf is None else Rf["args"])
    V = report(F, Rf, floors_f, floors_r)
    V["floors"] = {"fresh": floors_f, "ref": floors_r}
    json.dump(V, open(f"{REPO}/{FRESH['tag']}_VERDICTS.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
