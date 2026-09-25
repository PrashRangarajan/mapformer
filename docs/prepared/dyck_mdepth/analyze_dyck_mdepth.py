"""Readouts for DYCK_MDEPTH_PREREG.md -- the Dyck position effect at MATCHED nesting depth.

Every checkpoint is scored on the ladder's exact evaluation sequences (eval_dyck_literature.
cell_data, seeds 424242 + 1000 L + D) with A2f (dyck_mdepth_common), the registered primary,
and the ladder's plain A2 beside it. Runs are identified by CONTENT (the JSON's train_D /
train_D_set / n_sequences / rope_base are asserted), never by directory alone.

Run from /home/prashr:  python3 -m mapformer.analyze_dyck_mdepth [--out-stem PATH]
"""
import argparse, glob, json, math, os
import numpy as np
import torch

from mapformer.environment_dyck import DyckWorld
from mapformer.dyck_mdepth_common import CELLS, cell_with_mask, all_metrics, N
from mapformer.train_dyck import build
from mapformer.stats_core import mde, MDE_K, paired_p, signflip_p

REPO = "/home/prashr/mapformer"
RUNS = f"{REPO}/runs/dyck_mdepth"
LADDER = f"{REPO}/runs/dyck_ladder"
SEEDS = list(range(8))
PRIMARY_CELL = (32, 12)
KEY, KEY_OLD = "A2f_close_acc_dist_feasible", "A2_close_acc_dist"
# condition -> (sub-dir, depths, index arms, name suffix, expected JSON content)
COND = {
    "T12x3": ("T12x3", [4], ["RoPE", "PoPE", "RoPE_b32", "PoPE_b32"], "_tL32D12",
              dict(train_D=12, train_D_set=None, n_sequences=1_680_000)),
    "T12":   ("T12", [1, 2, 3, 4], ["RoPE", "PoPE"], "_tL32D12",
              dict(train_D=12, train_D_set=None, n_sequences=560_000)),
    "Tmix":  ("Tmix", [4], ["RoPE", "PoPE"], "_tL32Dset4-12",
              dict(train_D=None, train_D_set=list(range(4, 13)), n_sequences=560_000)),
}
PATH_ARMS = ["MapWM", "MapPoPE"]


def run_name(arm, L, suffix):
    arch, b32 = arm.replace("_b32", ""), arm.endswith("_b32")
    return f"{arch}-{L}L" + ("_r2" if arch.startswith("Map") else "") + ("_b32" if b32 else "") + suffix


def load_eval(path_pt, arm, L, data):
    arch, b32 = arm.replace("_b32", ""), arm.endswith("_b32")
    m = build(arch, 5, L, 2, 2, 32, rope_base=32.0 if b32 else None).cuda().eval()
    m.load_state_dict(torch.load(path_pt, map_location="cuda"))
    out = {}
    for c in CELLS:
        inp = data[c]["inp"]
        with torch.no_grad():
            lg = torch.cat([m(inp[i:i + 128].cuda()).float().cpu() for i in range(0, N, 128)])
        out[f"L{c[0]}D{c[1]}"] = all_metrics(lg.softmax(-1), data[c], lg)
    del m; torch.cuda.empty_cache()
    return out


def paired(d):
    """Every number a paired contrast is quoted with (exact-t MDE registered; house 2.8 for the ladder)."""
    d = np.asarray(d, float); n = d.size; sd = d.std(ddof=1)
    return dict(mean=float(d.mean()), sd=float(sd), n=n, pos=int((d > 0).sum()),
                mde_t=float(mde(sd, n)), mde_house=float(MDE_K * sd / math.sqrt(n)),
                p_t=float(paired_p(d)), p_signflip=float(signflip_p(d)["p"]))


def fmt(s):
    det = "DETECTABLE" if abs(s["mean"]) > s["mde_t"] else "within MDE"
    return (f"{s['mean']:+.3f} (MDE {s['mde_t']:.3f} exact-t / {s['mde_house']:.3f} house, {s['pos']}/{s['n']} +, "
            f"sign-flip p {s['p_signflip']:.4f}) {det}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-stem", default=f"{REPO}/DYCK_MDEPTH_RESULTS")
    ap.add_argument("--gates", default=f"{REPO}/DYCK_MDEPTH_GATES.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    ap.add_argument("--runs", default=RUNS, help="the batch directory (default runs/dyck_mdepth)")
    a = ap.parse_args()
    if os.path.exists(a.out_stem + ".md") and not a.force:
        raise SystemExit(f"{a.out_stem}.md exists; pass --force to overwrite")
    w = DyckWorld()
    data = {c: cell_with_mask(w, *c) for c in CELLS}
    R, J = {}, {}          # (cond, arm, L) -> {seed: metrics}, {seed: training json}

    # ---- the new batch, identified by content
    for cond, (sub, depths, idx_arms, suffix, want) in COND.items():
        for L in depths:
            for arm in idx_arms + PATH_ARMS:
                nm = run_name(arm, L, suffix)
                for s in SEEDS:
                    base = f"{a.runs}/{sub}/{nm}_s{s}/{nm}"
                    js = json.load(open(base + ".json"))
                    for k, v in want.items():
                        assert js.get(k) == v, f"{base}: {k}={js.get(k)} != {v}"
                    assert js["seed"] == s and js["n_layers"] == L, base
                    assert (js.get("rope_base") == 32.0) == arm.endswith("_b32"), base
                    R.setdefault((cond, arm, L), {})[s] = load_eval(base + ".pt", arm, L, data)
                    J.setdefault((cond, arm, L), {})[s] = js
                print(f"  {cond} {nm}: 8 seeds", flush=True)

    # ---- the ladder's D4-trained checkpoints, re-scored with A2f (cross-batch REFERENCE only)
    for L in (1, 2, 3, 4):
        for arm in ["RoPE", "PoPE"] + PATH_ARMS:
            nm = run_name(arm, L, "")
            for s in SEEDS:
                R.setdefault(("ladder_T4", arm, L), {})[s] = load_eval(f"{LADDER}/{nm}_s{s}/{nm}.pt", arm, L, data)
                J.setdefault(("ladder_T4", arm, L), {})[s] = json.load(open(f"{LADDER}/{nm}_s{s}/{nm}.json"))

    # ---- in-batch reproduction of two ladder runs (defaults = L32 D4): weights must be bitwise equal
    repro = {}
    for nm in ("RoPE-4L", "MapWM-4L_r2"):
        x = torch.load(f"{a.runs}/repro/{nm}_s0/{nm}.pt", map_location="cpu")
        y = torch.load(f"{LADDER}/{nm}_s0/{nm}.pt", map_location="cpu")
        jx = json.load(open(f"{a.runs}/repro/{nm}_s0/{nm}.json")); jy = json.load(open(f"{LADDER}/{nm}_s0/{nm}.json"))
        repro[nm] = dict(weights_differing=int(sum(int((x[k] != y[k]).sum()) for k in x)),
                         grid_equal=jx["grid"] == jy["grid"],
                         loss_curve_maxdiff=float(np.abs(np.array(jx["loss_curve"]) - np.array(jy["loss_curve"])).max()))

    v = lambda cond, arm, L, c=PRIMARY_CELL, k=KEY: np.array([R[(cond, arm, L)][s][f"L{c[0]}D{c[1]}"][k] for s in SEEDS])

    def position(cond, L, c=PRIMARY_CELL, k=KEY, rope="RoPE", pope="PoPE"):
        return ((v(cond, "MapWM", L, c, k) - v(cond, rope, L, c, k)) + (v(cond, "MapPoPE", L, c, k) - v(cond, pope, L, c, k))) / 2

    out = {"repro": repro, "cells": {}, "contrasts": {}, "convergence": {}}
    for (cond, arm, L), rs in R.items():
        for c in CELLS:
            for k in (KEY, KEY_OLD):
                x = v(cond, arm, L, c, k)
                out["cells"][f"{cond}|{arm}|{L}L|L{c[0]}D{c[1]}|{k}"] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)),
                                                                             per_seed=[float(z) for z in x])
    # ---- PRIMARY: 4 layers, 3x budget, trained at L32 D12, cell L32 D12, A2f
    P10k = paired(position("T12x3", 4))
    P32 = paired(position("T12x3", 4, rope="RoPE_b32", pope="PoPE_b32"))
    # registered: the index baseline per encoding is the base with the higher MEAN A2f (conservative for a positive claim)
    rope = max(("RoPE", "RoPE_b32"), key=lambda z: v("T12x3", z, 4).mean())
    pope = max(("PoPE", "PoPE_b32"), key=lambda z: v("T12x3", z, 4).mean())
    Peff = paired(position("T12x3", 4, rope=rope, pope=pope))
    out["contrasts"]["PRIMARY"] = dict(base10000=P10k, base32=P32, effective=Peff, index_arms_used=[rope, pope])
    if Peff["mean"] > Peff["mde_t"] and Peff["mean"] >= 0.10 and Peff["pos"] >= 7:
        verdict = ("SURVIVES: path integration beats the index arms at MATCHED depth (L32 D12 in the training "
                   "distribution), 4 layers, 3x budget, by >= 0.10 A2f -- a matched-distribution capability result.")
    elif Peff["mean"] <= Peff["mde_t"] or Peff["mean"] < 0.05:
        verdict = ("CLOSES: at matched depth the position effect at L32 D12 is below 0.05 A2f or within its MDE -- "
                   "the ladder's +0.168 was depth EXTRAPOLATION; the language line has no matched-distribution "
                   "positive result beyond the shrinking D4 training-cell effect.")
    else:
        verdict = "SHRINKS: detectable at matched depth but below 0.10 A2f; quote the matched-depth size, not +0.168."
    out["verdict"] = verdict

    # ---- secondaries
    out["contrasts"]["S1_T12_ladder"] = {f"{L}L": paired(position("T12", L)) for L in (1, 2, 3, 4)}
    out["contrasts"]["S1_index_3L_to_4L"] = {arm: paired(v("T12", arm, 4) - v("T12", arm, 3)) for arm in ("RoPE", "PoPE")}
    out["contrasts"]["S2_Tmix_4L"] = {f"L{c[0]}D{c[1]}": paired(position("Tmix", 4, c)) for c in CELLS}
    out["contrasts"]["S3_base_4L_x3"] = {z: paired(v("T12x3", z + "_b32", 4) - v("T12x3", z, 4)) for z in ("RoPE", "PoPE")}
    out["contrasts"]["S4_budget_4L"] = {arm: paired(v("T12x3", arm, 4) - v("T12", arm, 4)) for arm in ["RoPE", "PoPE"] + PATH_ARMS}
    idx_gain = paired((v("T12x3", "RoPE", 4) - v("T12", "RoPE", 4) + v("T12x3", "PoPE", 4) - v("T12", "PoPE", 4)) / 2)
    out["contrasts"]["S4_index_gain"] = idx_gain
    out["S4_budget_limited_1x"] = bool(idx_gain["mean"] > idx_gain["mde_t"] and idx_gain["pos"] >= 7)
    out["contrasts"]["REF_ladder_T4"] = {f"{L}L": paired(position("ladder_T4", L)) for L in (1, 2, 3, 4)}
    out["contrasts"]["REF_L128D12"] = {f"{cond}": paired(position(cond, 4, (128, 12))) for cond in ("T12x3", "T12", "ladder_T4")}
    for (cond, arm, L), js in J.items():
        fl = [js[s]["final_loss"] for s in SEEDS]
        floor = [js[s].get("train_ce_floor", 0.8005) for s in SEEDS]   # ladder JSONs predate the key: D4 floor
        out["convergence"][f"{cond}|{arm}|{L}L"] = dict(
            final_loss=float(np.mean(fl)), gap_to_floor=float(np.mean(np.subtract(fl, floor))),
            max_gap=float(np.max(np.subtract(fl, floor))),
            median_slope_per_1k=float(np.median([js[s]["final_slope_per_1k"] for s in SEEDS])))
    gates = json.load(open(a.gates)) if os.path.exists(a.gates) else None
    out["floors_A2f"] = gates["best_floor_A2f"] if gates else "GATES FILE MISSING"
    json.dump(out, open(a.out_stem + ".json", "w"), indent=1)

    # ---- markdown
    L_ = ["# Dyck position effect at MATCHED nesting depth -- results", "",
          "Pre-registration `DYCK_MDEPTH_PREREG.md`; gates `DYCK_MDEPTH_GATES.md`; runs `runs/dyck_mdepth`. "
          "Primary metric A2f (Hewitt distance-averaged closing accuracy over prefixes where the sampler can "
          "close; chance 0.500; CE-optimal predictor 1.000). Paired by seed, n=8, MDE exact-t (house 2.8 beside).", "",
          f"**Reproduction (in batch, ladder defaults L32 D4):** " + "; ".join(
              f"{k}: {r['weights_differing']} weights differ, eval grid equal {r['grid_equal']}, "
              f"loss curve max diff {r['loss_curve_maxdiff']:.1e}" for k, r in repro.items()), "",
          "## PRIMARY -- 4 layers, 3x budget, trained at L32 D12, scored at L32 D12 (A2f)", "",
          f"- index base 10000 (as the ladder): position main {fmt(P10k)}",
          f"- index base 32: position main {fmt(P32)}",
          f"- registered effective (stronger index base per encoding: {rope}, {pope}): **{fmt(Peff)}**",
          f"- best A2f floor at L32 D12 (gates, T12 fit): {out['floors_A2f']['T12']['L32D12'] if gates else 'n/a'}", "",
          f"**Verdict: {verdict}**", "",
          "## Arm means at L32 D12 (A2f / plain A2)", "",
          "| condition | depth | RoPE | PoPE | RoPE_b32 | PoPE_b32 | MapWM | MapPoPE |", "|---|---|---|---|---|---|---|---|"]
    for cond, depths in (("T12x3", [4]), ("T12", [1, 2, 3, 4]), ("Tmix", [4]), ("ladder_T4", [1, 2, 3, 4])):
        for Ld in depths:
            cells = []
            for arm in ("RoPE", "PoPE", "RoPE_b32", "PoPE_b32", "MapWM", "MapPoPE"):
                if (cond, arm, Ld) in R:
                    cells.append(f"{v(cond, arm, Ld).mean():.3f} / {v(cond, arm, Ld, k=KEY_OLD).mean():.3f}")
                else:
                    cells.append("--")
            L_.append(f"| {cond} | {Ld}L | " + " | ".join(cells) + " |")
    L_ += ["", "`ladder_T4` = the D4-trained ladder checkpoints re-scored on the same sequences (cross-batch "
           "reference: the depth-OOD effect).", "", "## Secondaries", "",
           "S1, depth ladder at matched depth (T12, 1x budget), position main at L32 D12:"]
    L_ += [f"- {k}: {fmt(s)}" for k, s in out["contrasts"]["S1_T12_ladder"].items()]
    L_ += ["", "S1, index arms 3L -> 4L (plateau check, T12 1x):"] + \
          [f"- {k}: {fmt(s)}" for k, s in out["contrasts"]["S1_index_3L_to_4L"].items()]
    L_ += ["", "S2, mixture training D in {4..12}, 4 layers, position main:"] + \
          [f"- {k}: {fmt(s)}" for k, s in out["contrasts"]["S2_Tmix_4L"].items()]
    L_ += ["", "S3, index base 32 minus base 10000 (4L, 3x, L32 D12):"] + \
          [f"- {k}: {fmt(s)}" for k, s in out["contrasts"]["S3_base_4L_x3"].items()]
    L_ += ["", "S4, budget 3x minus 1x at 4L (T12, L32 D12):"] + \
          [f"- {k}: {fmt(s)}" for k, s in out["contrasts"]["S4_budget_4L"].items()] + \
          [f"- index mean: {fmt(idx_gain)} -> 1x T12 ladder budget-limited: {out['S4_budget_limited_1x']}"]
    L_ += ["", "Reference, the ladder's D4-trained position main at L32 D12 re-scored with A2f:"] + \
          [f"- {k}: {fmt(s)}" for k, s in out["contrasts"]["REF_ladder_T4"].items()]
    L_ += ["", "Reference, length extrapolation L128 D12 (4L):"] + \
          [f"- {k}: {fmt(s)}" for k, s in out["contrasts"]["REF_L128D12"].items()]
    L_ += ["", "## Convergence (final loss = mean of the last 10% of the curve; floor = the training "
           "distribution's sampler entropy)", "", "| condition / arm / depth | final loss | gap to floor (mean / max) | median slope /1k |",
           "|---|---|---|---|"]
    for k, cv in out["convergence"].items():
        L_.append(f"| {k} | {cv['final_loss']:.4f} | {cv['gap_to_floor']:+.4f} / {cv['max_gap']:+.4f} | {cv['median_slope_per_1k']:+.5f} |")
    open(a.out_stem + ".md", "w").write("\n".join(L_) + "\n")
    print("\n".join(L_[:20]))


if __name__ == "__main__":
    main()
