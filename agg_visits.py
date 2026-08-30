"""Aggregate the visits-per-cell test.

Three predictors were perfectly confounded across grid 8/16/32 at T=512:
    grid 8   46 distinct,  8.64 prior,  32 occupied  -> -0.010
    grid 16  96 distinct,  4.61 prior, 128 occupied  -> +0.015
    grid 32 158 distinct,  3.05 prior, 512 occupied  -> +0.374
Two new conditions break the confound by varying T at fixed grid, each MATCHED ON
DISTINCT CELLS VISITED with one already measured:
    A  grid 32, T=128    48 distinct, 1.95 prior, 512 occupied  (vs grid 8:  -0.010)
    B  grid 16, T=1024  153 distinct, 6.20 prior, 128 occupied  (vs grid 32: +0.374)

All four joint outcomes are written here before the runs finish.
"""
import argparse, json, os
import numpy as np

FLAT = 5e-4
FLOOR = 0.150
REF = {"A": ("grid 8, T=512", -0.010, "46 distinct, 8.64 prior, 32 occupied"),
       "B": ("grid 32, T=512", 0.374, "158 distinct, 3.05 prior, 512 occupied")}
DESC = {"A": ("grid 32, T=128", "48 distinct, 1.95 prior, 512 occupied", 128),
        "B": ("grid 16, T=1024", "153 distinct, 6.20 prior, 128 occupied", 1024)}


def cell(d, T, seeds=(0, 1, 2)):
    import torch
    E, F, rows = [], [], []
    for s in seeds:
        jv, jr = f"{d}/s{s}/Vanilla_oracle.json", f"{d}/s{s}/RoPE_oracle.json"
        if not (os.path.exists(jv) and os.path.exists(jr)):
            continue
        k = str(T)
        av, ar = json.load(open(jv)), json.load(open(jr))
        if k not in av or k not in ar:
            continue
        lv = torch.load(jv[:-5] + ".pt", map_location="cpu")["losses"]
        lr = torch.load(jr[:-5] + ".pt", map_location="cpu")["losses"]
        sl = lambda X: (X[int(.9*len(X)):][-1] - X[int(.9*len(X)):][0]) / max(1, len(X[int(.9*len(X)):]) - 1)
        e = av[k]["nb_acc"] - ar[k]["nb_acc"]
        E.append(e); F.append(abs(sl(lv)) < FLAT and abs(sl(lr)) < FLAT)
        rows.append((s, av[k]["nb_acc"], ar[k]["nb_acc"], e, lv[-1], lr[-1]))
    return np.array(E), F, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    o = ["# Does the position effect track VISITS PER CELL?", "",
         "Three predictors were perfectly confounded across grid 8/16/32 at T=512 --",
         "distinct cells visited, prior visits at a scored position, and map extent all",
         "move together. These two conditions vary T at fixed grid to break that, each",
         "matched on DISTINCT CELLS VISITED with a condition already measured.", "",
         "Prior-visit counts are MEASURED (probe_visits_per_cell.py), not inferred from",
         "T/n_occupied -- the walk is directed, so realised counts are 8.64/4.61/3.05",
         "where the arithmetic predicted 16/4/1.", "",
         "400 epochs, warmup+cosine, fast-attn (licensed: +0.392 vs +0.374 reference),",
         "n=3, trained AND evaluated at the training length.", "",
         "| cond | config | statistics | matched against | reference | effect | per-seed | flat |",
         "|---|---|---|---|---|---|---|---|"]
    res = {}
    for key, sub in (("A", "A_g32_T128"), ("B", "B_g16_T1024")):
        nm, stats, T = DESC[key]
        rnm, rval, rstats = REF[key]
        E, F, rows = cell(os.path.join(a.runs_dir, sub), T)
        if not len(E):
            o.append(f"| {key} | {nm} | {stats} | {rnm} | {rval:+.3f} | — | missing | — |")
            continue
        res[key] = E
        o.append(f"| {key} | {nm} | {stats} | {rnm} | {rval:+.3f} | **{E.mean():+.3f}** | "
                 f"{', '.join(f'{x:+.3f}' for x in E)} | {sum(F)}/{len(F)} |")
    o += ["", f"Measured noise floor: **{FLOOR:.3f}**. 'large' below means above it.", ""]

    o += ["## Verdict", ""]
    if len(res) < 2:
        o += ["**INCOMPLETE** -- both conditions are needed; neither alone separates the "
              "three predictors."]
    else:
        A, B = res["A"].mean(), res["B"].mean()
        aL, bL = abs(A) > FLOOR, abs(B) > FLOOR
        if not aL and bL:
            o += ["**DISTINCT CELLS VISITED drives it. The visits-per-cell hypothesis is "
                  "DEAD.** A has the lowest prior-visit count anywhere in this study (1.95) "
                  "and shows nothing; B has more than triple that (6.20) and shows the full "
                  "effect. Map extent is ruled out in the same stroke -- A sits on the LARGE "
                  "map (512 occupied) and B on the small one (128). The single factor that "
                  "tracks the effect is how many distinct cells the agent actually visits."]
        elif aL and not bL:
            o += ["**Visits-per-cell SURVIVES, and so does map extent -- this pair cannot "
                  "separate them.** A (prior 1.95, map 512) is large and B (prior 6.20, map "
                  "128) is not, and those two predictors point the same way in both cells. "
                  "Distinct cells visited IS ruled out: B has 153, essentially the 158 that "
                  "produced +0.374, and shows nothing. A third condition varying prior "
                  "visits at fixed map extent is needed."]
        elif not aL and not bL:
            o += ["**No single factor survives.** Both new conditions are null, yet grid 32 "
                  "at T=512 (158 distinct, 3.05 prior, 512 occupied) gives +0.374. B matches "
                  "it on distinct cells and shows nothing; A matches its map extent and "
                  "shows nothing. So no ONE of the three is sufficient -- the effect needs a "
                  "conjunction. Report the four cells and claim nothing more."]
        else:
            o += ["**Both new conditions are large, which no single predictor explains.** A "
                  "(48 distinct) and B (153 distinct) are both large while grid 8 (46 "
                  "distinct) and grid 16 (96) were both null, so distinct cells cannot be "
                  "it; A and B sit at opposite ends of BOTH prior visits and map extent, so "
                  "neither of those can be it either. Something varying with T itself -- "
                  "sequence length, or total training tokens per episode -- is the "
                  "remaining candidate, and it was not controlled here."]
        o += ["", f"(A {A:+.3f} vs its reference {REF['A'][1]:+.3f}; "
                  f"B {B:+.3f} vs its reference {REF['B'][1]:+.3f}.)"]

    o += ["", "## Scope", "",
          "Varying T changes sequence length as well as visit statistics, so 'T itself' is",
          "an uncontrolled alternative in every cell here. The design controls it only by",
          "matching each new condition to a reference on distinct cells; it does not",
          "eliminate it. n=3 per condition."]
    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
