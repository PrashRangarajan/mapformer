"""Analyse the Bach Chorales 2x2 against JSB_PREREG.md.
Run from /home/prashr: python3 -m mapformer.analyze_jsb --runs-dir <dir> --out <md>"""
import argparse, glob, json, math
import numpy as np

PAPER = {"RoPE": 0.5081, "PoPE": 0.4889}
ORDER = ["RoPE", "PoPE", "MapWM_r2", "MapPoPE_r2"]
META = {"RoPE": ("index", "RoPE"), "PoPE": ("index", "PoPE"),
        "MapWM_r2": ("path integration", "RoPE"), "MapPoPE_r2": ("path integration", "PoPE")}
KEY = "test_at_best_valid"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    R = {}
    for f in sorted(glob.glob(f"{a.runs_dir}/*_s*/*.json")):
        j = json.load(open(f)); R.setdefault(j["name"], {})[j["seed"]] = j
    arms = [x for x in ORDER if x in R]
    g = lambda k, key=KEY: np.array([R[k][s][key] for s in sorted(R[k])])
    L = [f"# Bach Chorales (PoPE paper Table 2) with path integration -- `{a.runs_dir}`", "",
         "Pre-registration: `JSB_PREREG.md`. NLL per token, **lower is better**. 'test at best valid' "
         "is the primary reading; 'best test' is the literal reading of the paper's wording.", "",
         "| arm | position | encoding | test at best valid | best test | final test | paper | seeds |",
         "|---|---|---|---|---|---|---|---|"]
    for k in arms:
        p = PAPER.get(k)
        L.append(f"| {k} | {META[k][0]} | {META[k][1]} | {g(k).mean():.4f} +/- {g(k).std(ddof=1):.4f} | "
                 f"{g(k,'best_test').mean():.4f} | {g(k,'final_test').mean():.4f} | "
                 f"{('%.4f' % p) if p else '--'} | {len(R[k])} |")
    L += ["", "## Contrasts (paired by seed, MDE = 2.8 sd / sqrt(n)); negative = better NLL", "",
          "| contrast | delta | sd | MDE | seeds better | verdict |", "|---|---|---|---|---|---|"]
    def con(x, y):
        ss = sorted(set(R[x]) & set(R[y]))
        d = np.array([R[x][s][KEY] - R[y][s][KEY] for s in ss])
        sd = d.std(ddof=1); return d.mean(), sd, 2.8 * sd / math.sqrt(len(d)), int((d < 0).sum()), len(d)
    for x, y, lab in [("PoPE", "RoPE", "R1 the paper's contrast (paper -0.0192)"),
                      ("MapWM_r2", "RoPE", "R2 path integration on the RoPE row"),
                      ("MapPoPE_r2", "PoPE", "R3 path integration on the PoPE row"),
                      ("MapPoPE_r2", "MapWM_r2", "encoding on the path-integrated row")]:
        if x in R and y in R:
            d, sd, mde, better, n = con(x, y)
            L.append(f"| {lab}: {x} - {y} | {d:+.4f} | {sd:.4f} | {mde:.4f} | {better}/{n} | "
                     f"{'DETECTABLE' if abs(d) > mde else 'unmeasured'} |")
    if all(k in R for k in ORDER):
        ss = sorted(set.intersection(*[set(R[k]) for k in ORDER]))
        it = np.array([(R["MapPoPE_r2"][s][KEY] - R["PoPE"][s][KEY]) -
                       (R["MapWM_r2"][s][KEY] - R["RoPE"][s][KEY]) for s in ss])
        sd = it.std(ddof=1)
        L.append(f"| R4 interaction | {it.mean():+.4f} | {sd:.4f} | {2.8*sd/math.sqrt(len(ss)):.4f} | "
                 f"{int((it<0).sum())}/{len(ss)} | "
                 f"{'DETECTABLE' if abs(it.mean()) > 2.8*sd/math.sqrt(len(ss)) else 'unmeasured'} |")
    if "RoPE" in R and "PoPE" in R:
        d = g("PoPE").mean() - g("RoPE").mean()
        ok = d < 0 and abs(d + 0.0192) <= 0.02 and abs(g("RoPE").mean() - 0.5081) <= 0.05 \
             and abs(g("PoPE").mean() - 0.4889) <= 0.05
        L += ["", "## Registered verdicts", "",
              f"- **R1 replication**: PoPE - RoPE = {d:+.4f} (paper -0.0192), levels "
              f"{g('RoPE').mean():.4f} / {g('PoPE').mean():.4f} (paper 0.5081 / 0.4889) -> "
              f"**{'REPLICATES' if ok else 'DOES NOT REPLICATE'}**"]
    # overfitting and budget checks
    L += ["", "## Overfitting and budget (rule 9 / rule 10)", "",
          "| arm | train loss at end | valid at end | test at end | best valid step (median) | at last step? |",
          "|---|---|---|---|---|---|"]
    for k in arms:
        js = list(R[k].values())
        bs = [min(j["history"], key=lambda h: h["valid"])["step"] for j in js]
        last = js[0]["history"][-1]["step"]
        L.append(f"| {k} | {np.mean([j['history'][-1]['train'] for j in js]):.4f} | "
                 f"{np.mean([j['history'][-1]['valid'] for j in js]):.4f} | "
                 f"{np.mean([j['history'][-1]['test'] for j in js]):.4f} | {int(np.median(bs))} | "
                 f"{sum(b == last for b in bs)}/{len(bs)} |")
    tl = np.array([j["history"][-1]["train"] for k in arms for j in R[k].values()])
    te = np.array([j[KEY] for k in arms for j in R[k].values()])
    L += ["", f"rule 9: r(final train loss, test NLL) = {np.corrcoef(tl, te)[0,1]:+.3f} over {len(tl)} runs"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
