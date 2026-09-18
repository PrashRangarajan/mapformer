"""Analyse the Bach Chorales length-extrapolation 2x2 against JSBLEN_PREREG.md."""
import argparse, glob, json, math
import numpy as np

ORDER = ["RoPE", "PoPE", "MapWM_r2", "MapPoPE_r2"]
META = {"RoPE": ("index", "RoPE"), "PoPE": ("index", "PoPE"),
        "MapWM_r2": ("path integration", "RoPE"), "MapPoPE_r2": ("path integration", "PoPE")}
BUCK = ["0-512", "512-1024", "1024-2048"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    R = {}
    for f in sorted(glob.glob(f"{a.runs_dir}/*_s*/*.json")):
        j = json.load(open(f)); R.setdefault(j["name"], {})[j["seed"]] = j
    arms = [x for x in ORDER if x in R]
    g = lambda k, b: np.array([R[k][s]["test_buckets"][b] for s in sorted(R[k])])
    L = [f"# Bach Chorales, length extrapolation -- `{a.runs_dir}`", "",
         "Pre-registration: `JSBLEN_PREREG.md`. Trained on 512-token crops; test NLL by position "
         "bucket on whole pieces. **Lower is better.** 0-512 is in distribution.", "",
         "| arm | position | encoding | 0-512 | 512-1024 | 1024-2048 |", "|---|---|---|---|---|---|"]
    for k in arms:
        L.append(f"| {k} | {META[k][0]} | {META[k][1]} | " +
                 " | ".join(f"{g(k,b).mean():.4f} +/- {g(k,b).std(ddof=1):.4f}" for b in BUCK) + " |")
    L += ["", "## Paired contrasts per bucket (negative = better); MDE = 2.8 sd / sqrt(n)", "",
          "| contrast | " + " | ".join(BUCK) + " |", "|---|---|---|---|"]
    def con(x, y, b):
        ss = sorted(set(R[x]) & set(R[y]))
        d = np.array([R[x][s]["test_buckets"][b] - R[y][s]["test_buckets"][b] for s in ss])
        sd = d.std(ddof=1)
        return d.mean(), 2.8 * sd / math.sqrt(len(d)), int((d < 0).sum()), len(d)
    pairs = [("MapWM_r2", "RoPE", "R2 path integration, RoPE row"),
             ("MapPoPE_r2", "PoPE", "R2 path integration, PoPE row"),
             ("PoPE", "RoPE", "R3 encoding, index row"),
             ("MapPoPE_r2", "MapWM_r2", "R3 encoding, path-integrated row")]
    for x, y, lab in pairs:
        if x in R and y in R:
            cells = []
            for b in BUCK:
                d, mde, better, n = con(x, y, b)
                cells.append(f"{d:+.4f} (MDE {mde:.4f}, {better}/{n})" + (" **DET**" if abs(d) > mde else ""))
            L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "## Does the gap GROW with length? (contrast in 1024-2048 minus contrast in 0-512)", "",
          "| contrast | change | MDE | seeds growing | verdict |", "|---|---|---|---|---|"]
    for x, y, lab in pairs:
        if x in R and y in R:
            ss = sorted(set(R[x]) & set(R[y]))
            d = np.array([(R[x][s]["test_buckets"]["1024-2048"] - R[y][s]["test_buckets"]["1024-2048"]) -
                          (R[x][s]["test_buckets"]["0-512"] - R[y][s]["test_buckets"]["0-512"]) for s in ss])
            sd = d.std(ddof=1); mde = 2.8 * sd / math.sqrt(len(d))
            L.append(f"| {lab} | {d.mean():+.4f} | {mde:.4f} | {int((d<0).sum())}/{len(d)} | "
                     f"{'DETECTABLE' if abs(d.mean()) > mde else 'unmeasured'} |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
