"""Omega-base sweep on Bach Chorales length extrapolation (JSBLEN_PREREG amendment 2)."""
import argparse, glob, json, math
import numpy as np

BUCK = ["0-512", "512-1024", "1024-2048"]
DIRS = {512: "runs/jsb_base512", 2048: "runs/jsb_len512", 8192: "runs/jsb_base8192",
        32768: "runs/jsb_base32768"}
REPO = "/home/prashr/mapformer"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    R = {}
    for base, d in DIRS.items():
        for arm in ("MapPoPE_r2", "MapWM_r2"):
            for f in sorted(glob.glob(f"{REPO}/{d}/{arm}_s*/{arm}.json")):
                j = json.load(open(f))
                if j.get("train_len") != 512 or j.get("base", base) != base:
                    continue   # base was not recorded in the earliest runs; the directory carries it
                R.setdefault((arm, base), {})[j["seed"]] = j["test_buckets"]
    bases = sorted({b for _, b in R})
    L = ["# Bach Chorales length extrapolation: omega-base sweep", "",
         "Pre-registration: `JSBLEN_PREREG.md` amendment 2. Trained on 512-token crops, test NLL by "
         "position bucket, **lower is better**. base 2048 is the original run.", ""]
    for arm in ("MapPoPE_r2", "MapWM_r2"):
        L += [f"## {arm}", "", "| omega base | " + " | ".join(BUCK) + " | seeds |",
              "|---|---|---|---|---|"]
        for b in bases:
            if (arm, b) not in R:
                continue
            v = R[(arm, b)]
            cells = [f"{np.mean([v[s][k] for s in v]):.4f} +/- {np.std([v[s][k] for s in v], ddof=1):.4f}"
                     for k in BUCK]
            L.append(f"| {b} | " + " | ".join(cells) + f" | {len(v)} |")
        L.append("")
    L += ["## Contrasts against base 2048, paired by seed (negative = better)", "",
          "| arm | base | " + " | ".join(BUCK) + " |", "|---|---|---|---|---|"]
    for arm in ("MapPoPE_r2", "MapWM_r2"):
        for b in bases:
            if b == 2048 or (arm, b) not in R or (arm, 2048) not in R:
                continue
            ss = sorted(set(R[(arm, b)]) & set(R[(arm, 2048)]))
            cells = []
            for k in BUCK:
                d = np.array([R[(arm, b)][s][k] - R[(arm, 2048)][s][k] for s in ss])
                mde = 2.8 * d.std(ddof=1) / math.sqrt(len(d))
                cells.append(f"{d.mean():+.4f} (MDE {mde:.4f}, {int((d<0).sum())}/{len(d)})" +
                             (" **DET**" if abs(d.mean()) > mde else ""))
            L.append(f"| {arm} | {b} | " + " | ".join(cells) + " |")
    L += ["", "## Does any base let MapPoPE beat MapWM out of distribution?", "",
          "| base | MapPoPE - MapWM at 1024-2048 | MDE | seeds better |", "|---|---|---|---|"]
    for b in bases:
        if ("MapPoPE_r2", b) in R and ("MapWM_r2", b) in R:
            ss = sorted(set(R[("MapPoPE_r2", b)]) & set(R[("MapWM_r2", b)]))
            d = np.array([R[("MapPoPE_r2", b)][s]["1024-2048"] - R[("MapWM_r2", b)][s]["1024-2048"] for s in ss])
            L.append(f"| {b} | {d.mean():+.4f} | {2.8*d.std(ddof=1)/math.sqrt(len(d)):.4f} | {int((d<0).sum())}/{len(d)} |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
