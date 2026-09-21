"""Regenerates AUG_RESULTS.md (pitch-transposition augmentation on JSB).
Run from /home/prashr: python3 -m mapformer.analyze_aug"""
import glob, json, math
import numpy as np

R = "/home/prashr/mapformer/runs"
ARMS = [("PoPE", f"{R}/jsb/PoPE_s*/PoPE.json"), ("PoPE + aug", f"{R}/jsb_aug/PoPE_aug3_s*/PoPE_aug3.json"),
        ("MapPoPE", f"{R}/jsb/MapPoPE_r2_s*/MapPoPE_r2.json"),
        ("MapPoPE + aug", f"{R}/jsb_aug/MapPoPE_r2_aug3_s*/MapPoPE_r2_aug3.json")]


def main():
    A = {}
    print(f"{'arm':<16}{'test NLL':>10}{'best-val step':>15}{'n':>4}")
    for lab, pat in ARMS:
        d = {}
        for f in sorted(glob.glob(pat)):
            j = json.load(open(f)); d[j["seed"]] = j
        A[lab] = d
        t = np.array([d[s]["test_at_best_valid"] for s in d])
        bs = [min(d[s]["history"], key=lambda h: h["valid"])["step"] for s in d]
        print(f"{lab:<16}{t.mean():>10.4f}{int(np.median(bs)):>15}{len(d):>4}")
    def con(x, y):
        ss = sorted(set(A[x]) & set(A[y]))
        v = np.array([A[x][s]["test_at_best_valid"] - A[y][s]["test_at_best_valid"] for s in ss])
        mde = 2.8 * v.std(ddof=1) / math.sqrt(len(v))
        return f"{v.mean():+.4f} (MDE {mde:.4f}, better {int((v<0).sum())}/{len(v)})" + (" DET" if abs(v.mean()) > mde else "")
    print(f"\nA1 PoPE+aug - PoPE        {con('PoPE + aug','PoPE')}")
    print(f"A2 MapPoPE+aug - MapPoPE  {con('MapPoPE + aug','MapPoPE')}")
    print(f"A3 MapPoPE+aug - PoPE+aug {con('MapPoPE + aug','PoPE + aug')}")


if __name__ == "__main__":
    main()
