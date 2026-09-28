"""Readouts for SIGN_MATCHED_PREREG.md."""
import json
import numpy as np
import torch

from mapformer.stats_core import classify_run, perm2_p, fisher_solved, mde

REPO = "/home/prashr/mapformer"; S = list(range(8)); R = f"{REPO}/runs/sign_matched/p0"
ARMS = ["Signed_r4", "Abs_r4", "Pos_r4", "RoPE"]


def main():
    J = json.load(open(f"{REPO}/SIGN_MATCHED.json"))
    acc = {L: {v: [dict((x[0], x[1]) for x in J[f"0.0|{v}|{L}"])[s] for s in S] for v in ARMS} for L in (1024, 2048)}
    solved, tail = {}, {}
    print("== run classes (SOLVED = final-5% loss < 0.05) ==")
    for v in ARMS:
        cl = [classify_run(torch.load(f"{R}/{v}_s{s}/{v}.pt", map_location="cpu", weights_only=False)["losses"]) for s in S]
        solved[v] = sum(c["registered"] == "SOLVED" for c in cl); tail[v] = [c["tail"] for c in cl]
        print(f"  {v:10s} SOLVED {solved[v]}/8  acc@1024 {np.mean(acc[1024][v]):.3f} +/- {np.std(acc[1024][v], ddof=1):.3f}"
              f"  acc@2048 {np.mean(acc[2048][v]):.3f}  " + " ".join(f"{c['registered'][:4]}({c['tail']:.3f})" for c in cl))
    x = np.concatenate([tail[v] for v in ARMS]); y = np.concatenate([acc[1024][v] for v in ARMS])
    print(f"\n  r(final loss, acc@1024) over 32 runs: {np.corrcoef(x, y)[0, 1]:+.3f}")
    print("\n== contrasts (a - b) ==")
    verdict = None
    for a, b, L, prim in [("Abs_r4", "Signed_r4", 1024, True), ("Pos_r4", "Signed_r4", 1024, False),
                          ("Signed_r4", "RoPE", 1024, False), ("Abs_r4", "Signed_r4", 2048, False),
                          ("Pos_r4", "Signed_r4", 2048, False), ("Signed_r4", "RoPE", 2048, False)]:
        d = np.mean(acc[L][a]) - np.mean(acc[L][b]); pp = perm2_p(acc[L][b], acc[L][a])["p"]
        pf = fisher_solved(solved[b], 8, solved[a], 8) if L == 1024 else float("nan")
        sd = np.sqrt((np.var(acc[L][a], ddof=1) + np.var(acc[L][b], ddof=1)) / 2)
        M = mde(sd, 8) * np.sqrt(2)  # two independent arms
        fires = pp < 0.05 or (L == 1024 and pf < 0.05)
        print(f"  {'PRIMARY ' if prim else ''}{a} - {b} @T={L}: {d:+.3f} (MDE ~{M:.3f}) perm p {pp:.4f}"
              f"{f' | SOLVED {solved[a]}/8 vs {solved[b]}/8 Fisher p {pf:.4f}' if L == 1024 else ''} | "
              f"{'FIRES' if fires else 'UNMEASURED'}")
        if prim:
            if fires and d < 0:
                verdict = "SIGN IS CAPABILITY"
            elif not fires and abs(d) < M and solved[a] >= 6:
                verdict = "SIGN IS ROBUSTNESS"
            else:
                verdict = "UNMEASURED"
    print(f"\n== REGISTERED VERDICT: {verdict}")


if __name__ == "__main__":
    main()
