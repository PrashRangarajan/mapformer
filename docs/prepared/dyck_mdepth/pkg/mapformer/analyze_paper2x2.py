"""PAPER2X2_PREREG.md analysis. Written before any arm is trained.

    python3 -m mapformer.analyze_paper2x2    (from /home/prashr)
"""
from __future__ import annotations

import json

import numpy as np

from mapformer.ckpt_guard import REPO, load_checkpoint, compare_checkpoints
from mapformer.stats_guard import from_diffs, rule9, table

SEEDS = list(range(8))
RUNS = REPO / "runs/paper2x2/p0"
ARMS = ["RoPE", "PoPE-Flat", "Vanilla", "MapPoPE-Flat", "Vanilla_r4", "MapPoPE_r4"]
PI = {2: ("Vanilla", "MapPoPE-Flat"), 4: ("Vanilla_r4", "MapPoPE_r4")}
LENGTHS = (128, 512, 1024)


def main():
    ev = json.load(open(REPO / "_PAPER2X2_RAW.json"))
    acc = {(a, T): {x[0]: float(x[1]) for x in ev[f"0.0|{a}|{T}"]} for a in ARMS for T in LENGTHS}
    loss = {a: {s: float(load_checkpoint(RUNS / f"{a}_s{s}" / f"{a}.pt").losses[-1]) for s in SEEDS}
            for a in ARMS}
    L = ["# PAPER2X2 results (PAPER2X2_PREREG.md)\n",
         "Torus paper task, held-out map (env-seed 10000), 300 ep cosine lr 1e-3, n=8 per arm, one batch.\n",
         "## Per arm\n", "| arm | T=128 | T=512 | T=1024 | final loss mean (range) |", "|---|---|---|---|---|"]
    for a in ARMS:
        cells = [np.array([acc[(a, T)][s] for s in SEEDS]) for T in LENGTHS]
        lo = np.array([loss[a][s] for s in SEEDS])
        L.append(f"| `{a}` | " + " | ".join(f"{c.mean():.3f} +/- {c.std(ddof=1):.3f}" for c in cells)
                 + f" | {lo.mean():.4f} ({lo.min():.4f}-{lo.max():.4f}) |")

    J = {}
    for T in LENGTHS:
        xs = [loss[a][s] for a in ARMS for s in SEEDS]; ys = [acc[(a, T)][s] for a in ARMS for s in SEEDS]
        r9 = rule9(ys, xs); fit = lambda x: r9.intercept + r9.slope * x
        res = {a: {s: acc[(a, T)][s] - fit(loss[a][s]) for s in SEEDS} for a in ARMS}
        L += ["", f"## T={T}\n", f"{r9}\n"]
        for r, (mw, mp) in PI.items():
            rows = []
            for tag, A in (("raw", {a: acc[(a, T)] for a in ARMS}), ("loss-matched", res)):
                pos = {s: 0.5 * ((A[mw][s] - A["RoPE"][s]) + (A[mp][s] - A["PoPE-Flat"][s])) for s in SEEDS}
                enc = {s: 0.5 * ((A["PoPE-Flat"][s] - A["RoPE"][s]) + (A[mp][s] - A[mw][s])) for s in SEEDS}
                inter = {s: (A[mp][s] - A["PoPE-Flat"][s]) - (A[mw][s] - A["RoPE"][s]) for s in SEEDS}
                cs = [from_diffs(pos, f"r={r} position main effect, {tag}"),
                      from_diffs(enc, f"r={r} encoding main effect, {tag}"),
                      from_diffs(inter, f"r={r} interaction, {tag}")]
                rows += cs
                J[f"T{T}|r{r}|{tag}"] = {c.label: [c.delta, c.mde, c.n_pos, c.verdict] for c in cs}
            L.append(table(rows)); L.append("")

    # registered verdicts
    p = J["T128|r2|raw"]["r=2 position main effect, raw"]
    p1024 = J["T1024|r2|raw"]["r=2 position main effect, raw"]
    e = J["T128|r2|raw"]["r=2 encoding main effect, raw"]
    if p[3] == "DETECTABLE" and p[0] > 0:
        h1 = "STANDS (>= +0.15)" if p[0] >= 0.15 else "STANDS AT REDUCED SIZE (< +0.15)"
    else:
        h1 = "NOT DETECTABLE -- headline at training length withdrawn"
    h2 = "MET" if (p[3] == "DETECTABLE" and p1024[3] == "DETECTABLE" and p1024[0] > p[0]) else "NOT MET"
    enc_big = any(J[f"T{T}|r2|raw"]["r=2 encoding main effect, raw"][3] == "DETECTABLE" and
                  abs(J[f"T{T}|r2|raw"]["r=2 encoding main effect, raw"][0]) >
                  J[f"T{T}|r2|raw"]["r=2 position main effect, raw"][0] for T in LENGTHS)
    h3 = ("FALSIFIED" if enc_big else
          ("MET" if not (e[3] == "DETECTABLE" and abs(e[0]) > 0.05) else "NOT MET (|encoding| > 0.05, detectable)"))
    L += ["## Registered verdicts\n",
          f"- **H1** position main effect, r=2, T=128, raw: {p[0]:+.3f} (MDE {p[1]:.3f}) -> **{h1}**",
          f"- **H2** grows with length: T=128 {p[0]:+.3f} -> T=1024 {p1024[0]:+.3f} -> **{h2}**",
          f"- **H3** encoding main effect small: {e[0]:+.3f} (MDE {e[1]:.3f}) -> **{h3}**", ""]

    L += ["## Determinism check against runs/sign (not replication, rule 27)\n", "```"]
    for a, stored in (("RoPE", "RoPE"), ("Vanilla_r4", "Vanilla_r4")):
        try:
            c = compare_checkpoints(RUNS / f"{a}_s0" / f"{a}.pt", REPO / f"runs/sign/p0/{stored}_s0/{stored}.pt")
            L.append(f"{a} s0: compared {c.n_compared}, differing {len(c.differing)}, losses equal {c.losses_equal}")
        except Exception as ex:  # noqa: BLE001
            L.append(f"{a} s0: comparison failed: {ex}")
    L.append("```")
    open(REPO / "PAPER2X2_RESULTS.md", "w").write("\n".join(L) + "\n")
    json.dump(J, open(REPO / "PAPER2X2_RAW_CONTRASTS.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
