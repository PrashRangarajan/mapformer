"""Aggregate the filter x loop 2x2, with the verdict rules fixed before the runs.

The question: the InEKF's only established effect is at OOD LENGTH and the loop's
only cost is at OOD LENGTH. Does the filter repair the loop, or are they unrelated?

Primary measure is the per-seed interaction
    I = (Level15Looped - Looped) - (Level15 - Vanilla)
which is parameter-matched by construction (the loop adds 0 params on both rows;
the filter adds exactly 49,600 on both).

Rule 9 governs the analysis: over the L15 ablation's 30 runs, r(final training
loss, accuracy) was -0.930/-0.897/-0.812, and loss-matching flipped two readings in
OPPOSITE directions. So the loss-matched residual contrast is primary and the raw
contrast is printed beside it, whichever way each points.
"""
import argparse, json, os, re
import numpy as np

ARMS = ["Vanilla", "Level15", "Looped", "Level15Looped", "LoopedSampled"]
LENGTHS = [128, 512, 1024]


def final_loss(runs_dir, v, s):
    p = os.path.join(runs_dir, f"{v}_s{s}.log")
    if not os.path.exists(p):
        return None
    m = re.findall(r"Loss: ([0-9.]+)", open(p).read())
    return float(m[-1]) if m else None


def stat(d):
    """mean, sd, t, MDE, n for a vector of per-seed differences."""
    d = np.asarray([x for x in d if x is not None], dtype=float)
    n = len(d)
    if n < 2:
        return dict(m=float("nan"), sd=float("nan"), t=float("nan"),
                    mde=float("nan"), n=n, pos=0)
    sd = d.std(ddof=1); se = sd / np.sqrt(n)
    return dict(m=float(d.mean()), sd=float(sd),
                t=float(d.mean() / se) if se > 0 else float("nan"),
                mde=float(2.8 * sd / np.sqrt(n)), n=n, pos=int((d > 0).sum()))


def fmt(s):
    return (f"{s['m']:+.3f} (sd {s['sd']:.3f}, t {s['t']:+.2f}, "
            f"MDE {s['mde']:.3f}, {s['pos']}/{s['n']})")


def verdict(s):
    if not np.isfinite(s["m"]):
        return "NO DATA"
    if s["m"] > s["mde"]:
        return "DETECTABLE POSITIVE"
    if s["m"] < -s["mde"]:
        return "DETECTABLE NEGATIVE"
    return "UNMEASURED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = json.load(open(a.json))
    acc = {}                                   # (arm, T) -> {seed: acc}
    for k, rows in raw.items():
        _p, v, T = k.split("|")
        acc[(v, int(T))] = {int(s): x for s, x, _n in rows if x is not None}
    loss = {}
    for v in ARMS:
        for s in range(64):
            L = final_loss(a.runs_dir, v, s)
            if L is not None:
                loss[(v, s)] = L

    o = ["# Are the filter and the loop complementary?", "",
         "The 2x2 that had never been run. Clean torus paper task, 5 arms x 12 seeds,",
         "300 epochs warmup+cosine, held-out map, all arms trained in ONE batch.", "",
         "`Level15Looped` is verified to be bit-identical to `Level15` at n_loops=1",
         "(max|diff| 0.000e+00) and causal (leak 0.00e+00). The loop adds 0 parameters",
         "on both rows and the filter adds exactly 49,600 on both, so the interaction",
         "is parameter-matched; only the filter MAIN EFFECT carries a capacity gap.", ""]

    # ---- levels ------------------------------------------------------------
    o += ["## Accuracy (mean +/- sd)", "",
          "| arm | params | " + " | ".join(f"T={T}" for T in LENGTHS) + " |",
          "|---" * (len(LENGTHS) + 2) + "|"]
    P = {"Vanilla": "204,373", "Looped": "204,373", "LoopedSampled": "204,373",
         "Level15": "253,973", "Level15Looped": "253,973"}
    for v in ARMS:
        cells = []
        for T in LENGTHS:
            r = list(acc.get((v, T), {}).values())
            cells.append(f"{np.mean(r):.3f} +/- {np.std(r, ddof=1):.3f}" if len(r) > 1
                         else (f"{r[0]:.3f}" if r else "—"))
        o.append(f"| {v} | {P[v]} | " + " | ".join(cells) + " |")
    o.append("")

    # ---- convergence -------------------------------------------------------
    o += ["## Convergence (rule 9)", "",
          "| arm | mean final loss | per-seed range |", "|---|---|---|"]
    for v in ARMS:
        ls = [loss[(v, s)] for s in range(64) if (v, s) in loss]
        if ls:
            o.append(f"| {v} | {np.mean(ls):.4f} | {min(ls):.4f} – {max(ls):.4f} |")
    o.append("")
    for T in LENGTHS:
        ks = [(v, s) for v in ARMS for s in acc.get((v, T), {}) if (v, s) in loss]
        if len(ks) > 2:
            x = [loss[k] for k in ks]; y = [acc[(k[0], T)][k[1]] for k in ks]
            o.append(f"- r(final loss, accuracy) at T={T}: **{np.corrcoef(x, y)[0, 1]:+.3f}** "
                     f"over {len(ks)} runs")
    o.append("")

    # ---- contrasts, raw and loss-matched -----------------------------------
    def pair(v1, v2, T, resid=None):
        A, B = acc.get((v1, T), {}), acc.get((v2, T), {})
        ss = sorted(set(A) & set(B))
        if resid is None:
            return [A[s] - B[s] for s in ss]
        return [resid[(v1, s)] - resid[(v2, s)] for s in ss
                if (v1, s) in resid and (v2, s) in resid]

    def interaction(T, resid=None):
        need = [(v, T) for v in ("Level15Looped", "Looped", "Level15", "Vanilla")]
        ss = sorted(set.intersection(*[set(acc.get(k, {})) for k in need]))
        out = []
        for s in ss:
            if resid is None:
                out.append((acc[("Level15Looped", T)][s] - acc[("Looped", T)][s]) -
                           (acc[("Level15", T)][s] - acc[("Vanilla", T)][s]))
            elif all((v, s) in resid for v in ("Level15Looped", "Looped",
                                               "Level15", "Vanilla")):
                out.append((resid[("Level15Looped", s)] - resid[("Looped", s)]) -
                           (resid[("Level15", s)] - resid[("Vanilla", s)]))
        return out

    residual = {}
    for T in LENGTHS:
        ks = [(v, s) for v in ARMS for s in acc.get((v, T), {}) if (v, s) in loss]
        if len(ks) > 2:
            b = np.polyfit([loss[k] for k in ks],
                           [acc[(k[0], T)][k[1]] for k in ks], 1)
            residual[T] = {k: acc[(k[0], T)][k[1]] - np.polyval(b, loss[k]) for k in ks}

    tests = [("Looped", "Vanilla", "loop main effect (no filter)"),
             ("Level15", "Vanilla", "filter main effect (no loop)"),
             ("Level15Looped", "Looped", "filter, INSIDE the loop"),
             ("Level15Looped", "Level15", "loop, INSIDE the filter"),
             ("Level15Looped", "LoopedSampled", "the filter vs the free fix"),
             ("LoopedSampled", "Looped", "sampling the count (reference)")]

    o += ["## Contrasts", "",
          "Positive = first arm higher. DETECTABLE means |mean| > MDE = 2.8*sd/sqrt(n).", ""]
    for T in LENGTHS:
        o += [f"### T={T}", "",
              "| contrast | raw | loss-matched | |", "|---|---|---|---|"]
        for v1, v2, lab in tests:
            sr = stat(pair(v1, v2, T))
            sm = stat(pair(v1, v2, T, residual.get(T)))
            o.append(f"| {v1} - {v2} <br><sub>{lab}</sub> | {fmt(sr)} | {fmt(sm)} | "
                     f"{verdict(sm)} |")
        ir, im = stat(interaction(T)), stat(interaction(T, residual.get(T)))
        o += [f"| **INTERACTION** <br><sub>(L15Loop-Loop)-(L15-Vanilla)</sub> | "
              f"**{fmt(ir)}** | **{fmt(im)}** | **{verdict(im)}** |", ""]

    # ---- pre-registered verdict -------------------------------------------
    o += ["## Verdict against the pre-registration", ""]
    for T in (512, 1024):
        im = stat(interaction(T, residual.get(T)))
        ir = stat(interaction(T))
        v = verdict(im)
        o.append(f"**T={T}. Interaction {im['m']:+.3f} (loss-matched, MDE {im['mde']:.3f}); "
                 f"raw {ir['m']:+.3f} (MDE {ir['mde']:.3f}). -> {v}.**")
        if v == "DETECTABLE POSITIVE":
            o += ["", "COMPLEMENTARY. The filter buys more inside the loop than it buys "
                  "alone, i.e. the loop's OOD cost is specifically what the correction "
                  "repairs. Note this contradicts the mechanism argument written before "
                  "the run -- the loop's OOD damage was measured to be iteration count, "
                  "not theta drift -- so a positive here needs its own mechanism test "
                  "before it is explained, not just reported.", ""]
        elif v == "DETECTABLE NEGATIVE":
            o += ["", "THEY INTERFERE. Adding the filter to the loop costs more than the "
                  "filter gives on its own.", ""]
        else:
            best = {}
            for arm in ("Level15Looped", "Level15", "Looped"):
                r = list(acc.get((arm, T), {}).values())
                best[arm] = np.mean(r) if r else float("nan")
            add = (best["Level15Looped"] >= max(best["Level15"], best["Looped"]) - 1e-9)
            o += ["", f"NOT super-additive. Levels at T={T}: Level15Looped "
                  f"{best['Level15Looped']:.3f}, Level15 {best['Level15']:.3f}, "
                  f"Looped {best['Looped']:.3f}. " +
                  ("The combination is at least as good as either alone, so they are "
                   "ADDITIVE -- useful, but the filter is not repairing anything the "
                   "loop specifically broke."
                   if add else
                   "The combination is WORSE than the better single mechanism, so they "
                   "are not complementary in any useful sense.") +
                  " Per rule 11 an interaction inside its MDE is UNMEASURED, not zero.", ""]

    sm = stat(pair("Level15Looped", "LoopedSampled", 1024, residual.get(1024)))
    o += [f"**The filter vs the free fix at T=1024**: Level15Looped - LoopedSampled "
          f"{fmt(sm)} -> {verdict(sm)}. Sampling the loop count costs 0 parameters; "
          f"the filter costs 49,600. A filter that only matches sampling is not the "
          f"answer to the loop's length problem.", ""]

    o += ["## Scope", "",
          "Clean config only, n=12, one task, one width, one loop count (4). Noise was",
          "rejected as a condition here: at p=0.25 every arm sits within 0.11 of the",
          "0.500 blank floor at T=512, so an OOD interaction would be floor-compressed",
          "exactly where it needs measuring. The loss-matched analysis is a regression",
          "control, not a randomised one.", "",
          "One caveat on the reference arm: every looped model here is evaluated at 4",
          "passes, which is the consistent choice for the 2x2 but is NOT LoopedSampled's",
          "best count. Its own sweep peaks at 2 passes out of distribution (0.915 vs",
          "0.898 at T=512, 0.736 vs 0.719 at T=1024), so the free fix is understated by",
          "roughly 0.017 in the `Level15Looped - LoopedSampled` row. Small against the",
          "MDEs above, but it runs AGAINST sampling, so read that row as a lower bound",
          "on the free fix rather than a fair point estimate."]

    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
