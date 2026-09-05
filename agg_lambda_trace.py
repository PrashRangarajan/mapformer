"""Does the forget gate work through its TRAJECTORY rather than its endpoint?

Pre-registered in run_lambda_trace.sh: if it does, then across seeds
r(peak lambda, gain) > 0 while r(final lambda, gain) is the -0.516 already
measured. Rise-then-fall with a positive peak correlation supports it; monotone
drift, or a negative peak correlation, does not.
"""
import argparse, json, glob, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--gains-json", required=True)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    J = json.load(open(a.gains_json))
    acc = {v: {int(s): x for s, x, _ in J[f"0.0|{v}|{a.length}"]}
           for v in ("Vanilla", "Forget")}
    rows, peak, final, gain, shape = [], [], [], [], []
    for s in range(8):
        f = os.path.join(a.trace_dir, f"lam_s{s}.txt")
        if not os.path.exists(f) or s not in acc["Forget"]:
            continue
        d = np.loadtxt(f, ndmin=2)                     # step, lambda, E[sigmoid]
        step, lam, sig = d[:, 0], d[:, 1], d[:, 2]
        eff = lam * sig                                # decay per step
        i = int(np.argmax(np.abs(eff)))
        g = acc["Forget"][s] - acc["Vanilla"][s]
        # non-monotone = the peak is interior, not at the end
        interior = 0.02 < i / len(eff) < 0.95
        rows.append((s, eff[i], step[i] / step[-1], eff[-1], g, interior))
        peak.append(eff[i]); final.append(eff[-1]); gain.append(g)
        shape.append(interior)

    o = ["# Does the forget gate work through its trajectory?", "",
         "Effective decay per step is `lambda * E[sigmoid]`; lambda alone is not",
         "interpretable. `gain` is `Forget - Vanilla` at "
         f"T={a.length}, from the same seeds (the torus retrains bit-identically,",
         "so these runs reproduce the stored checkpoints exactly).", "",
         "| seed | peak decay | at frac of training | final decay | gain | peak interior |",
         "|---|---|---|---|---|---|"]
    for s, p, fr, fi, g, it in rows:
        o.append(f"| {s} | {p:+.5f} | {fr:.2f} | {fi:+.5f} | {g:+.3f} | "
                 f"{'yes' if it else 'no (monotone)'} |")
    if len(peak) > 2:
        rp = float(np.corrcoef(peak, gain)[0, 1])
        rf = float(np.corrcoef(final, gain)[0, 1])
        o += ["", f"- **r(peak decay, gain) = {rp:+.3f}**  (pre-registered: > 0)",
              f"- r(final decay, gain) = {rf:+.3f}",
              f"- peak is interior in **{sum(shape)}/{len(shape)}** seeds "
              f"(rise-then-fall rather than monotone drift)", "",
              "**Verdict.** " + (
                  "Supported: the peak predicts the gain where the endpoint does not, "
                  "and the trajectory is non-monotone."
                  if rp > 0 and sum(shape) > len(shape) / 2 else
                  "NOT supported. The transient-aid hypothesis does not survive its "
                  "own pre-registered test, and the mechanism behind the +0.081 "
                  "remains unidentified.")]
    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
