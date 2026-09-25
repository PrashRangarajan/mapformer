"""Does the RoPE baseline's frequency schedule matter? Decide, then act.

The repo computes inv_freq_c = base^(-c/(n_b-1)); canonical RoPE is base^(-c/n_b).
If the difference is inside the noise floor at every length, the right move is to
switch the code to canonical and delete the discussion -- a documented quirk is
worse than either fixing it or leaving it alone. This script writes the verdict and
an explicit instruction, and reports both the t-rule and a sign test, because the
t>2.8 bar has already hidden one real effect in this project (23/24 seeds at
t = 2.23).
"""
import argparse, json, statistics as st
from math import comb
from pathlib import Path

LENS = [16, 32, 64, 128, 256]


def acc(runs, v, s, L):
    f = Path(runs) / f"{v}_s{s}" / f"{v}_parity.json"
    return json.load(open(f))["acc"].get(str(L)) if f.exists() else None


def sign_p(d):
    n = len(d); k = sum(1 for x in d if x > 0)
    k = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(16)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    o = ["# Does the RoPE baseline's frequency schedule matter?", "",
         "The repository computes `inv_freq_c = base^(-c/(n_b-1))`; canonical RoPE",
         "is `base^(-c/n_b)`. They agree to within 1% over the high-frequency blocks",
         "that resolve position at these lengths and differ by up to 25% at",
         "frequencies whose wavelength (47k-63k tokens) is DC over anything run",
         "here. Same task, same recipe, same batch, n=16.", "",
         "| length | repo | canonical | canonical - repo | t | seeds | sign test |",
         "|---|---|---|---|---|---|---|"]
    verdicts = []
    for L in LENS:
        A = [acc(a.runs_dir, "RoPE_Canonical", s, L) for s in a.seeds]
        B = [acc(a.runs_dir, "RoPE", s, L) for s in a.seeds]
        d = [x - y for x, y in zip(A, B) if x is not None and y is not None]
        if len(d) < 2:
            o.append(f"| {L} | — | — | — | — | — | — |"); continue
        m, sd = st.mean(d), st.stdev(d)
        mde = 2.8 * sd / len(d) ** 0.5
        t = m / (sd / len(d) ** 0.5) if sd else 0.0
        det = abs(m) > mde
        verdicts.append((L, m, mde, det, sign_p(d)))
        o.append(f"| {L} | {st.mean([x for x in B if x is not None]):.3f} | "
                 f"{st.mean([x for x in A if x is not None]):.3f} | "
                 f"**{m:+.4f}** (MDE {mde:.4f}) | {t:+.2f} | "
                 f"{sum(1 for x in d if x > 0)}/{len(d)} | p={sign_p(d):.3f} |")
    o.append("")

    any_det = any(v[3] for v in verdicts)
    any_sign = any(v[4] < 0.01 for v in verdicts)
    o += ["## Verdict", ""]
    if not any_det and not any_sign:
        o += ["**The schedule does not matter.** Every contrast is inside its MDE "
              "and no sign count is lopsided enough to matter either.", "",
              "**Action, per the pre-registration:**", "",
              "1. Switch `model_baseline_rope.py` to the canonical "
              "`base**(-k_idx / self.n_blocks)`, so the line stops contradicting "
              "the comment above it.",
              "2. Delete the canonical-vs-repo discussion from `mapformer_math.tex` "
              "and print the canonical formula plainly.",
              "3. Record in `CLAUDE.md` that RoPE runs before this date used the "
              "`n_b-1` schedule. `inv_freq` is a registered buffer, so stored "
              "checkpoints keep their own values on load and remain valid; only "
              "future runs change.", ""]
    else:
        o += ["**The schedule DOES matter — do not switch silently.**", "",
              "Detectable at: "
              + ", ".join(f"L={L} ({m:+.4f}, MDE {mde:.4f})"
                          for L, m, mde, det, _p in verdicts if det)
              + (" | lopsided sign counts at: "
                 + ", ".join(f"L={L} (p={p:.3f})" for L, _m, _e, _d, p in verdicts
                             if p < 0.01) if any_sign else ""), "",
              "The index baseline has been mildly mis-specified throughout. Before "
              "changing anything, re-derive which reported margins move and by how "
              "much: every position-effect number in this project is measured "
              "against this arm.", ""]
    o += ["## Scope", "",
          "One task, one width, n=16. The schedules differ most at the lowest "
          "frequencies, whose effect should grow with sequence length — L=256 is "
          "the most informative row and the one to weight."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
