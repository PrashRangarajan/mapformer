"""Aggregate the sign ablation (axis A5), verdicts fixed in SIGN_ABLATION_PREREG.md.

PRIMARY CONTRAST: Abs_r4 - Signed_r4. Both build action_to_lie the same way and
draw the same RNG; Signed_r4 is verified bit-identical to Vanilla_r4 on shared
weights (max|diff| 0.0). They differ by one call to .abs(). Parameter count is
identical (204,757) across every MapFormer arm.

Rule 9: r(final loss, accuracy) is reported per length and the loss-matched
residual contrast is printed beside the raw one, whichever way each points.
Rule 11: anything inside its MDE is UNMEASURED, never a null.
"""
import argparse, json, os, re
import numpy as np

ARMS = ["Signed_r4", "Vanilla_r4", "Abs_r4", "Pos_r4", "CARoPE_r4", "RoPE"]
LENGTHS = [128, 512, 1024]


def final_loss(runs_dir, v, s):
    p = os.path.join(runs_dir, f"{v}_s{s}.log")
    if not os.path.exists(p):
        return None
    m = re.findall(r"Loss: ([0-9.]+)", open(p).read())
    return float(m[-1]) if m else None


def stat(d):
    d = np.asarray([x for x in d if x is not None], dtype=float)
    n = len(d)
    if n < 2:
        return dict(m=float("nan"), sd=float("nan"), t=float("nan"),
                    mde=float("nan"), n=n, neg=0)
    sd = d.std(ddof=1); se = sd / np.sqrt(n)
    return dict(m=float(d.mean()), sd=float(sd),
                t=float(d.mean() / se) if se > 0 else float("nan"),
                mde=float(2.8 * sd / np.sqrt(n)), n=n, neg=int((d < 0).sum()))


def fmt(s):
    return (f"{s['m']:+.3f} (sd {s['sd']:.3f}, t {s['t']:+.2f}, "
            f"MDE {s['mde']:.3f}, {s['neg']}/{s['n']} neg)")


def verdict(s):
    if not np.isfinite(s["m"]):
        return "NO DATA"
    if s["m"] < -s["mde"]:
        return "**DETECTABLE NEGATIVE**"
    if s["m"] > s["mde"]:
        return "DETECTABLE POSITIVE"
    return "UNMEASURED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = json.load(open(a.json))
    acc = {}
    for k, rows in raw.items():
        _p, v, T = k.split("|")
        acc[(v, int(T))] = {int(s): x for s, x, _n in rows if x is not None}
    loss = {(v, s): L for v in ARMS for s in range(64)
            if (L := final_loss(a.runs_dir, v, s)) is not None}

    o = ["# The sign ablation: may the phase increment be negative?", "",
         "Axis A5 of the relational map -- the one axis no paper varies deliberately.",
         "Every content-dependent-phase mechanism outside MapFormer is non-negative,",
         "and in every case that is a side-effect of a squashing function: CARoPE's",
         "`1/(softplus+1)` lands in the OPEN interval (0,1), GRAPE-AP requires",
         "`omega = g(x) >= 0`, CoPE's gate is a sigmoid. MapFormer's `Delta` is signed",
         "only because nothing squashes it.", "",
         "**Mechanism under test.** Non-negativity makes the per-channel angle",
         "monotone in t. Monotone is all a language clock needs; a cognitive map needs",
         "east and west to cancel. A monotone code can encode `(n_E, n_W)` but not",
         "`n_E - n_W`, and revisit retrieval needs the latter.", "",
         "Clean torus paper task, held-out map (env-seed 10000), one batch, 300 epochs",
         "warmup+cosine at lr 1e-3. Parameter count identical across all five",
         "MapFormer arms; the constraint adds nothing and does not change the rank.", ""]

    o += ["## Accuracy (mean +/- sd)", "",
          "| arm | Delta | " + " | ".join(f"T={T}" for T in LENGTHS) + " |",
          "|---" * (len(LENGTHS) + 2) + "|"]
    DESC = {"Signed_r4": "`W_out W_in x` (baseline)",
            "Vanilla_r4": "`W_out W_in x` (RNG control)",
            "Abs_r4": "`\\|W_out W_in x\\|` **primary**",
            "Pos_r4": "`softplus(.)` GRAPE-AP",
            "CARoPE_r4": "`1/(softplus(.)+1)` CARoPE",
            "RoPE": "index (floor)"}
    for v in ARMS:
        cells = []
        for T in LENGTHS:
            r = list(acc.get((v, T), {}).values())
            cells.append(f"{np.mean(r):.3f} +/- {np.std(r, ddof=1):.3f}" if len(r) > 1
                         else (f"{r[0]:.3f}" if r else "—"))
        o.append(f"| `{v}` | {DESC[v]} | " + " | ".join(cells) + " |")
    o.append("")

    o += ["## Convergence (rule 9)", "",
          "| arm | mean final loss | per-seed range |", "|---|---|---|"]
    for v in ARMS:
        ls = [loss[(v, s)] for s in range(64) if (v, s) in loss]
        if ls:
            o.append(f"| `{v}` | {np.mean(ls):.4f} | {min(ls):.4f} – {max(ls):.4f} |")
    o.append("")
    for T in LENGTHS:
        ks = [(v, s) for v in ARMS for s in acc.get((v, T), {}) if (v, s) in loss]
        if len(ks) > 2:
            x = [loss[k] for k in ks]; y = [acc[(k[0], T)][k[1]] for k in ks]
            o.append(f"- r(final loss, accuracy) at T={T}: **{np.corrcoef(x, y)[0,1]:+.3f}** "
                     f"over {len(ks)} runs")
    o.append("")

    residual = {}
    for T in LENGTHS:
        ks = [(v, s) for v in ARMS for s in acc.get((v, T), {}) if (v, s) in loss]
        if len(ks) > 2:
            b = np.polyfit([loss[k] for k in ks],
                           [acc[(k[0], T)][k[1]] for k in ks], 1)
            residual[T] = {k: acc[(k[0], T)][k[1]] - np.polyval(b, loss[k]) for k in ks}

    def pair(v1, v2, T, resid=None):
        A, B = acc.get((v1, T), {}), acc.get((v2, T), {})
        ss = sorted(set(A) & set(B))
        if resid is None:
            return [A[s] - B[s] for s in ss]
        return [resid[(v1, s)] - resid[(v2, s)] for s in ss
                if (v1, s) in resid and (v2, s) in resid]

    tests = [("Abs_r4", "Signed_r4", "PRIMARY -- sign removed, nothing else"),
             ("Pos_r4", "Signed_r4", "GRAPE-AP style (init confound)"),
             ("CARoPE_r4", "Signed_r4", "CARoPE verbatim (init confound)"),
             ("Vanilla_r4", "Signed_r4", "RNG/construction control -- expect ~0"),
             ("Abs_r4", "RoPE", "does the monotone clock beat an index code?"),
             ("Signed_r4", "RoPE", "the position effect, for scale")]

    o += ["## Contrasts", "",
          "Negative = first arm WORSE. DETECTABLE means |mean| > MDE = 2.8*sd/sqrt(n).", ""]
    for T in LENGTHS:
        o += [f"### T={T}", "",
              "| contrast | raw | loss-matched | verdict |", "|---|---|---|---|"]
        for v1, v2, lab in tests:
            sr = stat(pair(v1, v2, T))
            sm = stat(pair(v1, v2, T, residual.get(T)))
            o.append(f"| `{v1}` - `{v2}` <br><sub>{lab}</sub> | {fmt(sr)} | {fmt(sm)} | "
                     f"{verdict(sm)} |")
        o.append("")

    # ---- the pre-registered verdict ---------------------------------------
    o += ["## Verdict against the pre-registration", ""]
    at = {T: (stat(pair("Abs_r4", "Signed_r4", T)),
              stat(pair("Abs_r4", "Signed_r4", T, residual.get(T)))) for T in LENGTHS}
    train_hit = at[128][1]["m"] < -at[128][1]["mde"]
    ood_hit = any(at[T][1]["m"] < -at[T][1]["mde"] for T in (512, 1024))
    if train_hit and ood_hit:
        v = ("**H1 CONFIRMED as predicted.** The deficit is present at TRAINING length "
             "and at OOD length. That is the representational signature: a monotone "
             "clock cannot encode net displacement, so it fails where the code is "
             "formed, not only where it is extrapolated.")
    elif ood_hit:
        v = ("**H1 fires at OOD LENGTH ONLY -- NOT the predicted result.** The "
             "pre-registration names this case explicitly: 'helps at OOD length' is "
             "this project's universal signature (rank, the InEKF, the forget gate and "
             "PoPE all show it and nothing explains it). An OOD-only sign effect is "
             "another instance of that unexplained axis, NOT evidence for the "
             "net-displacement mechanism.")
    elif train_hit:
        v = ("**H1 fires at training length only.** Unanticipated; report the levels "
             "and the degradation slopes and do not claim the mechanism.")
    else:
        v = ("**H1 REFUTED at this n.** A monotone clock does the torus task as well "
             "as a signed one at every length measured. Axis A5 is not the opening the "
             "relational map suggested. Report as refuted, not as unmeasured, only if "
             "the MDEs are smaller than the effect being dismissed -- they are printed "
             "above.")
    o += [v, ""]
    ctl = stat(pair("Vanilla_r4", "Signed_r4", 512))
    o += [f"**Control.** `Vanilla_r4 - Signed_r4` at T=512: {fmt(ctl)} -> "
          f"{verdict(ctl)}. These two arms are mathematically identical and differ "
          f"only in how many draws they take from the RNG, so anything other than "
          f"UNMEASURED here means the batch is not readable.", ""]

    o += ["## Degradation with length (H3, reported not adjudicated)", "",
          "| arm | T=128 | T=1024 | drop |", "|---|---|---|---|"]
    for v in ARMS:
        a1 = list(acc.get((v, 128), {}).values()); a2 = list(acc.get((v, 1024), {}).values())
        if a1 and a2:
            o.append(f"| `{v}` | {np.mean(a1):.3f} | {np.mean(a2):.3f} | "
                     f"{np.mean(a2)-np.mean(a1):+.3f} |")
    o.append("")
    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
