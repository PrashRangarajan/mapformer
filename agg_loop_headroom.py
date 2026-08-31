"""Does a loop help where path integration is NOT sufficient?

The looped pilot found recursion substitutes for depth in the index arm (+0.363,
sd 0.018, 3/3) but added nothing on top of path integration (+0.046, MDE 0.120,
one seed negative). That null was uninterpretable: MapFormer already scored 0.948
on the torus with 0.052 of headroom. Match-Query at 128^2 leaves 0.177, so the
loop has room to show an effect if it has one.

Verdict rules hard-coded before the runs finish.
"""
import argparse, json, os
import numpy as np

MDE_K = 2.8
LBL = {"PI_flat":  ("path integration, no loop",  "Vanilla",    "204,630"),
       "PI_loop":  ("path integration + LOOP x4", "Looped",     "204,630"),
       "IX_flat":  ("index, no loop",             "RoPE",       "204,182"),
       "IX_loop":  ("index + LOOP x4",            "RoPELooped", "204,182"),
       "PI_L3":    ("path integration, 3 REAL layers", "Vanilla", "601,174")}
ORDER = ["PI_flat", "PI_loop", "IX_flat", "IX_loop", "PI_L3"]
CHANCE = 0.0625
REF = 0.823          # published 128^2 path-integrated, LinearLR -- reference only


def load(runs, lbl, variant, TQ="256"):
    out = []
    for s in range(8):
        f = f"{runs}/{lbl}/s{s}/{variant}_matchquery.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        if TQ in d:
            out.append(d[TQ]["match_acc"])
    return np.array(out)


def paired(a, b):
    """b - a per seed, plus mean/sd/MDE. Assumes seed order aligned."""
    n = min(len(a), len(b))
    if n < 2:
        return None
    d = b[:n] - a[:n]
    sd = d.std(ddof=1)
    return dict(d=d, mean=d.mean(), sd=sd, mde=MDE_K * sd / np.sqrt(n), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tq", default="256")
    a = ap.parse_args()

    A = {k: load(a.runs_dir, k, LBL[k][1], a.tq) for k in ORDER}
    o = ["# Does a loop help where path integration is NOT sufficient?", "",
         "The looped pilot found recursion substitutes for depth in the INDEX arm",
         "(+0.363, sd 0.018, 3/3 seeds) but nothing on top of PATH INTEGRATION",
         "(+0.046, sd 0.074, MDE 0.120, one seed negative). That null was",
         "uninterpretable -- MapFormer already scored 0.948 on the torus, leaving 0.052",
         "of headroom. A ceiling cannot distinguish 'the loop adds nothing' from 'there",
         "was nothing to add'.", "",
         f"Match-Query 128^2 leaves **{1.0 - REF:.3f}** of headroom (published",
         f"path-integrated {REF:.3f}, index 0.192, chance {CHANCE:.4f}), so the loop has",
         "room to show an effect if it has one. All arms retrained in ONE batch with",
         "warmup+cosine and fast-attn, so the published number is a reference, not a",
         f"baseline. n=3, scored at TQ={a.tq}.", "",
         "| arm | params | accuracy | per-seed |", "|---|---|---|---|"]
    for k in ORDER:
        v = A[k]
        if not len(v):
            o.append(f"| {LBL[k][0]} | {LBL[k][2]} | — | missing |"); continue
        o.append(f"| {LBL[k][0]} | {LBL[k][2]} | **{v.mean():.3f}** ± {v.std(ddof=1):.3f} | "
                 f"{', '.join(f'{x:.3f}' for x in v)} |")
    o += ["", f"chance = {CHANCE:.4f}, perfect = 1.000", ""]

    o += ["## The four pre-registered questions", ""]
    def line(tag, base, test, txt_yes, txt_no):
        p = paired(A[base], A[test])
        if p is None:
            o.append(f"**{tag}** — incomplete."); o.append(""); return None
        det = abs(p["mean"]) > p["mde"]
        o.append(f"**{tag}** {LBL[test][0]} − {LBL[base][0]} = **{p['mean']:+.3f}** "
                 f"(sd {p['sd']:.3f}, MDE {p['mde']:.3f}, n={p['n']}, "
                 f"per-seed {', '.join(f'{x:+.3f}' for x in p['d'])}, "
                 f"{sum(p['d'] > 0)}/{p['n']} positive)")
        o.append("")
        o.append(txt_yes if det and p["mean"] > 0 else
                 ("Detectable but NEGATIVE — the loop HURTS here." if det else txt_no))
        o.append("")
        return p

    line("Q1 sanity.", "IX_flat", "PI_flat",
         "Path integration is necessary on this task, as published. The rest is readable.",
         "**Path integration shows no advantage here — the task is NOT behaving as "
         "published (0.823 vs 0.192). Nothing below is interpretable.**")
    q2 = line("Q2 THE QUESTION.", "PI_flat", "PI_loop",
         "**The loop COMPLEMENTS path integration once there is headroom.** The torus "
         "null was a ceiling artifact, and a recursive MapFormer is worth building.",
         "**SUBSTITUTES, even with headroom.** The loop adds nothing on top of path "
         "integration on a task where 0.18 was available to win. This generalises the "
         "torus negative rather than explaining it away, and is the stronger result.")
    line("Q3 the reverse.", "IX_flat", "IX_loop",
         "A loop helps the arm with headroom regardless of position code.",
         "A loop needs a working position code to build on — near its floor the index "
         "arm gains nothing, unlike on the torus where it was already partly succeeding.")
    line("Q4 loop vs real depth.", "PI_L3", "PI_loop",
         "Looping BEATS three real layers at a third of the parameters.",
         "Looping does not beat real depth here; check the sign and the flat arm above "
         "to see whether it matches or falls short.")

    if q2 is not None:
        head = 1.0 - A["PI_flat"].mean() if len(A["PI_flat"]) else float("nan")
        o += [f"Headroom actually available to the loop was {head:.3f} "
              f"(1.0 − {A['PI_flat'].mean():.3f}); it captured "
              f"{q2['mean']/head*100:.0f}% of it.", ""]

    o += ["## Scope", "",
          "One task, one map size, one loop count (4), n=3. The loop here is the",
          "conservative form: a shared block with no per-iteration depth embedding, and",
          "theta computed once rather than refined per iteration. A negative on Q2 does",
          "NOT rule out the refine-theta variant, which is a different model."]
    open(a.out, "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
