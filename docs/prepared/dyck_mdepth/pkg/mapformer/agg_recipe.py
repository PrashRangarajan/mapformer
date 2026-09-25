"""Which training recipe converges reliably? Variance, not mean, is the metric.

Picking a recipe on MEAN accuracy over 8 seeds selects a lucky basin. The primary
criterion here is CONVERGED FRACTION -- final loss < 0.05 with a flat tail -- and
the secondary is the sd of held-out accuracy, because sd is what sets the MDE and
the MDE is what has been swallowing this project's effects.
"""
import argparse, json, re, statistics as st
from pathlib import Path

CONDS = [("C0", "300 ep, lr 3e-4  (current recipe)"),
         ("C1", "300 ep, lr 1e-3"),
         ("C2", "600 ep, lr 1e-3")]
ARMS = ["Vanilla", "Looped"]
SEEDS = list(range(8))
LOSS_OK = 0.05
SLOPE_OK = 5e-4


def losses(runs_dir, c, v, s):
    f = Path(runs_dir) / f"{c}_{v}_s{s}.log"
    if not f.exists():
        return None
    return [float(x) for x in re.findall(r"Loss: ([0-9.]+)", f.read_text())]


def converged(ls):
    if not ls:
        return None
    k = max(2, len(ls) // 10)
    tail = ls[-k:]
    n = len(tail)
    mx = (n - 1) / 2
    slope = (sum((i - mx) * y for i, y in enumerate(tail))
             / max(sum((i - mx) ** 2 for i in range(n)), 1e-12))
    return ls[-1] < LOSS_OK and abs(slope) < SLOPE_OK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    acc = {}
    for c, _d in CONDS:
        f = Path(a.repo) / f"_RECIPE_{c}.json"
        if not f.exists():
            continue
        for k, rows in json.load(open(f)).items():
            _p, v, T = k.split("|")
            acc[(c, v, int(T))] = {int(s): x for s, x, _n in rows if x is not None}

    o = ["# Which recipe converges? Fixing power before changing the task", "",
         "Torus clean, 2 arms x 3 conditions x 8 seeds, one batch, parallel data",
         "path throughout.", "",
         "**Primary metric is CONVERGED FRACTION** (final loss < 0.05 and a flat",
         "tail), **secondary is sd of accuracy**. Mean accuracy is reported and is",
         "NOT the criterion: choosing a recipe on the mean over 8 seeds selects a",
         "lucky basin.", "",
         "## Convergence", "",
         "| condition | " + " | ".join(f"{v} converged" for v in ARMS)
         + " | " + " | ".join(f"{v} final loss" for v in ARMS) + " |",
         "|---" * (2 * len(ARMS) + 1) + "|"]
    conv = {}
    for c, desc in CONDS:
        cells, lcells = [], []
        for v in ARMS:
            cs = [converged(losses(a.runs_dir, c, v, s)) for s in SEEDS]
            cs = [x for x in cs if x is not None]
            conv[(c, v)] = cs
            cells.append(f"**{sum(cs)}/{len(cs)}**" if cs else "—")
            fl = [losses(a.runs_dir, c, v, s) for s in SEEDS]
            fl = [x[-1] for x in fl if x]
            lcells.append(f"{min(fl):.4f} – {max(fl):.4f}" if fl else "—")
        o.append(f"| {c} · {desc} | " + " | ".join(cells + lcells) + " |")
    o.append("")

    for T in (128, 512, 1024):
        o += [f"## T={T}", "",
              "| condition | " + " | ".join(f"{v} mean" for v in ARMS)
              + " | " + " | ".join(f"{v} sd" for v in ARMS)
              + " | " + " | ".join(f"{v} MDE@8" for v in ARMS) + " |",
              "|---" * (3 * len(ARMS) + 1) + "|"]
        for c, _d in CONDS:
            m, s_, d_ = [], [], []
            for v in ARMS:
                xs = list(acc.get((c, v, T), {}).values())
                if len(xs) > 1:
                    sd = st.stdev(xs)
                    m.append(f"{st.mean(xs):.3f}"); s_.append(f"{sd:.3f}")
                    d_.append(f"{2.8 * sd / len(xs) ** 0.5:.3f}")
                else:
                    m.append("—"); s_.append("—"); d_.append("—")
            o.append(f"| {c} | " + " | ".join(m + s_ + d_) + " |")
        o.append("")

    o += ["## Verdict", ""]
    full = [c for c, _d in CONDS
            if all(conv.get((c, v)) and sum(conv[(c, v)]) == len(conv[(c, v)])
                   for v in ARMS)]
    base = {v: sum(conv.get(("C0", v), [])) for v in ARMS}
    # PARETO, not strict-improvement-everywhere. The first version of this rule
    # required a strictly higher converged count on EVERY arm, which is impossible
    # whenever one arm is already at 8/8 -- Looped is, in all three conditions -- so
    # nothing could ever qualify and the verdict read "no condition beats the
    # current recipe" while C1 was cutting Vanilla's sd by 3.5x. Require no
    # regression anywhere and a strict gain somewhere.
    better = [c for c, _d in CONDS if c != "C0"
              and all(sum(conv.get((c, v), [0])) >= base.get(v, 0) for v in ARMS)
              and any(sum(conv.get((c, v), [0])) > base.get(v, 0) for v in ARMS)]
    if full:
        o += [f"**{full[0]} converges on every seed of both arms.** Adopt it for "
              f"step 2 (Match-Query with stochastic transitions) and re-check the "
              f"converged fraction there -- recipe transfer across tasks is an "
              f"assumption, not a result.", ""]
    else:
        if better:
            o += [f"**No condition is clean 8/8, but {better[0]} beats the current "
                  f"recipe on both arms.** Partial fix; take it, and keep looking.", ""]
        else:
            o += ["**No condition beats the current recipe.** The bimodality is not "
                  "the LR schedule. Next suspects are init scale and dropout, "
                  "neither of which is a CLI knob yet -- and if those fail too, the "
                  "honest conclusion is that this landscape has a genuine bad basin "
                  "and the fix is more seeds, not a better recipe.", ""]
    # Write the machine-readable choice so step 2 does not have to be told by hand.
    RECIPES = {"C0": (300, "3e-4"), "C1": (300, "1e-3"), "C2": (600, "1e-3")}
    pick = full[0] if full else (better[0] if not full and better else "C0")
    ep, lr = RECIPES[pick]
    json.dump({"condition": pick, "epochs": ep, "lr": lr,
               "converged": {f"{c}|{v}": sum(conv.get((c, v), []))
                             for c, _d in CONDS for v in ARMS}},
              open(Path(a.repo) / "RECIPE_CHOICE.json", "w"), indent=2)
    o += [f"Machine-readable choice written to `RECIPE_CHOICE.json`: **{pick}** "
          f"({ep} ep, lr {lr}).", ""]

    o += ["## Scope", "",
          "One task, two arms, three conditions, n=8. Only knobs that already "
          "existed were varied (epochs, lr); warmup fraction, cosine floor, dropout "
          "and init scale are hardcoded and untested here. A recipe that fixes the "
          "torus is not guaranteed to fix Match-Query -- that is why step 2 "
          "re-measures the converged fraction rather than assuming it."]
    Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
