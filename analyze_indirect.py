"""Analyse the Indirect Indexing 2x2 against INDIRECT_PREREG.md.
Run from /home/prashr: python3 -m mapformer.analyze_indirect --runs-dir <dir> --out <md>"""
import argparse, glob, json, math
import numpy as np

from mapformer.environment_indirect import IndirectWorld, VOCAB, STOI, LETTERS

PAPER = {"RoPE": (11.16 / 100, 2.45 / 100), "PoPE": (94.82 / 100, 2.91 / 100)}
ORDER = ["RoPE", "PoPE", "MapWM_r2", "MapPoPE_r2"]


def floors(n=10000, seed=9012):
    """Strategies that need no pointer arithmetic, on the same test distribution."""
    w = IndirectWorld(); rng = np.random.default_rng(seed)
    X, Y = w.sample(n, rng)
    copy_src, rand_letter, adjacent = 0, 0.0, 0.0
    for x, y in zip(X.numpy(), Y.numpy()):
        row = [VOCAB[i] for i in x if i != 0]
        s = "".join(row).split(",")[0]
        src = row[len(s) + 2]
        tgt = VOCAB[y]
        copy_src += int(src == tgt)
        rand_letter += 1.0 / len(s)
        j = s.index(src)
        nb = [s[k] for k in (j - 1, j + 1) if 0 <= k < len(s)]
        adjacent += sum(c == tgt for c in nb) / max(len(nb), 1)
    return {"uniform over letters": 1 / 52, "copy the source character": copy_src / n,
            "a random letter of the string": rand_letter / n,
            "a neighbour of the source character": adjacent / n}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    R = {}
    for f in sorted(glob.glob(f"{a.runs_dir}/*_s*/*.json")):
        j = json.load(open(f)); R.setdefault(j["name"], {})[j["seed"]] = j
    arms = [x for x in ORDER if x in R]
    acc = {k: np.array([R[k][s]["test_acc"] for s in sorted(R[k])]) for k in arms}
    L = [f"# Indirect Indexing (PoPE paper sec 5.1) with path integration -- `{a.runs_dir}`", "",
         "Pre-registration: `INDIRECT_PREREG.md`. Final-token accuracy on a 10k test split.", "",
         "## Arms", "", "| arm | position | encoding | test accuracy | paper | seeds | final loss | slope /1k |",
         "|---|---|---|---|---|---|---|---|"]
    meta = {"RoPE": ("index", "RoPE"), "PoPE": ("index", "PoPE"),
            "MapWM_r2": ("path integration", "RoPE"), "MapPoPE_r2": ("path integration", "PoPE")}
    for k in arms:
        js = list(R[k].values())
        sl = [float(np.polyfit(np.arange(len(j["loss_curve"][-max(2, len(j["loss_curve"]) // 10):])) * 500,
                               j["loss_curve"][-max(2, len(j["loss_curve"]) // 10):], 1)[0] * 1000) for j in js]
        p = PAPER.get(k)
        L.append(f"| {k} | {meta[k][0]} | {meta[k][1]} | {acc[k].mean():.3f} +/- {acc[k].std(ddof=1):.3f} | "
                 f"{'%.3f +/- %.3f' % p if p else '--'} | {len(js)} | "
                 f"{np.mean([j['final_loss'] for j in js]):.3f} | {np.median(sl):+.4f} |")
    L += ["", "## Solve rate (the task is bimodal: a run either finds the solution or sits at ~0.09)", "",
          "| arm | solved (acc > 0.5) | per-seed accuracies |", "|---|---|---|"]
    for k in arms:
        v = acc[k]
        L.append(f"| {k} | {int((v > 0.5).sum())}/{len(v)} | " + ", ".join(f"{x:.3f}" for x in sorted(v)) + " |")
    L += ["", "## Floors (strategies needing no pointer arithmetic)", ""]
    for k, v in floors().items():
        L.append(f"- {k}: {v:.3f}")
    L += ["", "## Registered contrasts (paired by seed, MDE = 2.8 sd / sqrt(n))", "",
          "| contrast | delta | sd | MDE | seeds + | verdict |", "|---|---|---|---|---|---|"]
    def con(x, y):
        ss = sorted(set(R[x]) & set(R[y]))
        d = np.array([R[x][s]["test_acc"] - R[y][s]["test_acc"] for s in ss])
        sd = d.std(ddof=1) if len(d) > 1 else float("nan")
        return d.mean(), sd, 2.8 * sd / math.sqrt(len(d)), int((d > 0).sum()), len(d)
    for x, y, lab in [("MapWM_r2", "RoPE", "R2 path integration on the RoPE row"),
                      ("MapPoPE_r2", "PoPE", "R3 path integration on the PoPE row"),
                      ("PoPE", "RoPE", "the paper's own contrast"),
                      ("MapPoPE_r2", "MapWM_r2", "encoding, on the path-integrated row")]:
        if x in R and y in R:
            d, sd, mde, pos, n = con(x, y)
            L.append(f"| {lab}: {x} - {y} | {d:+.3f} | {sd:.3f} | {mde:.3f} | {pos}/{n} | "
                     f"{'DETECTABLE' if abs(d) > mde else 'unmeasured'} |")
    if all(k in R for k in ORDER):
        ss = sorted(set.intersection(*[set(R[k]) for k in ORDER]))
        inter = np.array([(R["MapPoPE_r2"][s]["test_acc"] - R["PoPE"][s]["test_acc"]) -
                          (R["MapWM_r2"][s]["test_acc"] - R["RoPE"][s]["test_acc"]) for s in ss])
        sd = inter.std(ddof=1)
        L.append(f"| R4 interaction | {inter.mean():+.3f} | {sd:.3f} | {2.8*sd/math.sqrt(len(ss)):.3f} | "
                 f"{int((inter>0).sum())}/{len(ss)} | "
                 f"{'DETECTABLE' if abs(inter.mean()) > 2.8*sd/math.sqrt(len(ss)) else 'unmeasured'} |")
    L += ["", "## Registered verdicts", ""]
    if "RoPE" in R and "PoPE" in R:
        ok = acc["RoPE"].mean() < 0.30 and acc["PoPE"].mean() > 0.80
        L.append(f"- **R1 replication**: RoPE {acc['RoPE'].mean():.3f} (needs < 0.30), PoPE "
                 f"{acc['PoPE'].mean():.3f} (needs > 0.80) -> **{'REPLICATES' if ok else 'DOES NOT REPLICATE'}**")
    L += ["", "## Validation curves (accuracy every 5,000 steps, seed 0)", ""]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
