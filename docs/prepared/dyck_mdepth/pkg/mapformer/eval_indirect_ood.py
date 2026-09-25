"""Indirect Indexing out of distribution, eval-only on the 200k checkpoints.

The PoPE paper tests this task in-distribution only (strings 20-40, shifts in [-15, +15]).
Path integration's wins elsewhere in this project are all OOD, so this asks the question the
paper does not: does either encoding survive larger shifts or longer strings?

Two axes, kept separate because they confound differently:
  SHIFT   strings 20-40 as trained, shifts drawn from |k| in [16, 30]. Block size unchanged at 56,
          so nothing about padding or absolute position moves -- the cleanest OOD axis.
  LENGTH  strings 41-56, which needs a larger block (72). That changes how many left-pad tokens
          the model sees, so a PAD CONTROL is run alongside: in-distribution strings (20-40,
          shifts <= 15) evaluated at block 72. Any drop there is the padding change, not length.
"""
import argparse, glob, json
import numpy as np
import torch

from mapformer import environment_indirect as E
from mapformer.environment_indirect import IndirectWorld, LETTERS, STOI
from mapformer.train_indirect import build

ARMS = [("PoPE", "PoPE", 1), ("MapPoPE_r2", "MapPoPE", 2)]


class ShiftWorld(IndirectWorld):
    """Same generator, but |shift| drawn from [lo, hi]."""

    def __init__(self, lo, hi, **kw):
        super().__init__(**kw); self.lo, self.hi = lo, hi

    def sample(self, n, rng):
        X = np.zeros((n, E.BLOCK), np.int64); Y = np.zeros(n, np.int64)
        for i in range(n):
            while True:
                L = int(rng.integers(self.min_len, self.max_len + 1))
                s = list(rng.choice(len(LETTERS), size=L, replace=False))
                mag = int(rng.integers(self.lo, self.hi + 1))
                if mag >= L:
                    continue
                k = mag if rng.random() < 0.5 else -mag
                j = int(rng.integers(0, L))
                if 0 <= j + k < L:
                    break
            src, tgt = LETTERS[s[j]], LETTERS[s[j + k]]
            txt = "".join(LETTERS[c] for c in s) + ", " + src + ", " + ("+" if k >= 0 else "-") + str(abs(k)) + ", "
            ids = [STOI[c] for c in txt]
            X[i, E.BLOCK - len(ids):] = ids; Y[i] = STOI[tgt]
        return torch.from_numpy(X), torch.from_numpy(Y)


@torch.no_grad()
def acc(model, X, Y, dev, bs=256):
    ok = 0
    for i in range(0, len(X), bs):
        ok += int((model(X[i:i + bs].to(dev))[:, -1].argmax(-1).cpu() == Y[i:i + bs]).sum())
    return ok / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/home/prashr/mapformer/runs/indirect_200k")
    ap.add_argument("--out", default="/home/prashr/mapformer/INDIRECT_OOD.md")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    base_block = E.BLOCK

    sets = {"in-distribution (20-40, |k|<=15)": (base_block, IndirectWorld()),
            "shift 16-30": (base_block, ShiftWorld(16, 30)),
            "shift 21-30": (base_block, ShiftWorld(21, 30)),
            "length 41-52 (block 68)": (68, IndirectWorld(min_len=41, max_len=52)),
            "PAD CONTROL: 20-40 at block 68": (68, IndirectWorld())}
    data = {}
    for k, (blk, w) in sets.items():
        E.BLOCK = blk; w.__class__.block = blk
        data[k] = (blk, *w.sample(a.n, np.random.default_rng(31337)))
    E.BLOCK = base_block

    res = {}
    for name, arch, _ in ARMS:
        for pt in sorted(glob.glob(f"{a.runs_dir}/{name}_s*/{name}.pt")):
            seed = int(pt.split("_s")[-1].split("/")[0])
            m = build(arch, 512, 8, 8, 2, base_block, "uniform", torch.Generator()).to(dev).eval()
            m.load_state_dict(torch.load(pt, map_location=dev))
            for k, (blk, X, Y) in data.items():
                res.setdefault((name, k), {})[seed] = acc(m, X, Y, dev)
            print(name, seed, {k: round(res[(name, k)][seed], 3) for k in data}, flush=True)

    keys = list(data)
    L = ["# Indirect Indexing out of distribution (eval-only, 200k checkpoints)", "",
         "Trained on strings of 20-40 letters with shifts in [-15, +15]. Cells are means over the "
         "seeds that SOLVED in distribution (accuracy > 0.5 there), with the solver count; a mean "
         "over failed seeds would measure the plateau, not generalisation. Chance 0.019.", "",
         "| arm | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for name, _, _ in ARMS:
        solved = [s for s, v in res[(name, keys[0])].items() if v > 0.5]
        row = []
        for k in keys:
            v = np.array([res[(name, k)][s] for s in solved])
            row.append(f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}")
        L.append(f"| {name} (n={len(solved)}) | " + " | ".join(row) + " |")
    L += ["", "## Paired contrast on the seeds both arms solve", "",
          "| condition | MapPoPE - PoPE | sd | MDE | seeds + |", "|---|---|---|---|---|"]
    sp = [s for s, v in res[("PoPE", keys[0])].items() if v > 0.5]
    sm = [s for s, v in res[("MapPoPE_r2", keys[0])].items() if v > 0.5]
    both = sorted(set(sp) & set(sm))
    for k in keys:
        d = np.array([res[("MapPoPE_r2", k)][s] - res[("PoPE", k)][s] for s in both])
        sd = d.std(ddof=1)
        L.append(f"| {k} | {d.mean():+.3f} | {sd:.3f} | {2.8*sd/np.sqrt(len(d)):.3f} | {(d>0).sum()}/{len(d)} |")
    json.dump({f"{n}|{k}": v for (n, k), v in res.items()}, open(a.out.replace(".md", ".json"), "w"), indent=1)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
