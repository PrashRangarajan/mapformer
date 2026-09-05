"""What did the forget gate learn? lambda is the whole answer.

lambda > 0 -> the model chose to DECAY (downweight distant keys).
lambda ~ 0 -> it DECLINED, from an escapable start (verified: |grad| 5.0e-03 at
              zero against a 4.8e-04 median, and lambda leaves zero at step 1).
lambda < 0 -> ANTI-recency: it chose to UPWEIGHT distant keys, which would be a
              result in its own right and the opposite of the design principle.

Also reports the effective decay over the training horizon, since lambda alone is
not interpretable without the gate's typical activation: the accumulated bias at
lag L is about -lambda * E[sigmoid] * L logits.
"""
import argparse, glob
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["Forget", "Forget_r4"])
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    o = ["## What the forget gate learned", "",
         "| arm | lambda (mean ± sd) | per-seed | E[sigmoid] | bias at lag "
         f"{a.horizon} |", "|---|---|---|---|---|"]
    for arm in a.arms:
        lams, sigs = [], []
        for cp in sorted(glob.glob(f"{a.runs_dir}/{arm}_s*/{arm}.pt")):
            sd = torch.load(cp, map_location="cpu", weights_only=False)["model_state_dict"]
            lams.append(float(sd["forget.lam"].item()))
            # E[sigmoid] over the vocabulary, a stand-in for the typical gate value
            E, W, b = sd["token_emb.weight"], sd["forget.proj.weight"], sd["forget.proj.bias"]
            sigs.append(float(torch.sigmoid(E @ W.T + b).mean()))
        if not lams:
            o.append(f"| `{arm}` | — | — | — | — |"); continue
        lam, sig = np.array(lams), float(np.mean(sigs))
        o.append(f"| `{arm}` | {lam.mean():+.4f} ± {lam.std(ddof=1):.4f} | "
                 + " ".join(f"{v:+.3f}" for v in lam)
                 + f" | {sig:.3f} | {-lam.mean()*sig*a.horizon:+.2f} logits |")
    o += ["", "lambda > 0 = decay, ~0 = declined (escapable start verified), "
          "< 0 = anti-recency."]
    open(a.out, "a").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
