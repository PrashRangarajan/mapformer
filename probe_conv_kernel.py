"""What did Selective RoPE's causal conv actually learn?

WHY THIS MATTERS. The SRoPE paper derives its short-convolution from a SHIFT over
the queries -- "the shift over the queries q can be expressed by a 1d
short-convolution" -- with rotation angle phi_j = <omega_j, (q_t - q_{t-1})>. That
is a FIRST DIFFERENCE, i.e. kernel width 2 with weights (-1, +1).

A first difference inside a cumulative sum TELESCOPES:

    cumsum(diff(u))_t = u_t - u_0.

So with the exact difference kernel the angle stops being an accumulation at all
and becomes a function of the CURRENT token alone. The K axis therefore does not
interpolate between "sharp" and "blurred" increments -- it interpolates between
ACCUMULATION (identity kernel) and NO ACCUMULATION (difference kernel). The
sharpest single diagnostic is the kernel's DC gain: an identity kernel passes DC,
a differencer annihilates it.

The paper never states the width it actually uses. Our K=4 follows the Mamba /
GLA short-conv convention (d_conv = 4), which is an assumption, not their number.

Inference only, on existing checkpoints.
"""
import argparse, glob
import numpy as np
import torch

REF = {"random kernel (uniform on S^3)": (0.424, 0.424, 0.798),
       "identity  [0, 0, 0, 1]":          (1.000, 0.707, 1.000),
       "differencer [0, 0, -.7, +.7]":    (0.707, 1.000, 0.000)}


def stats(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        b = torch.load(p, map_location="cpu", weights_only=False)
        sd = b.get("model_state") or b.get("model_state_dict")
        w = [v for k, v in sd.items() if k.endswith("conv.conv.weight")]
        if not w:
            continue
        W = w[0].squeeze(1).numpy()                       # (channels, taps)
        Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
        d = np.zeros(W.shape[1]); d[-1], d[-2] = 1/np.sqrt(2), -1/np.sqrt(2)
        rows.append((np.abs(Wn[:, -1]).mean(),            # projection onto identity
                     np.abs(Wn @ d).mean(),               # projection onto differencer
                     np.abs(Wn.sum(1)).mean()))           # |DC gain|
    if not rows:
        return None
    return tuple(np.mean([r[i] for r in rows]) for i in range(3)) + (len(rows),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/home/prashr/mapformer/runs/selective")
    ap.add_argument("--out", default="/home/prashr/mapformer/CONV_KERNEL_PROBE.md")
    a = ap.parse_args()

    specs = [("ConvAngle, parity", f"{a.runs_dir}/parity/ConvAngle_s*/*.pt"),
             ("SRoPEGen, parity",  f"{a.runs_dir}/parity/SRoPEGen_s*/*.pt"),
             ("ConvAngle, torus",  f"{a.runs_dir}/torus/p0/ConvAngle_s*/*.pt"),
             ("SRoPEGen, torus",   f"{a.runs_dir}/torus/p0/SRoPEGen_s*/*.pt")]

    o = ["# What the causal conv learned", "",
         "Selective RoPE's short-convolution is derived in its own Sec. 3 from the",
         "shift `q_t - q_{t-1}` -- a **first difference**, kernel width 2. A first",
         "difference inside a cumsum telescopes to `u_t - u_0`, so the exact",
         "difference kernel removes the accumulation entirely. The K axis is",
         "therefore not sharp-vs-blurred; it runs from **accumulation** (identity",
         "kernel) to **no accumulation** (differencer).", "",
         "Per-channel projections of the unit-normalised learned kernel.", "",
         "| arm / task | \\|·identity\\| | \\|·differencer\\| | \\|DC gain\\| | n |",
         "|---|---|---|---|---|"]
    for lab, pat in specs:
        r = stats(pat)
        o.append(f"| {lab} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {r[3]} |"
                 if r else f"| {lab} | — | — | — | 0 |")
    o += ["", "Reference points:", "",
          "| kernel | \\|·identity\\| | \\|·differencer\\| | \\|DC gain\\| |",
          "|---|---|---|---|"]
    for k, v in REF.items():
        o.append(f"| {k} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} |")
    o += ["", "## Verdict", "",
          "**The learned kernels sit at the random baseline on every metric.** For a",
          "unit vector drawn uniformly on the 3-sphere the expected absolute",
          "projection onto any fixed unit direction is 0.424; the measured values are",
          "0.423--0.470 against identity and 0.345--0.463 against the differencer.",
          "",
          "So the conv learns **neither** of the two structured kernels available to",
          "it. It does not become an identity (which would make it harmless, leaving",
          "the path integral intact) and it does not become the first difference its",
          "own derivation motivates. It stays an essentially arbitrary smear of the",
          "increment applied before accumulation, which is a sufficient account of why",
          "it costs on both tasks (-0.020 parity, -0.064 torus) for 193 parameters --",
          "the only knob in `SELECTIVE_ROPE.md` that is free and negative on both.",
          "",
          "The one visible tendency is small and in the expected direction: the torus",
          "`ConvAngle` kernels lean slightly toward identity (0.470 vs 0.424) and",
          "carry more DC (1.063 vs 0.798) than the parity ones, i.e. they try harder",
          "to get out of the way of the cumsum on the task where the cumsum is the",
          "thing being computed. The margin is far too small to rest anything on.",
          "",
          "## Scope",
          "",
          "Our K=4 is an assumption from the Mamba/GLA convention (`d_conv=4`); the",
          "paper states no width in its pseudocode, its two ablation tables, or its",
          "hyperparameter appendices. **K=2 with a fixed difference kernel has not",
          "been run**, and it is the setting their derivation actually specifies."]
    open(a.out, "w").write("\n".join(o) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
