"""Does refining theta each pass beat re-reading a fixed theta?

Verdict rules and the gate diagnostic are fixed before the runs finish.
"""
import argparse, glob, json, os
import numpy as np

def accs(R, v, TQ="256"):
    o = {}
    for s in range(8):
        f = f"{R}/{v}/s{s}/{v}_matchquery.json"
        if os.path.exists(f):
            d = json.load(open(f))
            if TQ in d:
                o[s] = d[TQ]["match_acc"]
    return o

def gates(R):
    """The learned gate per seed. Near zero = the model DECLINED to refine, which
    is a mechanism answer, not a null."""
    import torch
    o = {}
    for s in range(8):
        f = f"{R}/LoopedRefine/s{s}/LoopedRefine_matchquery.pt"
        if not os.path.exists(f):
            continue
        try:
            b = torch.load(f, map_location="cpu", weights_only=False)
            sd = b.get("model_state") or b.get("model") or b.get("state_dict") or b
            for k in sd:
                if k.endswith("gate"):
                    o[s] = float(np.ravel(sd[k].numpy())[0]); break
        except Exception:
            pass
    return o

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, B = accs(a.runs_dir, "Looped"), accs(a.runs_dir, "LoopedRefine")
    G = gates(a.runs_dir)
    o = ["# Does REFINING theta each pass beat re-reading a fixed one?", "",
         "`Looped` computes theta once from the token embeddings and re-reads it every",
         "pass. `LoopedRefine` carries and corrects it:",
         "`theta = theta_0 + gate * tanh(refine(x))` -- this repo's InEKF idea moved from",
         "the sequence axis to the DEPTH axis. gate starts at 0, so at init the two models",
         "are bit-identical (verified 0.00e+00); the gate gradient at 0 is 1.9e-03, so the",
         "no-op init is escapable. +385 params (0.19%).", "",
         "Match-Query 128^2, TQ=256, chance 0.0625, both arms retrained in one batch.", "",
         "| arm | n | mean | sd | min | per-seed |", "|---|---|---|---|---|---|"]
    for nm, D in (("Looped (theta fixed)", A), ("LoopedRefine (theta refined)", B)):
        v = np.array(list(D.values()))
        if not len(v):
            o.append(f"| {nm} | — | — | — | — | missing |"); continue
        o.append(f"| {nm} | {len(v)} | **{v.mean():.3f}** | {v.std(ddof=1):.3f} | "
                 f"{v.min():.3f} | {' '.join(f'{x:.2f}' for x in v)} |")
    ks = sorted(set(A) & set(B))
    o.append("")
    if len(ks) >= 2:
        d = np.array([B[s] - A[s] for s in ks]); sd = d.std(ddof=1)
        mde = 2.8 * sd / np.sqrt(len(d))
        det = abs(d.mean()) > mde
        o += [f"**Refine − fixed: {d.mean():+.3f}** (sd {sd:.3f}, MDE(n={len(d)}) {mde:.3f}, "
              f"{sum(d>0)}/{len(d)} positive, per-seed "
              f"{', '.join(f'{x:+.3f}' for x in d)})", ""]
        if det and d.mean() > 0:
            o += ["**REFINEMENT ADDS SOMETHING A FIXED THETA CANNOT.** The InEKF idea works "
                  "on the depth axis, where on the sequence axis it turned out to be "
                  "stabilisation rather than inference."]
        elif det:
            o += ["**REFINEMENT HURTS.** Correcting theta from the hidden state degrades the "
                  "position code -- the Level15EM failure mode, on a new axis."]
        else:
            o += ["**NO DETECTABLE DIFFERENCE.** The loop's benefit is ITERATION alone; "
                  "carrying and correcting a position estimate adds nothing on top. That "
                  "extends this project's standing finding -- the Kalman win was "
                  "stabilisation and token-type gating, not inference -- to a second axis."]
    o += ["", "## The learned gate (diagnostic, whatever the verdict)", ""]
    if G:
        g = np.array(list(G.values()))
        o += [f"per-seed {', '.join(f'{v:+.3f}' for v in G.values())}",
              f"mean |gate| {np.abs(g).mean():.4f}", ""]
        o += ["The gate stayed at essentially ZERO: the model DECLINED to refine, so any "
              "null above is a choice the optimiser made, not a failure to express the "
              "correction. Same tell as the learnable-beta red herring, where learned betas "
              "barely moved from init."
              if np.abs(g).mean() < 0.05 else
              "The gate moved away from zero, so the model DID use the correction -- "
              "whatever it bought is visible in the accuracy above."]
    else:
        o += ["(gate not recoverable from the checkpoints)"]
    o += ["", "## Scope", "", "One task, one loop count (4), correction applied to theta and",
          "bounded by tanh. A correction applied to delta (odometry) and re-integrated is a",
          "different model and is untested."]
    open(a.out, "w").write("\n".join(o) + "\n"); print("\n".join(o))

if __name__ == "__main__":
    main()
