"""Declared secondary for GAIN_GRAIN (Amendment 1, audit D4): the dropout-scale re-score. Registers the two new layer
classes with rescore_hook (both feed o_proj with softmax(scores) @ V; model_em_pope.py), installs the x1/(1-p) attention
hook, and re-runs the batch's own eval (eval_gain_grain -> eval_noise_refine) at T=128 into GAIN_GRAIN_RESCORE.json;
then prints per-arm means and the registered contrasts' differences on the re-scored accuracies (no verdict).
Usage: python3 gain_grain_rescore.py <eval args...>   (the driver passes the registered eval flags)."""
import json, runpy, sys
sys.path.insert(0, "/home/prashr")
import numpy as np
import mapformer.rescore_hook as RH

RH.KNOWN |= {"mapformer.model_em_pope.GainKernelLayer", "mapformer.model_em_pope.EMTransformerLayer_NonNeg"}
if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1]
    RH.install("auto")
    sys.argv = ["mapformer.eval_gain_grain", "mapformer.eval_noise_refine"] + sys.argv[1:]
    try:
        runpy.run_module("mapformer.eval_gain_grain", run_name="__main__", alter_sys=True)
    except SystemExit:
        pass
    RH._report()
    if RH.STATS["skipped"]:
        raise SystemExit(f"unknown layer classes skipped: {sorted(RH.STATS['skipped'])}")
    from mapformer.stats_core import perm2_p
    J = json.load(open(out.replace(".md", ".json"))); R = json.load(open("/home/prashr/mapformer/GAIN_GRAIN_EVAL.json"))
    arms = sorted({k.split("|")[1] for k in J})
    acc = lambda D, a: np.array([r[1] for r in sorted(D[f"0.0|{a}|128"])])
    print("re-scored (attention x 1/(1-p)) vs registered, T=128:")
    for a in arms:
        print(f"  {a:18s} {acc(R, a).mean():.4f} -> {acc(J, a).mean():.4f}")
    for x, y in (("MapPoPE-Pair", "GainScalar"), ("MapPoPE-Pair", "GainMod4"), ("VanillaEM", "VanillaEM_NonNeg"), ("Vanilla", "MapPoPE-Pair")):
        if x in arms and y in arms:
            for lab, D in (("registered", R), ("re-scored", J)):
                d = acc(D, y).mean() - acc(D, x).mean()
                print(f"  {y} - {x} ({lab}): {d:+.4f} (two-sided perm p {perm2_p(acc(D, x), acc(D, y))['p']:.4f})")
