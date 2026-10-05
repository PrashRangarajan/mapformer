"""Re-score batches whose registered eval is inside the trainer (eval.json per run): text world, TW_NORMSTEP, H3
(cancel), leak. Per run: rebuild the model as the trainer did, load the checkpoint, run the trainer's own evaluate on
its own stream with (a) no hook -- must reproduce eval.json -- and (b) rescore_hook scale auto (attention x 1/(1-p)).
Writes runs_rescore/<batch>.json: {run: [registered, rerun_none, rescored]}."""
import glob, json, os, sys
import numpy as np, torch
import torch.nn as nn
import mapformer.rescore_hook as RH

R = "/home/prashr/mapformer"; O = f"{R}/runs_rescore"; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
ORIG_EVAL = nn.Module.eval


def with_scale(fn, scale):
    nn.Module.eval = ORIG_EVAL
    if scale:
        RH.install("auto")
    try:
        return fn()
    finally:
        nn.Module.eval = ORIG_EVAL


def fresh(build, ck):
    m = build(); m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"])
    return m.to(dev)


def batch_textworld():
    from mapformer.environment_textworld import TextWorld
    from mapformer.train_textworld import evaluate
    from mapformer.train_variant import VARIANT_MAP
    out = {}
    for d in sorted(glob.glob(f"{R}/runs/textworld/p0/*_s*/")):
        ev = json.load(open(f"{d}eval.json")); ck = glob.glob(f"{d}*.pt")[0]; b = torch.load(ck, map_location="cpu", weights_only=False)
        a = b["config"]["args"]; V = b["config"]["vocab_size"]
        build = lambda: VARIANT_MAP[a["variant"]](vocab_size=V, d_model=128, n_heads=2, n_layers=a["n_layers"], grid_size=a["size"])
        te = lambda: TextWorld(size=a["size"], seed=10000)
        r = [with_scale(lambda: evaluate(fresh(build, ck), te(), a["n_steps"], a["n_trials"], dev)[0], s) for s in (False, True)]
        out[os.path.basename(d.rstrip("/"))] = [ev["eval"][str(a["n_steps"])]["acc"]] + r
    return out


def batch_tw_normstep():
    from mapformer.environment_textworld import TextWorld
    from mapformer.train_textworld import evaluate
    from mapformer.train_tw_normstep import build as tbuild, EVAL_SEED, HELDOUT
    out = {}
    for d in sorted(glob.glob(f"{R}/runs/tw_normstep/p0/*_s*/")):
        ev = json.load(open(f"{d}eval.json")); ck = glob.glob(f"{d}*.pt")[0]; b = torch.load(ck, map_location="cpu", weights_only=False)
        a = b["config"]["args"]
        build = lambda: tbuild(b["arm"], TextWorld(size=a["size"], seed=a["seed"]), a["n_layers"], a["size"])
        r = [with_scale(lambda: evaluate(fresh(build, ck), TextWorld(size=a["size"], seed=HELDOUT), a["n_steps"], a["n_trials"], dev,
                                         seed=EVAL_SEED)[0], s) for s in (False, True)]
        out[os.path.basename(d.rstrip("/"))] = [ev["eval"][str(a["n_steps"])]["acc"]] + r
    return out


def batch_cancel():
    from mapformer.environment_cancel import GridWorldCancel
    from mapformer.train_cancel import evaluate
    from mapformer.train_variant import VARIANT_MAP
    out = {}
    for d in sorted(glob.glob(f"{R}/runs/cancel/p0/*/")):
        if not os.path.exists(f"{d}eval.json"):
            continue
        ev = json.load(open(f"{d}eval.json")); ck = glob.glob(f"{d}*.pt")[0]; b = torch.load(ck, map_location="cpu", weights_only=False)
        a = b["config"]["args"]; V = b["config"]["vocab_size"]
        build = lambda: VARIANT_MAP[a["variant"]](vocab_size=V, d_model=128, n_heads=2, n_layers=a["n_layers"], grid_size=a["size"])
        te = lambda: GridWorldCancel(size=a["size"], seed=10000, p_plus=a["p_plus"])
        r = [with_scale(lambda: evaluate(fresh(build, ck), te(), a["n_steps"], a["n_trials"], dev)[0], s) for s in (False, True)]
        out[os.path.basename(d.rstrip("/"))] = [ev["eval"][str(a["n_steps"])]["acc"]] + r
    return out


def batch_leak():
    from mapformer.leak_eval import evaluate_run
    out = {}
    for d in sorted(glob.glob(f"{R}/runs/leak/p0/*_s*/")):
        ck = glob.glob(f"{d}*.pt")[0]; arm = os.path.basename(d.rstrip("/")).rsplit("_s", 1)[0]
        r = [with_scale(lambda: evaluate_run(ck, arm, dev, scales=(1.0,))["test|x1"]["intact"], s) for s in (False, True)]
        out[os.path.basename(d.rstrip("/"))] = [None] + r
    return out


if __name__ == "__main__":
    name = sys.argv[1]; res = globals()[f"batch_{name}"]()
    json.dump(res, open(f"{O}/{name}.json", "w"), indent=1)
    bad = [k for k, v in res.items() if v[0] is not None and abs(v[0] - v[1]) > 1e-9]
    print(f"{name}: {len(res)} runs; rerun reproduces eval.json on {len(res) - len(bad)}/{len(res)}" + (f"  MISMATCH {bad[:5]}" if bad else ""))
