"""Amendment 3 (CPU only): validation of the LEAK-FREE convergence measure nll_ms the D3 gate reads -- eval-mode mean NLL
at every revisit target (objects and blanks) with object steps mean-substituted (gain_phase_eval.obj_eval, mode 'mean'),
on the fixed test-pool eval stream (200 sequences). Output: gain_phase_lossgate_validate_out.txt / .json.
 G1 LEAK checkpoints (runs/leak/p0, seeds 0-7): nll (intact) and nll_ms, beside the training tail (final-5%).
 G2 leak independence (MapWM s0): the identity part of every object's step scaled by k in {1, 1.5, 2, 3} -- nll rises
    with k, nll_ms must not move.
 G3 must FAIL the gate: untrained models of all four GAIN_PHASE arms (seeds 8, 110) and the 8 pilot checkpoints
    (runs/gain_phase_pilot/p0, 30-epoch schedule)."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
torch.set_num_threads(12)
from mapformer.environment_newobj import NewObjectWorld, N_SPECIAL
from mapformer.gain_phase_eval import load, Probe, obj_eval, step_of, sequences, P
from mapformer.model_codes import set_pool, use_object_codes
from mapformer.model_gain_phase import ARMS as GP_ARMS
from mapformer.stats_core import classify_run

REPO = "/home/prashr/mapformer"; OUT = f"{REPO}/docs/audits/2026-10-06/gain_phase_lossgate_validate"
dev = "cpu"
toks, revs = sequences("test")
res = {"leak": {}, "scale": [], "untrained": [], "pilot": []}


def nlls(p):
    a, n = obj_eval(p, toks, revs, "test", None, dev); a_ms, n_ms = obj_eval(p, toks, revs, "test", "mean", dev)
    return {"acc": a, "nll": n, "acc_ms": a_ms, "nll_ms": n_ms}


print("== G1 LEAK checkpoints ==", flush=True)
for arm in ("MapWM", "NormStep", "ActOnly"):
    res["leak"][arm] = []
    for s in range(8):
        m, b = load(f"{REPO}/runs/leak/p0/{arm}_s{s}/{arm}.pt", arm, dev); p = Probe(m); set_pool(m, "test")
        r = nlls(p); r.update(seed=s, tail=classify_run(b["losses"])["tail"]); res["leak"][arm].append(r)
        print(f"  {arm:8s} s{s}: training tail {r['tail']:.4f} | eval nll {r['nll']:.4f} | nll_ms {r['nll_ms']:.4f} "
              f"(acc {r['acc']:.4f} -> {r['acc_ms']:.4f})", flush=True)

print("== G2 MapWM s0, identity step scaled by k ==", flush=True)
m, _ = load(f"{REPO}/runs/leak/p0/MapWM_s0/MapWM.pt", "MapWM", dev); p = Probe(m); set_pool(m, "test")
with torch.no_grad():
    d = step_of(m, torch.arange(N_SPECIAL, N_SPECIAL + 2 * P)[None])[0]                    # (2P, H, nb)
ident = d - p.mean                                                                          # identity part (test-pool mean)
for k in (1.0, 1.5, 2.0, 3.0):
    p.extra = (k - 1.0) * ident
    a, n = obj_eval(p, toks, revs, "test", "add", dev); a_ms, n_ms = obj_eval(p, toks, revs, "test", "mean", dev)
    res["scale"].append({"k": k, "acc": a, "nll": n, "nll_ms": n_ms, "L_ms": a_ms - a})
    print(f"  k {k:.1f}: eval nll {n:.4f} | nll_ms {n_ms:.4f} | acc {a:.4f} | L_ms {a_ms - a:+.4f}", flush=True)

print("== G3 untrained models ==", flush=True)
for s in (8, 110):
    for arm, cls in GP_ARMS.items():
        torch.manual_seed(s); np.random.seed(s)
        env = NewObjectWorld(size=32, seed=s, pool_size=P, pool="train")
        m = cls(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=32)
        use_object_codes(m, N_SPECIAL, P); m.eval(); set_pool(m, "test"); p = Probe(m)
        r = nlls(p); r.update(arm=arm, seed=s); res["untrained"].append(r)
        print(f"  untrained {arm:9s} s{s}: nll {r['nll']:.4f} | nll_ms {r['nll_ms']:.4f}", flush=True)
print("== G3 pilot checkpoints (30-epoch schedule) ==", flush=True)
for s in (110, 111):
    for arm in GP_ARMS:
        m, b = load(f"{REPO}/runs/gain_phase_pilot/p0/{arm}_s{s}/{arm}.pt", arm, dev); p = Probe(m); set_pool(m, "test")
        r = nlls(p); r.update(arm=arm, seed=s, tail=classify_run(b["losses"])["tail"]); res["pilot"].append(r)
        print(f"  pilot {arm:9s} s{s}: training tail {r['tail']:.4f} | nll {r['nll']:.4f} | nll_ms {r['nll_ms']:.4f} "
              f"(acc {r['acc']:.4f})", flush=True)

L = res["leak"]
print("\n== summary ==")
for arm in L:
    x = np.array([r["nll_ms"] for r in L[arm]]); y = np.array([r["nll"] for r in L[arm]])
    print(f"  LEAK {arm:8s}: nll {y.min():.4f}-{y.max():.4f}, nll_ms {x.min():.4f}-{x.max():.4f}")
sc = res["scale"]
print(f"  G2: nll {sc[0]['nll']:.4f} -> {sc[-1]['nll']:.4f} (k 1 -> 3); nll_ms range "
      f"{min(r['nll_ms'] for r in sc):.4f}-{max(r['nll_ms'] for r in sc):.4f}")
print(f"  G3: untrained nll_ms min {min(r['nll_ms'] for r in res['untrained']):.3f}; pilot nll_ms "
      + " ".join(f"{r['arm']}/{r['seed']} {r['nll_ms']:.3f}" for r in res["pilot"]))
json.dump(res, open(OUT + ".json", "w"), indent=1)
