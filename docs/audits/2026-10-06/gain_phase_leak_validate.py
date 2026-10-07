"""Validation of GAIN_PHASE's leak readouts on the 24 committed LEAK checkpoints (runs/leak/p0, seeds 0-7; eval-only,
before any GAIN_PHASE run). Output: gain_phase_leak_validate_out.txt and gain_phase_leak_validate.json (the per-seed
values the power computation resamples).

Checks:
 V0 wiring: gain_phase_eval's acc and L_zero equal LEAK.json's registered test|x1 'intact' and 'leak' (same stream).
 V1 ActOnly (no object step by construction): L_ms == 0 exactly, S_id == 0.
 V2 MapWM (raw step, shared observation step small): L_ms close to L_zero (per seed), both positive.
 V3 NormStep: L_ms is not the -0.2 gauge artefact of L_zero; S_id ~5x below MapWM (docs/NORMSTEP_NOTES.md 3).
 V4 gauge invariance (NormStep s0-s3, MapWM s0-s1): add c to every observation step (objects AND blank) and -c to every
    action step, c = the model's own shared object step (doubling it): S_id identical, L_ms ~unchanged, L_zero moves.
 V5 positive control (NormStep s0-s3): add a fixed random zero-mean identity-dependent vector to each object's step,
    scaled so S_id equals MapWM's median S_id: S_id reads it, and L_ms becomes positive (the readout detects an
    identity leak of MapWM's size in a model that has none). (a) isotropic: iid over all H x 32 channels;
    (b) a pure field shift: a random per-object displacement (in cells) along the model's own two axis steps -- the
    structure MapWM's identity step has (resid ~0 above).
Amendment 1 (audit, CPU only):
 V6 theta reliance (acc - acc with every step replaced by its token-type mean) on the 24 LEAK checkpoints (large for
    path-integrating models) and on UNTRAINED models of all four GAIN_PHASE arms, built exactly as the trainer builds
    them at seeds 8 and 110 (must read ~0; their L_ms reads ~0 too, which is why D3 needs the convergence gate).
 V7 gain_phase_eval.rescored_acc equals rescore_hook's re-score (install('auto')) on MapWM s0 and NormStep s0.
Run on CPU since Amendment 1 (the first version ran on cuda:0): V0 is then exact only up to CPU/GPU float differences.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
from mapformer.environment_newobj import N_SPECIAL
from mapformer.gain_phase_eval import (load, Probe, obj_acc, step_geometry, step_of, sequences, P, evaluate_model,
                                       rescored_acc)
from mapformer.model_codes import set_pool

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/leak/p0"; OUT = f"{REPO}/docs/audits/2026-10-06/gain_phase_leak_validate"
dev = "cpu"                                     # Amendment 1: CPU only (user's instruction)
torch.set_num_threads(8)
L = json.load(open(f"{REPO}/LEAK.json"))["res"]
data = {pool: sequences(pool) for pool in ("test",)}
toks, revs = data["test"]
res = {}


class Gauge:
    """Adds +c to observation-token steps and -c to action-token steps (hook registered BEFORE the Probe's)."""

    def __init__(self, m):
        self.c, self.tok = None, None
        m.token_emb.register_forward_pre_hook(lambda mod, args: setattr(self, "tok", args[0]))
        m.action_to_lie.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.c is None:
            return out
        obs = (self.tok >= 4)[..., None, None]
        return out + torch.where(obs, self.c, -self.c).to(out.dtype)


def readouts(probe):
    a = obj_acc(probe, toks, revs, "test", None, dev)
    ms = obj_acc(probe, toks, revs, "test", "mean", dev)
    z = obj_acc(probe, toks, revs, "test", "zero", dev)
    tm = obj_acc(probe, toks, revs, "test", "typemean", dev)
    return a, ms - a, z - a, a - tm


print(f"device {dev}; LEAK eval stream: 200 test-pool sequences, T=1024")
for arm in ("MapWM", "ActOnly", "NormStep"):
    res[arm] = []
    for s in range(8):
        m, _ = load(f"{R}/{arm}_s{s}/{arm}.pt", arm, dev); g = Gauge(m); p = Probe(m); set_pool(m, "test")
        a, lms, lz, rel = readouts(p); geo = step_geometry(m)
        row = {"seed": s, "acc": a, "L_ms": lms, "L_zero": lz, "reliance": rel, **geo,
               "leak_json_acc": L[arm][s]["test|x1"]["intact"], "leak_json_L": L[arm][s]["test|x1"]["leak"]}
        if (arm == "NormStep" and s < 4) or (arm == "MapWM" and s < 2):          # V4
            g.c = p.mean.clone(); p.mean = p.mean_step()                          # mean now includes the gauge
            ga, gms, gz, _ = readouts(p); row["gauge"] = {"acc": ga, "L_ms": gms, "L_zero": gz,
                                                        "S_id": step_geometry(m)["S_id"]}
            g.c = None; p.mean = p.mean_step()
        res[arm].append(row)
        print(f"{arm:8s} s{s}: acc {a:.4f} (LEAK.json {row['leak_json_acc']:.4f}) L_zero {lz:+.4f} (LEAK.json "
              f"{row['leak_json_L']:+.4f}) L_ms {lms:+.4f} reliance {rel:.4f} | S_id {geo['S_id']:.4f} shift {geo['shift_cells']:.4f} cells "
              f"resid {geo['resid']:.3f} shared {geo['shared']:.4f} blank {geo['blank']:.4f}"
              + (f" | GAUGE: acc {row['gauge']['acc']:.4f} L_ms {row['gauge']['L_ms']:+.4f} L_zero "
                 f"{row['gauge']['L_zero']:+.4f} S_id {row['gauge']['S_id']:.4f}" if "gauge" in row else ""))

# V5 positive control on NormStep s0-s3
target = float(np.median([r["S_id"] for r in res["MapWM"]]))
res["inject"] = []
for s in range(4):
    m, _ = load(f"{R}/NormStep_s{s}/NormStep.pt", "NormStep", dev); p = Probe(m); set_pool(m, "test")
    H, nb = p.mean.shape
    om = m.path_integrator.omega.detach().cpu().double()
    base = step_geometry(m); a0 = obj_acc(p, toks, revs, "test", None, dev)
    with torch.no_grad():
        ids = torch.arange(4)[None].to(dev); st = step_of(m, ids)[0].detach().cpu().double()   # (4, H, nb) step units
    ux, uy = (st[0] - st[1]) / 2, (st[2] - st[3]) / 2
    for kind in ("isotropic", "shift"):
        gen = torch.Generator().manual_seed(1000 + s)
        if kind == "isotropic":
            u = torch.randn(2 * P, H, nb, generator=gen, dtype=torch.float64)
        else:
            c = torch.randn(2 * P, 2, generator=gen, dtype=torch.float64)
            u = c[:, :1, None] * ux + c[:, 1:, None] * uy
        u[P:] -= u[P:].mean(0)                                                    # zero mean over the test pool
        ph = (u[P:] * om).flatten(1); k = target * base["axis"] / float(ph.norm(dim=1).pow(2).mean().sqrt())
        extra = (u * k).float().to(dev)
        geo = step_geometry(m, extra=extra)
        p.extra = extra
        a_inj = obj_acc(p, toks, revs, "test", "add", dev); a_ms = obj_acc(p, toks, revs, "test", "mean", dev)
        res["inject"].append({"seed": s, "kind": kind, "S_id_before": base["S_id"], "S_id_after": geo["S_id"],
                              "shift_after": geo["shift_cells"], "acc_before": a0, "acc_injected": a_inj,
                              "L_ms_injected": a_ms - a_inj})
        print(f"V5 {kind:9s} NormStep s{s}: S_id {base['S_id']:.4f} -> {geo['S_id']:.4f} (target {target:.4f}), shift "
              f"{geo['shift_cells']:.4f} cells; acc {a0:.4f} -> injected {a_inj:.4f}; L_ms on the injected model "
              f"{a_ms - a_inj:+.4f}")

# V6 untrained models (trainer's construction order), all four GAIN_PHASE arms
from mapformer.environment_newobj import NewObjectWorld
from mapformer.model_codes import use_object_codes
from mapformer.model_gain_phase import ARMS as GP_ARMS
res["untrained"] = []
for s in (8, 110):
    for arm, cls in GP_ARMS.items():
        torch.manual_seed(s); np.random.seed(s)
        env = NewObjectWorld(size=32, seed=s, pool_size=P, pool="train")
        m = cls(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=32)
        use_object_codes(m, N_SPECIAL, P); m = m.to(dev); set_pool(m, "test")
        r = evaluate_model(m, dev, data, full=False); r.update(arm=arm, seed=s); res["untrained"].append(r)
        print(f"V6 untrained {arm:9s} s{s}: acc {r['acc']:.4f} reliance {r['reliance']:+.4f} L_ms {r['L_ms']:+.4f} "
              f"S_id {r['S_id']:.4f}")

g = lambda arm, k: np.array([r[k] for r in res[arm]])
print("\n== checks ==")
v0 = max(max(abs(g(a, "acc") - g(a, "leak_json_acc")).max(), abs(g(a, "L_zero") - g(a, "leak_json_L")).max())
         for a in ("MapWM", "ActOnly", "NormStep"))
print(f"V0 wiring: max |acc, L_zero - LEAK.json| = {v0:.2e} -> {'PASS (exact)' if v0 == 0 else ('PASS (CPU vs the GPU eval of LEAK.json; <= 2e-4 = ~10 of 47,033 targets)' if v0 <= 2e-4 else 'FAIL')}")
v1 = max(abs(g("ActOnly", "L_ms")).max(), abs(g("ActOnly", "S_id")).max())
print(f"V1 ActOnly: max |L_ms|, |S_id| = {v1:.2e} -> {'PASS' if v1 == 0 else 'FAIL'}")
dd = g("MapWM", "L_ms") - g("MapWM", "L_zero")
print(f"V2 MapWM: L_ms mean {g('MapWM', 'L_ms').mean():+.4f} (median {np.median(g('MapWM', 'L_ms')):+.4f}, min "
      f"{g('MapWM', 'L_ms').min():+.4f}) vs L_zero {g('MapWM', 'L_zero').mean():+.4f}; per-seed L_ms - L_zero "
      f"{dd.mean():+.4f} (range {dd.min():+.4f} .. {dd.max():+.4f}); r = {np.corrcoef(g('MapWM', 'L_ms'), g('MapWM', 'L_zero'))[0, 1]:+.3f}")
print(f"V3 NormStep: L_ms mean {g('NormStep', 'L_ms').mean():+.4f} (range {g('NormStep', 'L_ms').min():+.4f} .. "
      f"{g('NormStep', 'L_ms').max():+.4f}) vs L_zero {g('NormStep', 'L_zero').mean():+.4f}; S_id NormStep "
      f"{g('NormStep', 'S_id').mean():.4f} vs MapWM {g('MapWM', 'S_id').mean():.4f} (ratio "
      f"{g('MapWM', 'S_id').mean() / g('NormStep', 'S_id').mean():.1f}x); shift_cells NormStep "
      f"{g('NormStep', 'shift_cells').mean():.4f} vs MapWM {g('MapWM', 'shift_cells').mean():.4f}")
for arm in ("NormStep", "MapWM"):
    for r in res[arm]:
        if "gauge" in r:
            G = r["gauge"]
            print(f"V4 {arm} s{r['seed']}: S_id {r['S_id']:.6f} -> {G['S_id']:.6f}; L_ms {r['L_ms']:+.4f} -> {G['L_ms']:+.4f}; "
                  f"L_zero {r['L_zero']:+.4f} -> {G['L_zero']:+.4f}; acc {r['acc']:.4f} -> {G['acc']:.4f}")
for kind in ("isotropic", "shift"):
    print(f"V5 positive control ({kind}): L_ms on injected NormStep models "
          + " ".join(f"{r['L_ms_injected']:+.4f}" for r in res["inject"] if r["kind"] == kind)
          + f" (MapWM's own L_ms median {np.median(g('MapWM', 'L_ms')):+.4f})")
for arm in ("MapWM", "ActOnly", "NormStep"):
    print(f"V6 reliance, LEAK {arm}: {' '.join(f'{x:.3f}' for x in g(arm, 'reliance'))} (min {g(arm, 'reliance').min():.3f})")
un = np.array([r["reliance"] for r in res["untrained"]])
print(f"V6 reliance, untrained (8 models): max |reliance| {abs(un).max():.4f}; acc max "
      f"{max(r['acc'] for r in res['untrained']):.4f}; L_ms max |.| {max(abs(r['L_ms']) for r in res['untrained']):.4f}; "
      f"S_id {min(r['S_id'] for r in res['untrained']):.3f} .. {max(r['S_id'] for r in res['untrained']):.3f} -> "
      f"{'PASS' if abs(un).max() < 0.02 else 'FAIL'} (must be ~0)")

# V7 last: rescore_hook patches nn.Module.eval for the rest of the process
import mapformer.rescore_hook as RH
mine = {}
for arm in ("MapWM", "NormStep"):
    m, _ = load(f"{R}/{arm}_s0/{arm}.pt", arm, dev); p = Probe(m); set_pool(m, "test")
    mine[arm] = rescored_acc(p, toks, revs, dev)
RH.install("auto")
for arm in ("MapWM", "NormStep"):
    m, _ = load(f"{R}/{arm}_s0/{arm}.pt", arm, dev); p = Probe(m); set_pool(m, "test")
    theirs = obj_acc(p, toks, revs, "test", None, dev)
    print(f"V7 {arm} s0: rescored_acc {mine[arm]:.6f} vs rescore_hook {theirs:.6f} -> {'PASS' if mine[arm] == theirs else 'FAIL'}")
res["rescore_check"] = mine
json.dump(res, open(OUT + ".json", "w"), indent=1)
