"""GPU gate + benchmark for the proposed efficiency patches. NOT RUN by the audit
(both GPUs were busy); run it on an IDLE GPU before applying anything:

    cd /home/prashr && PYTHONPATH=/home/prashr python3 \
        <scratch>/audit_efficiency/verify_gpu_bitexact.py --device cuda:0

Part 1 (the gate): rank config (B=16, n_steps=1024, d=128, 1 layer, 2 heads, lr 1e-3,
cosine), N batches from the same seeds, ORIGINAL stack vs PROPOSED bit-exact stack
(environment fast walk + model scale fold + sync-free loop). Losses and every
parameter must be bitwise equal, for Vanilla and Vanilla_r4. Any difference means
the scale fold is not bit-exact on this GPU's GEMM kernels: set
model._POW2_SCALE_FOLD = False and re-run (the loop and walk changes stand alone).

Part 2 (the benchmark): seconds per 98-batch epoch and peak memory for original,
proposed bit-exact, and proposed + SDPA (not bit-exact), and whether SDPA training
is run-to-run bitwise reproducible with and without --deterministic.
"""
import argparse, copy, importlib, sys, time
import numpy as np, torch

SCR = "/tmp/claude-1002/-home-prashr-mapformer/11c678ec-9c7c-4954-8b14-36979f03e955/scratchpad/audit_efficiency"
sys.path.insert(0, SCR)
OM = importlib.import_module("mapformer.model"); OE = importlib.import_module("mapformer.environment")
OT = importlib.import_module("mapformer.train")
NM = importlib.import_module("mfprop.model"); NE = importlib.import_module("mfprop.environment")
NT = importlib.import_module("mfprop.train")
from mapformer.train_variant import VARIANT_MAP

ORIG = dict(fwd=OM.WMTransformerLayer.forward, gb=OE.GridWorld.generate_batch,
            gt=OE.GridWorld.generate_trajectory)


def use(proposed: bool):
    """Swap the three patched code paths in or out of the live mapformer modules."""
    if proposed:
        OM.WMTransformerLayer.forward = NM.WMTransformerLayer.forward
        OE.GridWorld._fast_walk_ok = NE.GridWorld._fast_walk_ok
        OE.GridWorld.generate_batch = NE.GridWorld.generate_batch
        OE.GridWorld.generate_trajectory = NE.GridWorld.generate_trajectory
        # the patched methods resolve helpers through NE's globals
        NE.GridWorld = OE.GridWorld
    else:
        OM.WMTransformerLayer.forward = ORIG["fwd"]
        OE.GridWorld.generate_batch = ORIG["gb"]; OE.GridWorld.generate_trajectory = ORIG["gt"]
    return NT.train if proposed else OT.train


def run(variant, proposed, dev, n_batches, sdpa=False, det=False, seed=0):
    NM.USE_SDPA = OM.USE_SDPA = sdpa
    torch.use_deterministic_algorithms(det, warn_only=True)
    train = use(proposed)
    env = OE.GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    m = VARIANT_MAP[variant](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2,
                             n_layers=1, grid_size=64)
    torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev); t = time.perf_counter()
    losses = train(m, env, n_epochs=1, n_batches=n_batches, batch_size=16, n_steps=1024,
                   lr=1e-3, schedule="cosine", device=str(dev), verbose=False)
    torch.cuda.synchronize(dev)
    return losses, m, time.perf_counter() - t, torch.cuda.max_memory_allocated(dev) / 2**30


def bitwise(a, b):
    return a[0] == b[0] and all(torch.equal(x, y) for x, y in
                                zip(a[1].state_dict().values(), b[1].state_dict().values()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gate-batches", type=int, default=10)
    ap.add_argument("--bench-batches", type=int, default=98)
    a = ap.parse_args(); dev = torch.device(a.device)
    print("== Part 1: bitwise gate ==")
    for v in ("Vanilla", "Vanilla_r4"):
        o = run(v, False, dev, a.gate_batches); n = run(v, True, dev, a.gate_batches)
        print(f"  {v:11s} original vs proposed: {'BITWISE EQUAL' if bitwise(o, n) else 'DIFFERENT'}"
              f"  losses {o[0]} / {n[0]}")
    print("== Part 2: seconds per epoch (98 batches) and peak GiB ==")
    for name, kw in [("original", dict(proposed=False)), ("proposed bit-exact", dict(proposed=True)),
                     ("proposed + SDPA", dict(proposed=True, sdpa=True)),
                     ("proposed + SDPA + deterministic", dict(proposed=True, sdpa=True, det=True))]:
        r = run("Vanilla", n_batches=a.bench_batches, dev=dev, **kw)
        print(f"  {name:32s} {r[2]:6.2f} s/epoch   peak {r[3]:5.2f} GiB   loss {r[0][0]:.5f}")
    for det in (False, True):
        r1 = run("Vanilla", True, dev, a.gate_batches, sdpa=True, det=det)
        r2 = run("Vanilla", True, dev, a.gate_batches, sdpa=True, det=det)
        print(f"  SDPA run-to-run (deterministic={det}): "
              f"{'BITWISE REPRODUCIBLE' if bitwise(r1, r2) else 'NOT reproducible'}")


if __name__ == "__main__":
    main()
