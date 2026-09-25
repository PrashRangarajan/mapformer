"""GPU gate + benchmark (adapted from docs/audits/2026-09-24/efficiency/verify_gpu_bitexact.py).

ORIGINAL stack = the pristine pre-change worktree, imported as package `mfbase`; PROPOSED
stack = the repo (`mapformer`). Models are built from the repo's VARIANT_MAP; the ORIGINAL
run swaps in the old WMTransformerLayer.forward, the old GridWorld walk and the old train()
(serial data path, as in the audit script).

Part 1 (gate): rank config (B=16, n_steps=1024, d=128, 1 layer, 2 heads, lr 1e-3, cosine),
N batches from the same seed: losses and every parameter must be bitwise equal, for
Vanilla, Vanilla_r4 and Vanilla_r2ph.
Part 2 (benchmark): seconds per 98-batch epoch and peak memory for original, proposed
bit-exact, proposed + SDPA (TF32 off), and SDPA under three determinism settings; the max
|logit| difference SDPA vs explicit on identical weights; and whether same-seed SDPA
training is run-to-run bitwise reproducible under each setting.
"""
import argparse, os, sys, time, warnings
import numpy as np, torch

SP = __import__("os").environ["MF_SCRATCH"]  # holds base/mapformer: a git worktree of the pre-change commit
sys.path.insert(0, f"{SP}/pk"); sys.path.insert(0, "/home/prashr")
import importlib
# import_module, not `import pkg.train as X`: the package __init__ re-exports the train()
# FUNCTION under the same name, which shadows the submodule attribute
OM, OE, OT = (importlib.import_module(f"mfbase.{m}") for m in ("model", "environment", "train"))
NM, NE, NT = (importlib.import_module(f"mapformer.{m}") for m in ("model", "environment", "train"))
from mapformer.train_variant import VARIANT_MAP
assert hasattr(NM, "_POW2_SCALE_FOLD") and not hasattr(OM, "_POW2_SCALE_FOLD")
NEW = dict(fwd=NM.WMTransformerLayer.forward, gb=NE.GridWorld.generate_batch,
           gt=NE.GridWorld.generate_trajectory)


def use(original: bool):
    if original:
        NM.WMTransformerLayer.forward = OM.WMTransformerLayer.forward
        NE.GridWorld.generate_batch = OE.GridWorld.generate_batch
        NE.GridWorld.generate_trajectory = OE.GridWorld.generate_trajectory
        return OT.train
    NM.WMTransformerLayer.forward = NEW["fwd"]
    NE.GridWorld.generate_batch = NEW["gb"]; NE.GridWorld.generate_trajectory = NEW["gt"]
    return NT.train


def set_det(mode):
    """mode: None | 'warn' (use_deterministic_algorithms(True, warn_only=True)) | 'strict'."""
    if mode is None:
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True, warn_only=(mode == "warn"))


def build(variant, seed, dev):
    env = NE.GridWorld(size=64, n_obs_types=16, p_empty=0.5, seed=seed)
    torch.manual_seed(seed); np.random.seed(seed)
    m = VARIANT_MAP[variant](vocab_size=env.unified_vocab_size, d_model=128, n_heads=2,
                             n_layers=1, grid_size=64)
    return env, m


def run(variant, original, dev, n_batches, sdpa=False, det=None, seed=0):
    OM.USE_SDPA = NM.USE_SDPA = sdpa
    set_det(det)
    train = use(original)
    env, m = build(variant, seed, dev)
    torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev); t = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        losses = train(m, env, n_epochs=1, n_batches=n_batches, batch_size=16, n_steps=1024,
                       lr=1e-3, schedule="cosine", device=str(dev), verbose=False)
    torch.cuda.synchronize(dev)
    msgs = sorted({str(x.message)[:90] for x in w})
    set_det(None)
    return losses, m, time.perf_counter() - t, torch.cuda.max_memory_allocated(dev) / 2**30, msgs


def bitwise(a, b):
    return a[0] == b[0] and all(torch.equal(x, y) for x, y in
                                zip(a[1].state_dict().values(), b[1].state_dict().values()))


@torch.no_grad()
def logit_diff(dev):
    """Same weights, eval mode (no dropout): SDPA vs explicit, max |logit| diff at T=1024."""
    use(False)
    env, m = build("Vanilla", 0, dev); m = m.to(dev).eval()
    np.random.seed(1); tok, _, _, _ = env.generate_batch(16, 1024)
    x = tok[:, :-1].to(dev)
    NM.USE_SDPA = False; a = m(x).float()
    NM.USE_SDPA = True; b = m(x).float()
    NM.USE_SDPA = False
    return float((a - b).abs().max()), float(a.abs().max())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gate-batches", type=int, default=10)
    ap.add_argument("--bench-batches", type=int, default=98)
    ap.add_argument("--part", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args(); dev = torch.device(a.device)
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(dev)}  "
          f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}  "
          f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)
    if 1 in a.part:
        print("== Part 1: bitwise gate (original vs proposed, serial data, explicit attention) ==", flush=True)
        for v in ("Vanilla", "Vanilla_r4", "Vanilla_r2ph"):
            o = run(v, True, dev, a.gate_batches); n = run(v, False, dev, a.gate_batches)
            print(f"  {v:12s} {'BITWISE EQUAL' if bitwise(o, n) else 'DIFFERENT'}  losses {o[0]} / {n[0]}", flush=True)
    if 2 in a.part:
        print(f"== Part 2: seconds per {a.bench_batches}-batch epoch and peak GiB (Vanilla, B16 T1024) ==", flush=True)
        run("Vanilla", False, dev, 5)                           # warm-up (CUDA context, kernels)
        res = {}
        for name, kw in [("original (HEAD)", dict(original=True)),
                         ("proposed bit-exact", dict(original=False)),
                         ("proposed + SDPA", dict(original=False, sdpa=True)),
                         ("proposed + SDPA + det(warn_only)", dict(original=False, sdpa=True, det="warn")),
                         ("proposed + SDPA + det(strict)", dict(original=False, sdpa=True, det="strict"))]:
            try:
                r = run("Vanilla", n_batches=a.bench_batches, dev=dev, **kw); res[name] = r[2]
                print(f"  {name:34s} {r[2]:6.2f} s/epoch   peak {r[3]:5.2f} GiB   loss {r[0][0]:.5f}"
                      + (f"   warnings: {r[4]}" if r[4] else ""), flush=True)
            except Exception as e:
                print(f"  {name:34s} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        if "proposed + SDPA" in res:
            print(f"  SDPA speed-up vs proposed explicit: {res['proposed bit-exact'] / res['proposed + SDPA']:.2f}x;"
                  f" vs HEAD: {res['original (HEAD)'] / res['proposed + SDPA']:.2f}x", flush=True)
        d, s = logit_diff(dev)
        print(f"  max |logit| diff SDPA vs explicit (eval, same weights, B16 T1024): {d:.3e} (max |logit| {s:.2f})", flush=True)
        for det in (None, "warn", "strict"):
            try:
                r1 = run("Vanilla", False, dev, a.gate_batches, sdpa=True, det=det)
                r2 = run("Vanilla", False, dev, a.gate_batches, sdpa=True, det=det)
                print(f"  SDPA run-to-run (deterministic={det}): "
                      f"{'BITWISE REPRODUCIBLE' if bitwise(r1, r2) else 'NOT reproducible'}  losses {r1[0]} / {r2[0]}", flush=True)
            except Exception as e:
                print(f"  SDPA run-to-run (deterministic={det}): FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        for det in ("strict",):
            try:
                r1 = run("Vanilla", False, dev, a.gate_batches, det=det)
                r0 = run("Vanilla", False, dev, a.gate_batches)
                print(f"  explicit path, deterministic={det} vs default: {'BITWISE EQUAL' if bitwise(r0, r1) else 'DIFFERENT'}", flush=True)
            except Exception as e:
                print(f"  explicit path, deterministic={det}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
