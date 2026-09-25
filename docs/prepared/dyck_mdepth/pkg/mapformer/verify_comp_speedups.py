"""Prove the two speedups before any experiment uses them.

They are NOT the same kind of change and must not be licensed the same way.

  --fast-attn      SDPA + TF32. MATHEMATICALLY EQUIVALENT: the same function, a
                   faster kernel. Checked here on logits and gradients. Because it
                   is equivalent, a fast-attn run stays comparable to a stored
                   non-fast-attn checkpoint. NOT valid for MapEM, whose Hadamard
                   A_X (*) A_P cannot be expressed as SDPA.

  --data-workers   Generates batches in parallel. Batch i is seeded by its INDEX,
                   so the stream is byte-identical for ANY worker count -- but it
                   DIFFERS from the serial path's stream. This is reproducible
                   among data-worker runs and NOT comparable to a stored serial
                   checkpoint. It is a change of data, not a change of kernel, and
                   any batch mixing the two is a between-code comparison.

The compositional headroom batch (runs/comp_headroom) was trained serially without
fast-attn. Under the rule above, later fast-attn runs remain comparable to it and
later data-worker runs do not.
"""
import argparse, time
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="Hourglass_k2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    dev = torch.device(a.device)

    import mapformer.model as M
    from mapformer.train_variant import VARIANT_MAP
    from mapformer.environment_compositional import CompositionalGridWorld

    env = CompositionalGridWorld(seed=0)
    rng = np.random.RandomState(0)
    batch = env.generate_batch(a.batch_size, a.n_steps)
    tok = batch[0].to(dev)

    def run(use_sdpa):
        M.USE_SDPA = use_sdpa
        torch.backends.cuda.matmul.allow_tf32 = use_sdpa
        torch.backends.cudnn.allow_tf32 = use_sdpa
        torch.manual_seed(0)
        m = VARIANT_MAP[a.variant](vocab_size=env.unified_vocab_size, d_model=128,
                                   n_heads=2, n_layers=3, grid_size=64).to(dev)
        m.train()
        out = m(tok[:, :-1])
        out.sum().backward()
        g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
        return out.detach(), g.detach()

    o0, g0 = run(False)
    o1, g1 = run(True)
    dl = (o0 - o1).abs().max().item()
    dg = (g0 - g1).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(g0.unsqueeze(0), g1.unsqueeze(0)).item()

    print(f"variant           {a.variant}")
    print(f"max |logit diff|  {dl:.3e}")
    print(f"max |grad diff|   {dg:.3e}")
    print(f"grad cosine       {cos:.10f}")

    # timing, same batch, several passes
    for flag in (False, True):
        M.USE_SDPA = flag
        torch.backends.cuda.matmul.allow_tf32 = flag
        torch.manual_seed(0)
        m = VARIANT_MAP[a.variant](vocab_size=env.unified_vocab_size, d_model=128,
                                   n_heads=2, n_layers=3, grid_size=64).to(dev)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(20):
            m(tok[:, :-1]).sum().backward()
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated(dev) / 2**20
        print(f"{'fast-attn' if flag else 'baseline ':9s}  {time.time()-t0:.2f}s / 20 passes   peak {mem:.0f} MiB")
        torch.cuda.reset_peak_memory_stats(dev)

    ok = dl < 1e-4 and cos > 0.9999
    print("\nVERDICT:", "EQUIVALENT -- safe to enable" if ok else "NOT EQUIVALENT -- do not enable")


if __name__ == "__main__":
    main()
