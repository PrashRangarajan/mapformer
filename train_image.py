#!/usr/bin/env python3
"""Trainer for the image-observation torus (IMAGE_TORUS_PREREG.md). A separate entry point: train_variant.py, train.py
and environment.py are used unchanged (two registered batches md5-guard them).

The environment is GridWorld exactly as train_variant builds it for the headline torus (64x64, 16 classes, p_empty
0.5, map seeded by --seed, fixed for the run; walks fresh every batch); train() is called unchanged. --obs image
swaps each observation token for an image of its class (model_image.ImageObsEmbedding, train split only); --obs
token is the unmodified model (the token twin). The recipe defaults are FACTORIAL's: 900 epochs x 98 batches of 16
walks of 1024 steps, lr 1e-3, AdamW wd 0.05, warmup + cosine, --data-workers 3; --fast-attn as there.

  python3 -m mapformer.train_image --arm MapWM --obs image --seed 3100 --fast-attn --device cuda:0 \
      --output-dir runs/image_torus/I_MapWM_s3100
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapformer.environment import GridWorld
from mapformer.train import train
from mapformer import model_image as MI
from mapformer import image_bank


def make_env(seed, grid_size=64):
    """THE training environment (Amendment 1, m5: the gate and the tests call this, not a copy): GridWorld exactly as
    train_variant builds it for the headline torus."""
    env = GridWorld(size=grid_size, n_obs_types=MI.N_CLS, p_empty=0.5, n_landmarks=0, seed=seed)
    assert env.unified_vocab_size == MI.VOCAB and env.unified_blank == MI.BLANK
    return env


def check_interleaving(env, n_steps):
    """One-time check that the stream is [action, observation, ...] (the image embedding relies on it in training).
    Restores the global numpy RNG, so the training walk stream is unchanged (Amendment 1, n4)."""
    st = np.random.get_state()
    try:
        tok = env.generate_batch(2, min(n_steps, 64))[0]
    finally:
        np.random.set_state(st)
    assert bool((tok[:, 0::2] < MI.N_ACT).all()) and bool((tok[:, 1::2] >= MI.N_ACT).all()), "stream not interleaved"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=MI.ARMS)
    ap.add_argument("--obs", required=True, choices=["image", "token"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-batches", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--data-workers", type=int, default=3)
    ap.add_argument("--grid-size", type=int, default=64)
    ap.add_argument("--fast-attn", action="store_true", help="as train_variant: SDPA in WMTransformerLayer, TF32 off")
    ap.add_argument("--deterministic", action="store_true", help="as train_variant")
    ap.add_argument("--enc-gain", type=float, default=MI.ENC_GAIN,
                    help="fixed encoder-output gain for image arms (default ENC_GAIN = 30.3729, Amendment 1; 1.0 = no "
                         "gain). A non-default gain is part of the cell name (_g<gain>) and the checkpoint config")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="cudnn autotuning for the encoder convs (pilot decides; off by default)")
    ap.add_argument("--bank", default=image_bank.CACHE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if args.fast_attn:
        import mapformer.model as _M
        _M.USE_SDPA = True
    # TF32 off on every path (train_variant turns it off under --fast-attn; the image arms add cudnn convs, and
    # TF32 convs would make the image and token twins differ in numerics as well as input)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    torch.manual_seed(args.seed); np.random.seed(args.seed)            # as train_variant, same order
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    env = make_env(args.seed, args.grid_size)
    check_interleaving(env, args.n_steps)
    banks = image_bank.load_banks(args.bank) if args.obs == "image" else None
    bank_md5 = image_bank.bank_md5(args.bank) if args.obs == "image" else None
    if args.obs == "token" and abs(args.enc_gain - MI.ENC_GAIN) > 1e-6:
        raise SystemExit("--enc-gain applies to image arms only")
    model = MI.build(args.arm, args.obs, args.seed, n_layers=args.n_layers, banks=banks, enc_gain=args.enc_gain,
                     grid_size=args.grid_size, seed_rng=True)
    if args.fast_attn:
        from mapformer.model import WMTransformerLayer as _WML
        plain = [m for m in model.modules() if type(m).forward is _WML.forward]
        if not plain:
            raise SystemExit(f"--fast-attn changes nothing for {args.arm}")
    cell = MI.cell_name(args.arm, args.obs, args.n_layers, args.enc_gain)
    n_enc = sum(p.numel() for p in model.token_emb.encoder.parameters()) if args.obs == "image" else 0
    print(f"{cell} seed={args.seed} params={sum(p.numel() for p in model.parameters()):,} (encoder {n_enc:,}) "
          f"bank {bank_md5}")
    t0 = time.time()
    losses = train(model, env, n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, n_steps=args.n_steps,
                   n_batches=args.n_batches, device=args.device, schedule=args.schedule,
                   data_workers=args.data_workers, weight_decay=args.weight_decay)
    wall = time.time() - t0
    emb = MI.image_emb(model)
    draws = dict(emb.draws) if emb is not None else None
    if emb is not None:
        assert draws["test"] == 0, draws                                  # never a test image in training
    ck = out / f"{cell}.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "losses": losses, "variant": cell, "seed": args.seed,
        "config": {"enc_gain": float(MI.image_emb(model).enc_gain) if emb is not None else None, "arm": args.arm, "obs": args.obs, "n_layers": args.n_layers, "vocab_size": MI.VOCAB,
                   "d_model": 128, "n_heads": 2, "grid_size": args.grid_size, "n_obs_types": MI.N_CLS,
                   "p_empty": 0.5, "n_steps": args.n_steps, "epochs": args.epochs, "lr": args.lr,
                   "schedule": args.schedule, "n_batches": args.n_batches, "batch_size": args.batch_size,
                   "fast_attn": args.fast_attn, "deterministic": args.deterministic, "bank_md5": bank_md5,
                   "train_draws": draws, "wall_s": wall, "torch": torch.__version__, "args": dict(vars(args))},
    }, ck)
    print(f"Saved: {ck}  ({wall / max(1, args.epochs):.2f} s/epoch)")


if __name__ == "__main__":
    main()
