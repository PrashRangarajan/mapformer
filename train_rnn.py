"""Trainer entry for the RNN_TORUS arms: train_variant's own CLI, data, loss, recipe and checkpoint, with the LSTM / GRU
arms of model_rnn registered into its VARIANT_MAP first. Nothing in train_variant.py or train.py changes.

    python3 -m mapformer.train_rnn --variant LSTM_L1_h157 --seed S --epochs 900 --lr LR --n-batches 98 --batch-size 16 \
        --n-steps 1024 --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0 --schedule cosine --data-workers 3 \
        --device cuda:G --output-dir OUT

Differences from a transformer run, all forced here (RNN_TORUS_PREREG.md, Recipe):
- no --fast-attn (train_variant refuses it for a model with no WMTransformerLayer) and no --deterministic;
- cuDNN TF32 OFF (torch.backends.cudnn.allow_tf32 defaults to True, which would run the LSTM / GRU matmuls in TF32; the
  anchors run with --fast-attn, which turns both TF32 switches off, so all arms are fp32);
- only the registered RNN arms are accepted (anchors are launched through mapformer.train_variant itself).
"""
import sys

import torch

from mapformer import model_rnn


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    v = argv[argv.index("--variant") + 1] if "--variant" in argv else None
    if v not in model_rnn.RNN_VARIANTS:
        raise SystemExit(f"train_rnn: --variant must be one of {sorted(model_rnn.RNN_VARIANTS)} (got {v}); "
                         f"run anchors through mapformer.train_variant")
    for bad in ("--fast-attn", "--deterministic", "--init-from"):
        if bad in argv:
            raise SystemExit(f"train_rnn: {bad} is not part of the registered RNN recipe")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_rnn.register()
    from mapformer import train_variant
    print(f"train_rnn: {v} {model_rnn.ARM_SPECS[v]}; cudnn {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}, "
          f"cudnn.allow_tf32 {torch.backends.cudnn.allow_tf32}, matmul.allow_tf32 {torch.backends.cuda.matmul.allow_tf32}",
          flush=True)
    sys.argv = [sys.argv[0]] + argv
    train_variant.main()


if __name__ == "__main__":
    main()
