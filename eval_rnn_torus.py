"""Evaluator entry for RNN_TORUS: runs eval_noise_refine or eval_rank_strata UNCHANGED, with the LSTM / GRU arms of model_rnn
registered into train_variant.VARIANT_MAP (which both evaluators import by reference) first. Anchors (Vanilla_r4, RoPE)
need no registration and evaluate identically through this entry.

    python3 -m mapformer.eval_rnn_torus noise_refine <eval_noise_refine args>
    python3 -m mapformer.eval_rnn_torus strata       <eval_rank_strata args>

Evaluation is in eval mode (dropout off) with TF32 off; the RNNs have no attention-dropout rescale issue (the 1/(1-p)
note in CLAUDE.md concerns attention dropout in one-layer transformers).
"""
import sys

import torch

from mapformer import model_rnn


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("noise_refine", "strata"):
        raise SystemExit("usage: python3 -m mapformer.eval_rnn_torus {noise_refine|strata} <args>")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_rnn.register()
    which = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if which == "noise_refine":
        from mapformer import eval_noise_refine as E
    else:
        from mapformer import eval_rank_strata as E
    E.main()


if __name__ == "__main__":
    main()
