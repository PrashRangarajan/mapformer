"""Evaluator entry for EXPTORUS_PREREG.md: eval_noise_refine or eval_rank_strata, unchanged, with ExpAO4_r4 /
SoftplusAO4_r4 registered in train_variant.VARIANT_MAP first (both evaluators build models from that dict).

  python3 -m mapformer.eval_exptorus noise_refine <eval_noise_refine's arguments>
  python3 -m mapformer.eval_exptorus strata       <eval_rank_strata's arguments>
"""
import sys

from mapformer.model_exptorus import register


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("noise_refine", "strata"):
        raise SystemExit("usage: python3 -m mapformer.eval_exptorus {noise_refine|strata} ARGS...")
    which = sys.argv.pop(1)
    register()
    if which == "noise_refine":
        from mapformer.eval_noise_refine import main as m
    else:
        from mapformer.eval_rank_strata import main as m
    m()


if __name__ == "__main__":
    main()
