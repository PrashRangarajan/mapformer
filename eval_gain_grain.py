"""Run an evaluation module unchanged (via runpy) with the GAIN_GRAIN arms registered in VARIANT_MAP.

    python3 -m mapformer.eval_gain_grain mapformer.eval_noise_refine --runs-dir ... (its own arguments)

eval_noise_refine imports VARIANT_MAP from train_variant at module level, i.e. the dict mutated here."""
import runpy
import sys

from mapformer import train_variant
from mapformer.model_em_pope import register

register(train_variant.VARIANT_MAP)
ALLOWED = ("mapformer.eval_noise_refine",)

if __name__ == "__main__":
    mod = sys.argv[1]
    if mod not in ALLOWED:
        raise SystemExit(f"eval_gain_grain: module must be one of {ALLOWED}, got {mod}")
    sys.argv = [mod] + sys.argv[2:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
