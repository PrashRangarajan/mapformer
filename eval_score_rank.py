"""Run an evaluation module unchanged (via runpy) with the SCORE_RANK arms registered in VARIANT_MAP.

    python3 -m mapformer.eval_score_rank mapformer.eval_noise_refine --runs-dir ... (its own arguments)
    python3 -m mapformer.eval_score_rank mapformer.eval_rank_strata --runs-dir ...

Both evaluators import VARIANT_MAP from train_variant at module level, i.e. the dict mutated here."""
import runpy
import sys

from mapformer import train_variant
from mapformer.model_pope_pair import MapFormerWM_PoPEPair
from mapformer.model_pope_pair_mi import MapFormerWM_PoPEPair_r4mi

train_variant.VARIANT_MAP["MapPoPE-Pair"] = MapFormerWM_PoPEPair
train_variant.VARIANT_MAP["MapPoPE-Pair_r4mi"] = MapFormerWM_PoPEPair_r4mi
ALLOWED = ("mapformer.eval_noise_refine", "mapformer.eval_rank_strata")

if __name__ == "__main__":
    mod = sys.argv[1]
    if mod not in ALLOWED:
        raise SystemExit(f"eval_score_rank: module must be one of {ALLOWED}, got {mod}")
    sys.argv = [mod] + sys.argv[2:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
