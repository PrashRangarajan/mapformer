"""eval_noise_refine with the pairwise-frequency MapPoPE arms registered (MAPPOPE_PAIR_PREREG.md); the eval code is
run unchanged via runpy."""
import runpy
import sys

from mapformer import train_variant
from mapformer.model_pope_pair import MapFormerWM_PoPEPair, MapFormerWM_PoPEPair_r4

train_variant.VARIANT_MAP["MapPoPE-Pair"] = MapFormerWM_PoPEPair
train_variant.VARIANT_MAP["MapPoPE-Pair_r4"] = MapFormerWM_PoPEPair_r4

if __name__ == "__main__":
    sys.argv[0] = "mapformer.eval_noise_refine"
    runpy.run_module("mapformer.eval_noise_refine", run_name="__main__", alter_sys=True)
