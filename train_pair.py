"""Entry point for MAPPOPE_PAIR_PREREG.md: train_variant.main() unchanged, with the two pairwise-frequency MapPoPE
arms added to VARIANT_MAP (so train_variant.py, which earlier batches md5-guard, is not edited)."""
from mapformer import train_variant
from mapformer.model_pope_pair import MapFormerWM_PoPEPair, MapFormerWM_PoPEPair_r4

train_variant.VARIANT_MAP["MapPoPE-Pair"] = MapFormerWM_PoPEPair
train_variant.VARIANT_MAP["MapPoPE-Pair_r4"] = MapFormerWM_PoPEPair_r4

if __name__ == "__main__":
    train_variant.main()
