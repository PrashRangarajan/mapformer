"""Entry point for SCORE_RANK_PREREG.md: train_variant.main() unchanged, with the two PoPE-score arms added to
VARIANT_MAP (train_variant.py, which earlier batches md5-guard, is not edited). MapPoPE-Pair is the class of
MAPPOPE_PAIR_PREREG.md, unchanged; MapPoPE-Pair_r4mi is its matched-initialisation rank-4 twin (model_pope_pair_mi)."""
from mapformer import train_variant
from mapformer.model_pope_pair import MapFormerWM_PoPEPair
from mapformer.model_pope_pair_mi import MapFormerWM_PoPEPair_r4mi

train_variant.VARIANT_MAP["MapPoPE-Pair"] = MapFormerWM_PoPEPair
train_variant.VARIANT_MAP["MapPoPE-Pair_r4mi"] = MapFormerWM_PoPEPair_r4mi

if __name__ == "__main__":
    train_variant.main()
