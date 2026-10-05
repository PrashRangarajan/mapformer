"""Entry point for GAIN_GRAIN_PREREG.md: train_variant.main() unchanged, with MapPoPE-Pair (model_pope_pair,
unchanged) and the new arms GainScalar / GainMod4 / VanillaEM_NonNeg (model_em_pope) added to VARIANT_MAP
(train_variant.py, which earlier batches md5-guard, is not edited). Vanilla and VanillaEM are train_variant's own."""
from mapformer import train_variant
from mapformer.model_em_pope import register

register(train_variant.VARIANT_MAP)

if __name__ == "__main__":
    train_variant.main()
