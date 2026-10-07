"""Entry point for RANK_NOWRAP_PREREG.md: train_variant.main() unchanged, with two arms added to VARIANT_MAP whose
omega initialisation is FIXED at the grid-32 values whatever --grid-size the environment uses (train_variant.py,
model.py and model_rank_perhead.py, which earlier batches md5-guard, are not edited).

PathIntegrator initialises omega_i = 2 pi * (1/grid_size)^(i/(n_b-1)); train_variant passes the ENVIRONMENT's grid
size, so on a larger torus the stock arms would also start with slower frequencies. Here the model is built with
grid_size = OMEGA_GRID = 32 always, so at a given seed every initial weight (omega included) is identical across the
torus sizes of the batch, and on the 32-torus the arms are bit-identical to `Vanilla_r2ph` / `Vanilla_r3ph`
(the RANK_ND D=2 cells). Only the training data (map, wrap-around) differ between grids. omega is a trained
parameter saved in the state dict, so evaluators that rebuild the model with the checkpoint's grid_size load the
trained omega regardless.

  Vanilla_r2ph_om32 : per-head rank 2 (model_rank_perhead.MapFormerWM_PerHead), omega init of grid 32
  Vanilla_r3ph_om32 : per-head rank 3 (PER_HEAD_R = 3, as MapFormerWM_PerHead3), omega init of grid 32
"""
from mapformer import train_variant
from mapformer.model_rank_perhead import MapFormerWM_PerHead

OMEGA_GRID = 32


class MapFormerWM_PerHead_Om32(MapFormerWM_PerHead):
    PER_HEAD_R = 2

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64,
                 bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, OMEGA_GRID, bottleneck_r, **kw)


class MapFormerWM_PerHead3_Om32(MapFormerWM_PerHead_Om32):
    PER_HEAD_R = 3


train_variant.VARIANT_MAP["Vanilla_r2ph_om32"] = MapFormerWM_PerHead_Om32
train_variant.VARIANT_MAP["Vanilla_r3ph_om32"] = MapFormerWM_PerHead3_Om32

if __name__ == "__main__":
    train_variant.main()
