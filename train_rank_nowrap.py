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
  Vanilla_r2ph_om32_redraw : the rank-2 model, trained on environment_nd_redraw.GridWorldNDRedraw (observation map
      redrawn every trajectory; Amendment 1 memorisation control). The variant name selects the environment: main()
      replaces environment_nd.GridWorldND, which train_variant imports inside main() at call time, by the redraw class
      for this variant only. Evaluation uses the stock GridWorldND (fixed held-out map, env seed 10000) for every arm.
"""
import sys

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
train_variant.VARIANT_MAP["Vanilla_r2ph_om32_redraw"] = MapFormerWM_PerHead_Om32
REDRAW_SUFFIX = "_redraw"


def main():
    v = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else ""
    if v.endswith(REDRAW_SUFFIX):
        from mapformer import environment_nd
        from mapformer.environment_nd_redraw import GridWorldNDRedraw
        assert "--env" in sys.argv and sys.argv[sys.argv.index("--env") + 1] == "nd", "redraw arms need --env nd"
        environment_nd.GridWorldND = GridWorldNDRedraw
        print(f"{v}: training environment = GridWorldNDRedraw (map redrawn per trajectory)")
    train_variant.main()
    record_config(v)


def record_config(v):
    """Amendment 2: write the arm's environment and omega initialisation into the checkpoint's config (train_variant saves
    only the variant name). Atomic: written to a temporary file in the same directory, then os.replace'd."""
    import os
    import torch
    out = sys.argv[sys.argv.index("--output-dir") + 1]
    ck = os.path.join(out, f"{v}.pt")
    b = torch.load(ck, map_location="cpu", weights_only=False)
    b["config"]["map_redrawn"] = v.endswith(REDRAW_SUFFIX)
    b["config"]["train_env_class"] = "GridWorldNDRedraw" if v.endswith(REDRAW_SUFFIX) else "GridWorldND"
    b["config"]["omega_init_grid"] = OMEGA_GRID
    tmp = ck + ".tmp"
    torch.save(b, tmp); os.replace(tmp, ck)
    print(f"recorded in config: map_redrawn={b['config']['map_redrawn']}, omega_init_grid={OMEGA_GRID}")


if __name__ == "__main__":
    main()
