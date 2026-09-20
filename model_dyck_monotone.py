"""T2 (THEORY_MAPPOPE.md): force a CLOCK accumulator inside Dyck-2.

The clock/map boundary -- a per-token phase pays where the accumulator leaves its trained range, and
not where it is bounded -- rests on comparing Bach (alpha 1.00) with Dyck and the torus (bounded).
Those differ in dataset, model size and metric as well as in accumulator, so the boundary is a
BETWEEN-TASK association. T2 is the within-task version: on Dyck the increments cancel because opens
and closes push the accumulator in opposite directions, so constraining them to be non-negative
turns the same task's accumulator into a clock while changing nothing else.

Uses the repo's existing `SignConstrainedActionToLie(mode="abs")`, built inside `fork_rng` and loaded
with the signed parent's weights, so at seed s each arm starts as the same function as its parent up
to the absolute value -- the construction already used by `model_monotone.py`.
"""
import torch

from mapformer.model import MapFormerWM
from mapformer.model_pope import MapFormerWM_PoPE
from mapformer.model_pope_t3 import (MapFormerWM_PoPE_T3, MapFormerWM_PoPE_T3_PI01,
                                     MapFormerWM_PoPE_T3_Inert)
from mapformer.model_sign import SignConstrainedActionToLie


def _monotone(parent, name):
    class _M(parent):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1,
                     grid_size=64, bottleneck_r=2, **kw):
            super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                             grid_size, bottleneck_r, **kw)
            old = self.action_to_lie
            with torch.random.fork_rng(devices=[]):
                new = SignConstrainedActionToLie(d_model, n_heads, self.n_blocks,
                                                 old.w_in.out_features, mode="abs")
            new.load_state_dict(old.state_dict())
            self.action_to_lie = new
    _M.__name__ = name
    _M.__doc__ = f"{parent.__name__} with the phase increment forced non-negative (a clock)."
    return _M


MapFormerWM_Abs = _monotone(MapFormerWM, "MapFormerWM_Abs")
MapFormerWM_PoPE_Abs = _monotone(MapFormerWM_PoPE, "MapFormerWM_PoPE_Abs")


# T2b: does the per-token phase pay once Dyck's accumulator is a clock? (T2_RESULTS.md closing test)
MapFormerWM_PoPE_T3_PI01_Abs = _monotone(MapFormerWM_PoPE_T3_PI01, "MapFormerWM_PoPE_T3_PI01_Abs")
MapFormerWM_PoPE_T3_Inert_Abs = _monotone(MapFormerWM_PoPE_T3_Inert, "MapFormerWM_PoPE_T3_Inert_Abs")

# T2c: the missing initialisation control -- phase live but ZERO-initialised, so "phase - inert twin"
# is not bundled with a 0.1-scale perturbation of the starting function (audit, 2026-09-20).
MapFormerWM_PoPE_T3_Zero_Abs = _monotone(MapFormerWM_PoPE_T3, "MapFormerWM_PoPE_T3_Zero_Abs")
