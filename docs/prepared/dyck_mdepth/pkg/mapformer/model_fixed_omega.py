"""Frequency control: MapFormer-WM with omega FROZEN at its geometric init.

Why this exists
---------------
Every "position effect" reported in `INDEX_BASELINE_PAPER_TASK_n8.md`,
`MINIGRID_2X2.md` and `MINIGRID_2X2X2.md` compares a path-integrated arm against
an index arm -- but those two groups also differ in a second way nobody
controlled:

    Vanilla, MapPoPE-*    omega is nn.Parameter        LEARNABLE frequencies
    RoPE, PoPE-Flat       inv_freq/theta_c is a buffer FIXED frequencies

So the factor labelled "position" is really "path-integration AND frequency
learning" bundled together.

On the torus this does not threaten anything: the effect is +0.461 and the index
arms sit exactly on the measured blank floor, which frequency learning cannot
explain -- a model that has learned nothing about the map has not been held back
by its frequency schedule. On MiniGrid the position effect is -0.012 to -0.037,
small enough that a confound of this size could account for the whole thing, and
the sign of that number is currently doing real work in the argument.

What this isolates, and what it does not
----------------------------------------
`Vanilla_FixedOmega` is `MapFormerWM` with omega frozen at the same geometric
init it would otherwise start from, everything else identical.

  Vanilla vs Vanilla_FixedOmega   isolates FREQUENCY LEARNING exactly: same
                                  architecture, same init, same path
                                  integration, one parameter group frozen.
  Vanilla_FixedOmega vs RoPE      is closer to a pure position comparison, but
                                  NOT exact -- the two still use different
                                  frequency SCHEDULES (PathIntegrator's
                                  omega_max*(1/grid)^(i/(nb-1)) vs RoPE's
                                  base^(-k/(nb-1))). It removes the
                                  learned-vs-fixed difference, not the schedule
                                  difference.

Stated rather than glossed, because a control that is described as isolating
more than it does is worse than no control.
"""
import torch.nn as nn

from .model import MapFormerWM


class MapFormerWM_FixedOmega(MapFormerWM):
    """MapFormer-WM, path integration intact, angular velocities frozen."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.path_integrator.omega.requires_grad_(False)
