"""The 8th cell: PoPE encoding + INDEX position + hierarchy.

The MiniGrid factorial (`MINIGRID_2X2X2_n8.md`) is a 2x2x2 over
{RoPE, PoPE} x {index, path-integrated} x {flat, hierarchical} with one cell
missing -- no PoPE+index+hierarchy variant existed. That matters more than a
tidiness gap: at n=8, **PoPE-Flat (index, flat) is the best arm on that
benchmark** at T=1024, and its hierarchy pair is precisely the cell that did not
exist.

FREQUENCY PARAMETERISATION, decided rather than defaulted. `MapPoPE-Hier` uses a
learnable `omega`; `PoPE-Flat` uses a fixed `theta_c = base^(-c/d)`. The new cell
has to pick one, and the pick is determined rather than arbitrary:

  fixed theta_c (chosen)        learnable omega
  ---------------------------   ---------------------------
  hierarchy pair vs PoPE-Flat:  hierarchy pair: CONFOUNDED
    clean (both fixed, index)     (fixed vs learnable)
  encoding pair vs RoPE-Hier:   encoding pair: CONFOUNDED
    clean (both fixed, index)
  position pair vs MapPoPE-Hier: position pair: clean
    confounded -- but IDENTICALLY
    to every other position pair
    in the grid, so consistent

Two clean pairs and one confound that is already uniform across the factorial
beats one clean pair and one inconsistently-confounded pair. `FREQ_CONTROL.md`
separately measures that confound and finds it empirically negligible
(+0.004 / -0.008).

PARAMETER COUNT. Dropping the path-integration machinery removes
`action_to_lie` (512) and `omega` (128) = 640 parameters, giving 614,474 against
`MapPoPE-Hier`'s 615,114. That is exactly the same 640-parameter deficit
`PoPE-Flat` already carries against `MapPoPE-Flat` in the published grid, so the
new cell is matched to its own row rather than introducing a new asymmetry.
"""
import torch
import torch.nn as nn

from .model_pope import MapFormerWM_Hourglass_PoPE


class _IndexDelta(nn.Module):
    """Emits 1 for every token, so the forward pass's cumsum yields the
    sequence index. Parameter-free, replacing the learned action->Lie map."""

    def __init__(self, n_heads: int, n_blocks: int):
        super().__init__()
        self.n_heads, self.n_blocks = n_heads, n_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.new_ones(B, T, self.n_heads, self.n_blocks)


class MapFormerWM_Hourglass_PoPE_Index(MapFormerWM_Hourglass_PoPE):
    """PoPE + sequence-index position + single-level hourglass.

    NOTE (2026-08-28): this arm is CORRECTLY invariant to bottleneck_r. It
    replaces action_to_lie with _IndexDelta, which has no learned action
    subspace, so there is no rank to set -- unlike the sibling PoPE hourglass
    classes, where r was being silently swallowed (fixed in model_pope.py).
    Do not "fix" the invariance here; it is what the index arm means.
    """

    def __init__(self, *a, base: float = 10000.0, **kw):
        super().__init__(*a, **kw)
        # position becomes the sequence index at both resolutions: the fine
        # level cumsums ones, and the coarse level pools that ramp, giving the
        # mean index of each segment -- the natural coarse index.
        self.action_to_lie = _IndexDelta(self.n_heads, self.n_blocks)
        c = torch.arange(self.n_blocks, dtype=torch.float32)
        theta = (base ** (-c / self.n_blocks))
        with torch.no_grad():
            self.path_integrator.omega.copy_(
                theta.unsqueeze(0).expand(self.n_heads, -1))
        # frozen AND removed from the parameter list, so the count matches
        # PoPE-Flat's relationship to MapPoPE-Flat rather than inventing a new one
        omega = self.path_integrator.omega.detach().clone()
        del self.path_integrator.omega
        self.path_integrator.register_buffer("omega", omega)
