"""The two Selective RoPE components MapFormer lacks, ported to navigation.

Selective RoPE (arXiv:2511.17388, ICLR 2026) and MapFormer (arXiv:2511.19279) were
posted three days apart with the same primitive -- position as a cumulative sum of
input-derived angles applied as a rotation -- and neither cites the other. Selective
RoPE carries two components MapFormer does not, and NEITHER has been tested in the
other's domain. These are those two, added surgically to MapFormerWM.

--------------------------------------------------------------------------------
1. CONV-DELTA  (SRoPE's conv1d, which MapFormer has no analogue of)

SRoPE:  omega = temp * cumsum(conv1d(W_omega @ q))
MapFormer: theta = cumsum(omega * W_out W_in x)          <- no filter

Why the filter is interesting rather than incidental: cumsum and a difference
filter are inverses. If the conv learns [.., 1, -1] then cumsum(diff(d)) = d and
there is NO accumulation -- position becomes CURRENT content. If it learns the
identity, you get FULL accumulation. So the conv1d effectively LEARNS HOW MUCH TO
ACCUMULATE, interpolating between "position = what I see now" and "position =
everything I have integrated".

PREDICTION for navigation: position IS the integral of displacement, so full
accumulation is exactly correct and the conv should be UNNECESSARY -- the model
should learn it back toward the identity. If that holds it explains the divergence:
MapFormer omits the filter and is fine on navigation; SRoPE needs it for language,
where how-much-to-accumulate is genuinely uncertain. If instead the conv HELPS on
navigation, MapFormer is leaving something on the table.

Initialised to the IDENTITY (causal kernel [0,0,1]) so at step 0 this is exactly
MapFormerWM and any deviation is learned.

--------------------------------------------------------------------------------
2. GATE-DELTA  (SRoPE's sigmoid gate -- its best-performing addition on MAD)

SRoPE adds "a sigmoid gate on the rotation angles to allow the model to control
whether to rotate or not". MapFormer has no gate.

Why it should matter HERE specifically: our token stream alternates
[action, obs, action, obs, ...] and ONLY ACTIONS should displace the agent.
MapFormer has to learn Delta ~= 0 for observation tokens implicitly, through the
rank-2 bottleneck. An explicit per-token gate makes that a single sigmoid. This is
the cleanest possible test of whether that implicit burden costs anything.

Gate bias initialised to +4 (sigmoid ~= 0.982), so it starts near pass-through;
the residual ~1.8% attenuation is absorbable by the learnable omega.
--------------------------------------------------------------------------------
Both keep every other part of MapFormerWM identical, so each isolates one component.
"""
import torch
import torch.nn as nn

from mapformer.model import MapFormerWM, PathIntegrator


class PathIntegratorConv(PathIntegrator):
    """PathIntegrator + a causal depthwise conv1d on Delta BEFORE the cumsum."""

    def __init__(self, n_heads, n_blocks, grid_size=64, kernel_size=3):
        super().__init__(n_heads, n_blocks, grid_size)
        self.kernel_size = kernel_size
        # one filter per (head, block) channel; identity init -> exactly MapFormer
        w = torch.zeros(n_heads, n_blocks, kernel_size)
        w[:, :, -1] = 1.0                      # causal: take the current step only
        self.conv_w = nn.Parameter(w)

    def forward(self, delta):
        # delta: (B, T, H, nb) -> filter along T, causally
        B, T, H, nb = delta.shape
        x = delta.permute(0, 2, 3, 1).reshape(B, H * nb, T)          # (B, H*nb, T)
        x = nn.functional.pad(x, (self.kernel_size - 1, 0))          # causal left-pad
        w = self.conv_w.reshape(H * nb, 1, self.kernel_size)
        x = nn.functional.conv1d(x, w, groups=H * nb)                # depthwise
        delta = x.reshape(B, H, nb, T).permute(0, 3, 1, 2)           # back to (B,T,H,nb)
        return super().forward(delta)


class MapFormerWM_ConvDelta(MapFormerWM):
    """MapFormer-WM + SRoPE's conv1d filter on Delta before the cumulative sum."""

    def __init__(self, *args, conv_kernel: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        pi = self.path_integrator
        new = PathIntegratorConv(pi.n_heads, pi.n_blocks, kernel_size=conv_kernel)
        new.omega = pi.omega                    # keep the trained/initialised ladder
        self.path_integrator = new


class GatedActionToLie(nn.Module):
    """Wraps ActionToLieAlgebra: Delta_t <- sigmoid(W_g x_t) * Delta_t.

    Wrapping the Delta producer (rather than patching forward) means every call
    site inside MapFormerWM picks the gate up automatically.
    """

    def __init__(self, inner, d_model, n_heads, n_blocks):
        super().__init__()
        self.inner = inner
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.gate_proj = nn.Linear(d_model, n_heads * n_blocks)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 4.0)    # sigmoid(4) ~= 0.982

    def forward(self, x):
        delta = self.inner(x)                                   # (B, T, H, nb)
        B, T = x.shape[0], x.shape[1]
        g = torch.sigmoid(self.gate_proj(x)).reshape(B, T, self.n_heads, self.n_blocks)
        return delta * g


class MapFormerWM_GateDelta(MapFormerWM):
    """MapFormer-WM + SRoPE's sigmoid gate on the per-token displacement.

    On an interleaved [action, obs, ...] stream the gate can simply learn to zero
    the observation tokens, which MapFormer must otherwise do implicitly through
    the rank-2 bottleneck.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pi = self.path_integrator
        self.action_to_lie = GatedActionToLie(
            self.action_to_lie, self.d_model, pi.n_heads, pi.n_blocks)


class DeadGateActionToLie(GatedActionToLie):
    """CAPACITY CONTROL for GateDelta: identical parameters, no gating effect.

    Same nn.Linear(d_model, n_heads*n_blocks) is created in the same order (so it
    consumes the same init RNG draws and shifts every subsequent parameter
    identically), lives in the optimizer, and is evaluated in the forward pass --
    but its output does NOT gate Delta. Delta passes through unchanged.

    This is the control that decides whether GateDelta's advantage is the GATE or
    merely +32,896 parameters (+1.0%) plus an RNG shift. Same pattern that killed
    the Level-1.5 accuracy claim (Vanilla_ExtraHead tied it at t=0.79) and that
    v4's aux_coef=0.0 arm used.

    Note: the gate params receive ZERO gradient here (multiplied out), so they also
    contribute nothing to the grad-clip norm -- matching run_v4_control.sh's design.
    """

    def forward(self, x):
        delta = self.inner(x)
        B, T = x.shape[0], x.shape[1]
        g = torch.sigmoid(self.gate_proj(x)).reshape(B, T, self.n_heads, self.n_blocks)
        return delta + 0.0 * g          # params used, effect removed


class MapFormerWM_GateDeltaControl(MapFormerWM):
    """MapFormer-WM + GateDelta's parameters, with the gate disabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pi = self.path_integrator
        self.action_to_lie = DeadGateActionToLie(
            self.action_to_lie, self.d_model, pi.n_heads, pi.n_blocks)
