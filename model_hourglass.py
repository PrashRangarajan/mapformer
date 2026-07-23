"""
Hourglass-MapFormer: the Hourglass Transformer (Nawrot et al., 2021,
"Hierarchical Transformers Are More Efficient Language Models") used as a
hierarchical SCAFFOLD, with its vanilla transformer layers replaced by
MapFormer-WM layers.

Motivation for this line of work
--------------------------------
All earlier hierarchy attempts in this repo were HOMEGROWN pooling schemes
(HierAttn, SpaceTimeHier, Recursive) and all were clean negatives. This file
tests a different methodology: take an established, published hierarchical
transformer wholesale and only swap the layer primitive. If an
externally-validated hierarchy design also fails, the negative is much
stronger; if it wins, we learned the homegrown pooling was the problem.

Two lessons from the prior negatives are baked in here:

  1. NEVER pool ROTATED keys/values. model_hier_attn.py pooled K/V AFTER
     applying RoPE (mean over rotated vectors), which sums different phases
     and cancels 56-69% of the position-code magnitude. Here we pool the
     UNROTATED hidden state; the MapFormer rotation is applied INSIDE the
     coarse layer, on the coarse token's own pooled position angle. So the
     coarse level is a genuine cognitive map over regions, not a bag of
     phase-cancelled summaries.

  2. The coarse level went INERT before because it was an ADDITIVE SIDE
     BRANCH (coarse_proj -> 0 under training, zero cost). Hourglass instead
     places the shortened stack IN THE MAIN RESIDUAL STREAM (U-Net): the
     post-shortening layers only receive information about distant context
     through the upsampled coarse path plus the U-Net skip. The skip is a
     pass-through the model CAN still learn to lean on, so we measure the
     coarse contribution explicitly (see coarse_contribution()).

Architecture (single-level, the canonical Hourglass shape)
----------------------------------------------------------
    tokens -> embed -> [n_pre  full-res MapFormer layers]  --.
                          |                                   |  U-Net skip
                     shorten(k)  (causal shift-then-pool)     |
                          |                                   |
                  [n_coarse coarse MapFormer layers]          |
                          |                                   |
                     upsample(k)  (repeat-interleave)         |
                          +  <---------------------------------'
                          |
                   [n_post full-res MapFormer layers] -> norm -> vocab

Causal shortening (Nawrot et al. 2021, section 2.3)
---------------------------------------------------
Shift right by (k-1) then average consecutive groups of k. After the shift,
group j spans ORIGINAL indices [jk-(k-1) .. jk], so coarse token j summarises
tokens no later than index jk. On upsample (repeat-interleave by k) coarse
token j feeds fine outputs [jk .. jk+k-1], every one of which has index >= jk.
Hence no fine position ever receives information from a strictly-future token.
This is verified numerically in test_hourglass_causal.py (perturb token t;
all logits at positions < t must be unchanged).

The MapFormer position angles are shortened with the SAME operator applied to
the cumulative path-integration angle (cumsum of the Lie-algebra deltas), so
the coarse token's rotation angle is a function of exactly the tokens it
pooled. omega is shared with the fine level (the omega spectrum is already
geometric fine->coarse; the low-frequency blocks are the ones that resolve
region scale, which is the Stensola grid-module reading of the spectrum).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ActionToLieAlgebra, PathIntegrator, WMTransformerLayer


def _causal_shorten(x: torch.Tensor, k: int) -> torch.Tensor:
    """Causal shift-then-pool downsample by factor k.

    Args:
        x: (B, L, D) with L a multiple of k.
    Returns:
        (B, L // k, D), mean-pooled over causal groups of k.

    Shift right by (k-1) (prepend zeros, drop tail), then average each
    contiguous block of k. Guarantees coarse token j depends only on original
    tokens with index <= j*k.
    """
    B, L, D = x.shape
    assert L % k == 0, f"length {L} not divisible by shorten factor {k}"
    if k == 1:
        return x
    shifted = F.pad(x, (0, 0, k - 1, 0))[:, :L]        # right shift by k-1
    return shifted.view(B, L // k, k, D).mean(dim=2)


def _upsample(c: torch.Tensor, k: int) -> torch.Tensor:
    """Repeat-interleave upsample by factor k. Inverse group-map of the
    causal shorten above: coarse token j -> fine outputs [jk .. jk+k-1]."""
    if k == 1:
        return c
    return c.repeat_interleave(k, dim=1)


class MapFormerWM_Hourglass(nn.Module):
    """Hourglass Transformer with MapFormer-WM layers.

    Standard constructor signature (vocab_size, d_model, n_heads, n_layers,
    grid_size, dropout, bottleneck_r) so it drops into train_variant.py. The
    hierarchy shape is set by class attributes (subclass to change), which the
    orchestrator does not need to pass:

        shorten_factor : k, the single-level downsample rate
        n_pre / n_coarse / n_post : layer counts at each stage

    ``n_layers`` from the CLI is ignored (the hierarchy fixes its own depth);
    this keeps the shared launcher working without special-casing.
    """

    # -- hierarchy shape (override in subclasses) --
    shorten_factor = 2
    n_pre = 1
    n_coarse = 1
    n_post = 1

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 1,          # ignored; hierarchy sets its own depth
        dropout: float = 0.1,
        grid_size: int = 64,
        bottleneck_r: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_blocks = self.d_head // 2
        self.vocab_size = vocab_size
        self.k = self.shorten_factor

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.action_to_lie = ActionToLieAlgebra(d_model, n_heads, self.n_blocks, bottleneck_r)
        self.path_integrator = PathIntegrator(n_heads, self.n_blocks, grid_size)

        self.pre_layers = nn.ModuleList(
            [WMTransformerLayer(d_model, n_heads, dropout) for _ in range(self.n_pre)]
        )
        self.coarse_layers = nn.ModuleList(
            [WMTransformerLayer(d_model, n_heads, dropout) for _ in range(self.n_coarse)]
        )
        self.post_layers = nn.ModuleList(
            [WMTransformerLayer(d_model, n_heads, dropout) for _ in range(self.n_post)]
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # Diagnostic switch: when False, the upsampled coarse path is dropped
        # (only the U-Net skip survives). Used by coarse_contribution() to
        # measure whether the shortened stack does any work.
        self._use_coarse = True

    # -- position machinery ------------------------------------------------
    def _angles(self, cum_delta: torch.Tensor):
        """cum_delta (B, L, H, nb) -> cos/sin (B, H, L, nb) at the fine level."""
        angles = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        angles = angles.transpose(1, 2)
        return torch.cos(angles), torch.sin(angles)

    def _coarse_angles(self, cum_delta_coarse: torch.Tensor):
        return self._angles(cum_delta_coarse)

    def _causal_mask(self, n: int, device):
        return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        k = self.k

        x = self.token_emb(tokens)                       # (B, L, D)
        delta = self.action_to_lie(x)                    # (B, L, H, nb)
        cum_delta = torch.cumsum(delta, dim=1)           # cumulative path angle

        # Pad sequence length up to a multiple of k (pad at the END; loss never
        # touches padded positions and causality of earlier tokens is intact).
        pad = (-L) % k
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            # extend cum_delta by holding the last cumulative angle (padded
            # tokens carry zero delta, i.e. no further movement)
            cum_delta = F.pad(cum_delta, (0, 0, 0, 0, 0, pad))
            cum_delta[:, L:] = cum_delta[:, L - 1:L]
        Lp = L + pad

        cos_f, sin_f = self._angles(cum_delta)           # fine angles
        mask_f = self._causal_mask(Lp, tokens.device)

        # -- pre: full-resolution MapFormer layers --
        for layer in self.pre_layers:
            x = layer(x, cos_f, sin_f, mask_f)
        skip = x                                         # U-Net residual source

        # -- shorten (content: unrotated hidden; position: cumulative angle) --
        xc = _causal_shorten(x, k)                       # (B, Lp/k, D)
        cum_delta_c = _causal_shorten(cum_delta.reshape(B, Lp, -1), k)
        cum_delta_c = cum_delta_c.view(B, Lp // k, self.n_heads, self.n_blocks)
        cos_c, sin_c = self._coarse_angles(cum_delta_c)
        mask_c = self._causal_mask(Lp // k, tokens.device)

        for layer in self.coarse_layers:
            xc = layer(xc, cos_c, sin_c, mask_c)

        # -- upsample + U-Net skip --
        up = _upsample(xc, k)                            # (B, Lp, D)
        if self._use_coarse:
            x = skip + up
        else:
            x = skip                                     # diagnostic: coarse off

        # -- post: full-resolution MapFormer layers --
        for layer in self.post_layers:
            x = layer(x, cos_f, sin_f, mask_f)

        x = x[:, :L]                                     # drop padding
        x = self.out_norm(x)
        return self.out_proj(x)

    @torch.no_grad()
    def coarse_contribution(self, tokens: torch.Tensor):
        """Relative L2 norm of the upsampled coarse path vs the skip stream,
        averaged over positions. A near-zero value means the shortened stack
        is inert (the failure mode observed in SpaceTimeHier / the Kalman
        cascade). Returns a python float."""
        B, L = tokens.shape
        k = self.k
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        pad = (-L) % k
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            cum_delta = F.pad(cum_delta, (0, 0, 0, 0, 0, pad))
            cum_delta[:, L:] = cum_delta[:, L - 1:L]
        Lp = L + pad
        cos_f, sin_f = self._angles(cum_delta)
        mask_f = self._causal_mask(Lp, tokens.device)
        for layer in self.pre_layers:
            x = layer(x, cos_f, sin_f, mask_f)
        skip = x
        xc = _causal_shorten(x, k)
        cum_delta_c = _causal_shorten(cum_delta.reshape(B, Lp, -1), k).view(
            B, Lp // k, self.n_heads, self.n_blocks)
        cos_c, sin_c = self._coarse_angles(cum_delta_c)
        mask_c = self._causal_mask(Lp // k, tokens.device)
        for layer in self.coarse_layers:
            xc = layer(xc, cos_c, sin_c, mask_c)
        up = _upsample(xc, k)
        return (up.norm(dim=-1).mean() / (skip.norm(dim=-1).mean() + 1e-8)).item()


# --------------------------------------------------------------------------
# Registered configurations
# --------------------------------------------------------------------------
class MapFormerWM_Hourglass_k2(MapFormerWM_Hourglass):
    """1 pre / 1 coarse / 1 post, shorten factor 2 (3 MapFormer layers)."""
    shorten_factor = 2
    n_pre, n_coarse, n_post = 1, 1, 1


class MapFormerWM_Hourglass_k4(MapFormerWM_Hourglass):
    """1 pre / 1 coarse / 1 post, shorten factor 4."""
    shorten_factor = 4
    n_pre, n_coarse, n_post = 1, 1, 1


class MapFormerWM_Hourglass_k2_deep(MapFormerWM_Hourglass):
    """1 pre / 2 coarse / 1 post, shorten factor 2 (4 layers)."""
    shorten_factor = 2
    n_pre, n_coarse, n_post = 1, 2, 1


class MapFormerWM_HourglassFlat3(MapFormerWM_Hourglass):
    """Matched-compute FLAT control: 3 full-resolution MapFormer layers, no
    shortening (shorten_factor=1). Identical parameter count and layer count
    to Hourglass_k2; the ONLY difference is that the middle layer runs at full
    resolution instead of coarse. This is the honest control for 'does putting
    the middle layer at coarse resolution help?'."""
    shorten_factor = 1
    n_pre, n_coarse, n_post = 1, 1, 1
