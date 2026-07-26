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
    # -- local-frame-reset (H3 ingredient 3); off by default (backward compat) --
    frame_reset = False
    wants_seg_id = False

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

    def _resolve_seg_id(self, tokens, L, B):
        """Per-token room-visit id (B, L). Uses the oracle signal stashed as
        self._batch_seg_id; falls back to fixed-stride (k) grouping so smoke /
        causality checks still run without an oracle signal."""
        seg_id = getattr(self, "_batch_seg_id", None)
        if seg_id is None or seg_id.shape[0] != B or seg_id.shape[1] < L:
            seg_id = (torch.arange(L, device=tokens.device) // self.k).unsqueeze(0).expand(B, L)
        return seg_id[:, :L].to(device=tokens.device, dtype=torch.long)

    def _reset_cum_delta(self, cum_delta, seg_id):
        """Local-frame reset: subtract, per token, the cumulative path angle at
        the ENTRY token of its room-visit, so position is measured RELATIVE to
        room entry. Two visits to the same motif-cell then get the SAME local
        angle -> identical fine codes -> identical motifs collapse. Causal: the
        subtracted baseline sits at an index <= t. cum_delta is (B, L, H, nb)."""
        B, L, H, nb = cum_delta.shape
        S_max = int(seg_id.max().item()) + 1
        t_idx = torch.arange(L, device=cum_delta.device).unsqueeze(0).expand(B, L)
        entry = torch.full((B, S_max), L, device=cum_delta.device, dtype=torch.long)
        entry.scatter_reduce_(1, seg_id, t_idx, reduce="amin", include_self=True)
        entry_per_tok = torch.gather(entry, 1, seg_id)                 # (B, L)
        idx = entry_per_tok.view(B, L, 1, 1).expand(B, L, H, nb)
        return cum_delta - torch.gather(cum_delta, 1, idx)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        k = self.k

        x = self.token_emb(tokens)                       # (B, L, D)
        delta = self.action_to_lie(x)                    # (B, L, H, nb)
        cum_delta = torch.cumsum(delta, dim=1)           # cumulative path angle
        if self.frame_reset:                             # room-relative position angles
            cum_delta = self._reset_cum_delta(cum_delta, self._resolve_seg_id(tokens, L, B))

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


# --------------------------------------------------------------------------
# Phase 2 (H3): oracle motif-segmented Hourglass
# --------------------------------------------------------------------------
class MapFormerWM_Hourglass_MotifSeg(MapFormerWM_Hourglass):
    """Room-boundary-segmented Hourglass (ORACLE segmentation).

    Identical to Hourglass_k2 in every component EXCEPT the shortening: instead
    of pooling on a FIXED token stride, it pools on ORACLE room boundaries --
    one coarse token per room-visit. This isolates H3's first ingredient
    (segmentation ALIGNED to motif structure) from generic fixed-stride pooling;
    MotifSeg vs Hourglass_k2 differ ONLY in the segment boundaries.

    Segmentation signal: a per-token segment id (B, L), stashed as
    ``self._batch_seg_id`` by the trainer/evaluator (derived from the env's
    ``meta['new_room']``). Falls back to fixed-stride (k) grouping when no
    signal is supplied, so smoke / causality checks still run.

    The coarse path is CAUSAL and content-summarising:
      - coarse token s = MEAN of the (unrotated) pre-layer hidden states over
        the tokens of room-visit s (content summary; motif-carrying);
      - coarse angle s = MEAN of the cumulative path angle over room-visit s;
      - coarse MapFormer layers attend causally over the room sequence;
      - a fine token in room j receives the coarse output of room j-1 (its most
        recent COMPLETED room), zero for the first room -> no within-room future
        leak. The coarse path is thus a memory of PAST rooms' motifs, exactly
        the sufficient statistic the cross-instance target rewards.

    NOTE (v1): the local-coordinate-frame reset (H3 ingredient 3) is NOT here;
    this build tests segmentation alignment alone.
    """
    wants_seg_id = True
    shorten_factor = 2                 # only used by the no-oracle fallback
    n_pre, n_coarse, n_post = 1, 1, 1

    def _segment_pool(self, x, seg_id, S_max):
        """Mean-pool x (B, L, D) into (B, S_max, D) by segment id (B, L)."""
        B, L, D = x.shape
        out = x.new_zeros(B, S_max, D)
        out.scatter_add_(1, seg_id.unsqueeze(-1).expand(B, L, D), x)
        cnt = x.new_zeros(B, S_max, 1)
        cnt.scatter_add_(1, seg_id.unsqueeze(-1).expand(B, L, 1), x.new_ones(B, L, 1))
        return out / cnt.clamp(min=1.0)

    def forward(self, tokens):
        B, L = tokens.shape
        seg_id = self._resolve_seg_id(tokens, L, B)            # oracle room segmentation
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)                 # (B, L, H, nb)
        if self.frame_reset:                                  # room-relative angles (v2)
            cum_delta = self._reset_cum_delta(cum_delta, seg_id)
        cos_f, sin_f = self._angles(cum_delta)
        mask_f = self._causal_mask(L, tokens.device)

        for layer in self.pre_layers:
            x = layer(x, cos_f, sin_f, mask_f)
        skip = x

        S_max = int(seg_id.max().item()) + 1

        xc = self._segment_pool(skip, seg_id, S_max)           # (B, S_max, D)
        cd = cum_delta.reshape(B, L, self.n_heads * self.n_blocks)
        cdc = self._segment_pool(cd, seg_id, S_max).view(
            B, S_max, self.n_heads, self.n_blocks)
        cos_c, sin_c = self._coarse_angles(cdc)
        mask_c = self._causal_mask(S_max, tokens.device)
        for layer in self.coarse_layers:
            xc = layer(xc, cos_c, sin_c, mask_c)

        # -- causal upsample: token in room j gets coarse output of room j-1 --
        prev = (seg_id - 1).clamp(min=0)                       # (B, L)
        up = torch.gather(xc, 1, prev.unsqueeze(-1).expand(B, L, self.d_model))
        up = up * (seg_id > 0).unsqueeze(-1).to(up.dtype)      # zero for first room
        x = skip + up if self._use_coarse else skip

        for layer in self.post_layers:
            x = layer(x, cos_f, sin_f, mask_f)
        x = self.out_norm(x)
        return self.out_proj(x)


# --------------------------------------------------------------------------
# Phase 2 v2: local-frame-reset variants (H3 ingredient 3)
# --------------------------------------------------------------------------
class MapFormerWM_Hourglass_MotifSeg_FR(MapFormerWM_Hourglass_MotifSeg):
    """v2 = MotifSeg + LOCAL-FRAME-RESET (the full H3). Oracle room segmentation
    AND room-relative position angles: identical motifs at different locations
    now produce identical fine codes, so they collapse to the same coarse token
    -- the motif-level sufficient statistic v1 (segmentation alone) never formed.
    Same params as Hourglass_k2 / MotifSeg (600,917); the ONLY change vs
    MotifSeg is that position is measured relative to room entry."""
    frame_reset = True


class MapFormerWM_FrameResetFlat(MapFormerWM_Hourglass):
    """Isolating control: FLAT MapFormer-WM (3 layers, shorten=1, matched to
    MapWM-FlatHG) with the frame-reset ONLY -- no coarse hierarchy. Tells us
    whether the reset alone drives any gain, or whether it needs the segmented
    coarse stack. (Uses oracle room boundaries only to place the reset origin.)"""
    shorten_factor = 1
    n_pre, n_coarse, n_post = 1, 1, 1
    frame_reset = True
    wants_seg_id = True
