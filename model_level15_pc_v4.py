"""Level15PC v4: full PC isolation (Fix 5 + 6 + 8).

The progressive fix sequence:
    v1 (Level15PC, original):       no fixes — R saturates at lower clamp
    v2 (Level15PC_NoBypass):        Fix 5 (stop-grad on d_t) + Fix 6 (mask
                                    aux at landmarks). Closes the *direct*
                                    saturation route, but PC still leaks
                                    into ``action_to_lie`` via the
                                    ``theta_path`` gradient. Result: |θ̂|
                                    blows up to ~3800 at T=512 because
                                    PC drives action_to_lie to amplify
                                    Δ values.
    v3 (Level15PC_v3):              v2 + Fix 7 (tighter R clamp [-1, 5]).
                                    Recovers R distribution but not |θ̂|
                                    magnitude. Partial improvement only.
    v4 (THIS FILE, Level15PC_v4):   v2 + Fix 8 (also detach the
                                    theta_path input to the PC forward
                                    model). PC's aux loss can ONLY train
                                    the forward_model itself. action_to_lie,
                                    InEKF, embeddings — all isolated.

In v4, the PC module is essentially a *passive observer* of the model's
position estimates. The forward model trains to predict observations
from θ̂ as a representation-quality probe, but its gradient cannot
shape the model. If v4 matches Level15 alone, this confirms that ANY
non-trivial coupling between PC's aux loss and the rest of the model
degrades length generalisation — the negative-result for combinable
PC + Kalman is mechanistically airtight.

If v4 does NOT match Level15, something else (e.g., the architecture
itself, even with no gradient flow, induces different optimisation
behaviour) is going on. That would be a smaller mystery to chase.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_level15_pc_v2 import MapFormerWM_Level15PC_NoBypass


class MapFormerWM_Level15PC_v4(MapFormerWM_Level15PC_NoBypass):
    """v4: NoBypass + full detach of theta_path inside PC aux loss.

    PC's aux loss gradient flows ONLY into the forward_model. Path
    integration parameters (``action_to_lie``, ``omega``), the InEKF
    correction parameters (``log_R_head``, ``measure_head``,
    ``log_Pi``), and the embedding / output parameters all develop
    purely from the cross-entropy loss — exactly as in vanilla Level 1.5.

    The forward model still trains, but its predictions don't shape
    anything else. It's a representation-quality probe, not a
    representation-shaper.
    """

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)                         # (B, L, d_model)

        # Standard MapFormer-WM path integration (full grad through CE)
        delta = self.action_to_lie(x)                      # (B, L, H, NB)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = (
            cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)
        )                                                   # (B, L, H, NB)

        # Level 1.5 InEKF correction (full grad through CE)
        theta_hat, Pi, K, R = self.inekf(theta_path, x)    # (B, L, H, NB)

        # ===== Fix 8: detach the ENTIRE theta_hat for PC =====
        # PC's aux loss can only train forward_model itself; no gradient
        # leaks into action_to_lie / omega / InEKF / embeddings.
        theta_for_pc = theta_hat.detach()
        # =====================================================

        pos_feat = torch.cat(
            [torch.cos(theta_for_pc), torch.sin(theta_for_pc)], dim=-1
        )                                                  # (B, L, H, 2*NB)
        pos_feat = pos_feat.reshape(B, L, -1)              # (B, L, H*2*NB)
        obs_pred = self.forward_model(pos_feat)            # (B, L, d_model)

        # Aux loss masking: obs positions only AND non-landmark only
        # (Fix 6 retained from NoBypass). Note: x here has full grad,
        # but it's multiplied with mask_f and the squared error of a
        # detached-input forward model. The grad flows only into
        # forward_model parameters — not into x via the err computation,
        # since the forward model receives only detached input.
        # However, ‖x - obs_pred‖² does have a grad path through x, so
        # we additionally detach x to ensure embeddings aren't shaped:
        x_detached = x.detach()
        err_raw = x_detached - obs_pred                    # (B, L, d_model)
        obs_mask = torch.zeros(B, L, device=tokens.device, dtype=torch.bool)
        obs_mask[:, 1::2] = True
        not_landmark = tokens < self.LANDMARK_START_ID     # (B, L)
        mask = obs_mask & not_landmark
        mask_f = mask.unsqueeze(-1).to(err_raw.dtype)
        err = err_raw * mask_f
        denom = max(1.0, mask_f.sum().item() * err.shape[-1])
        self.prediction_error_squared = err.pow(2).sum() / denom

        # Diagnostics
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach()
        self.last_K = K.detach()
        self.last_R = R.detach()
        self.last_pred_err = err_raw.detach()

        # RoPE attention with corrected θ̂ (full grad through CE — unchanged)
        theta_for_rope = theta_hat.transpose(1, 2)         # (B, H, L, NB)
        cos_a = torch.cos(theta_for_rope)
        sin_a = torch.sin(theta_for_rope)

        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)

        x = self.out_norm(x)
        return self.out_proj(x)
