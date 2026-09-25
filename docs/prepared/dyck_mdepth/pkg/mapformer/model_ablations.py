"""
Level 1.5 ablation variants. Each isolates one component of the full
Level 1.5 InEKF to test whether it's doing load-bearing work.

Ablations implemented:

  L15_ConstR  — constant R_t (matches Level 1's constant K* mathematically,
                but with learnable Π instead of DARE-derived Π). Isolates
                whether heteroscedastic R_t is necessary.
  L15_NoMeas  — measurement head z_t = 0 always. Innovation is then just
                the (wrapped) negation of θ_path, so corrections are
                driven only by the path-integration trajectory. Tests
                whether the measurement signal matters at all.
  L15_NoCorr  — scan output forced to 0 (no state correction). Recovers
                vanilla MapFormer but with extra parameters idling. Used
                to confirm the correction mechanism is load-bearing.
  L15_DARE    — Π computed via closed-form scalar DARE (like Level 1) but
                R_t per-token. Isolates whether the learnable-Π degree of
                freedom matters once you have heteroscedastic R_t.
"""

import math
import torch
import torch.nn as nn

from .model import MapFormerWM, WMTransformerLayer
from .model_inekf_level15 import assoc_scan_affine_scalar


class InEKFLevel15_ConstR(nn.Module):
    """Constant R (learned scalar). K_t = Π / (Π + R) is also constant ≈ Level 1
    with learnable Π. Tests whether heteroscedastic R_t matters."""

    def __init__(self, d_model, n_heads, n_blocks):
        super().__init__()
        self.n_heads = n_heads; self.n_blocks = n_blocks
        self.log_Pi = nn.Parameter(torch.full((n_heads, n_blocks), 0.0))
        self.log_R = nn.Parameter(torch.full((n_heads, n_blocks), 0.0))  # constant
        self.measure_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, n_heads * n_blocks),
        )

    def forward(self, theta_path, content_emb):
        B, L, H, NB = theta_path.shape
        z = math.pi * torch.tanh(self.measure_head(content_emb).view(B, L, H, NB))
        diff = z - theta_path
        nu = torch.atan2(torch.sin(diff), torch.cos(diff))
        Pi = self.log_Pi.exp().unsqueeze(0).unsqueeze(0)
        R = self.log_R.exp().unsqueeze(0).unsqueeze(0)  # constant across tokens
        K = Pi / (Pi + R).clamp_min(1e-8)  # (1, 1, H, NB) broadcasts
        K = K.expand(B, L, H, NB)
        alpha = 1.0 - K; u = K * nu
        d = assoc_scan_affine_scalar(alpha, u)
        return theta_path + d, Pi.expand(B, L, H, NB), K, R.expand(B, L, H, NB)


class InEKFLevel15_NoMeas(nn.Module):
    """z_t = 0 always. Innovation is just wrapped(-θ_path). Tests whether the
    measurement signal matters at all."""

    def __init__(self, d_model, n_heads, n_blocks):
        super().__init__()
        self.n_heads = n_heads; self.n_blocks = n_blocks
        self.log_Pi = nn.Parameter(torch.full((n_heads, n_blocks), 0.0))
        self.log_R_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, n_heads * n_blocks),
        )
        with torch.no_grad():
            self.log_R_head[-1].weight.mul_(0.01); self.log_R_head[-1].bias.fill_(0.0)

    def forward(self, theta_path, content_emb):
        B, L, H, NB = theta_path.shape
        log_R = self.log_R_head(content_emb).view(B, L, H, NB).clamp(-5, 5)
        R = log_R.exp()
        # z = 0 -> diff = -theta_path; wrap
        nu = torch.atan2(torch.sin(-theta_path), torch.cos(-theta_path))
        Pi = self.log_Pi.exp().unsqueeze(0).unsqueeze(0)
        K = Pi / (Pi + R).clamp_min(1e-8)
        alpha = 1.0 - K; u = K * nu
        d = assoc_scan_affine_scalar(alpha, u)
        return theta_path + d, Pi.expand_as(R), K, R


class InEKFLevel15_NoCorr(nn.Module):
    """Scan output forced to 0. Recovers vanilla MapFormer (θ_path passed
    through) but with extra idling parameters. Sanity check that correction
    is load-bearing."""

    def __init__(self, d_model, n_heads, n_blocks):
        super().__init__()
        self.n_heads = n_heads; self.n_blocks = n_blocks
        self.log_Pi = nn.Parameter(torch.full((n_heads, n_blocks), 0.0))
        self.log_R_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, n_heads * n_blocks),
        )
        self.measure_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, n_heads * n_blocks),
        )

    def forward(self, theta_path, content_emb):
        B, L, H, NB = theta_path.shape
        zero = torch.zeros_like(theta_path)
        Pi = self.log_Pi.exp().unsqueeze(0).unsqueeze(0).expand(B, L, H, NB)
        K = zero
        R = torch.zeros_like(theta_path)
        return theta_path, Pi, K, R  # correction = 0, return raw path


class InEKFLevel15_DARE(nn.Module):
    """Π derived from scalar DARE (not learnable), R_t per-token from head.
    Tests whether the free learnable Π gives any extra expressivity beyond
    the DARE-prescribed value."""

    def __init__(self, d_model, n_heads, n_blocks):
        super().__init__()
        self.n_heads = n_heads; self.n_blocks = n_blocks
        # Q is learnable; R effective = mean of R_t (we approximate with a
        # separate learnable R_avg used in DARE)
        self.log_Q = nn.Parameter(torch.full((n_heads, n_blocks), -3.0))
        self.log_R_avg = nn.Parameter(torch.full((n_heads, n_blocks), 1.0))
        self.log_R_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, n_heads * n_blocks),
        )
        self.measure_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(),
            nn.Linear(128, n_heads * n_blocks),
        )
        with torch.no_grad():
            self.log_R_head[-1].weight.mul_(0.01); self.log_R_head[-1].bias.fill_(0.0)

    def _compute_Pi_DARE(self):
        Q = self.log_Q.exp(); R = self.log_R_avg.exp()
        # Π_post = (-Q + sqrt(Q^2 + 4 Q R)) / 2
        return 0.5 * (-Q + torch.sqrt(Q * Q + 4.0 * Q * R))

    def forward(self, theta_path, content_emb):
        B, L, H, NB = theta_path.shape
        log_R = self.log_R_head(content_emb).view(B, L, H, NB).clamp(-5, 5)
        R = log_R.exp()
        z = math.pi * torch.tanh(self.measure_head(content_emb).view(B, L, H, NB))
        diff = z - theta_path
        nu = torch.atan2(torch.sin(diff), torch.cos(diff))
        Pi_dare = self._compute_Pi_DARE().unsqueeze(0).unsqueeze(0)  # (1,1,H,NB)
        Pi = Pi_dare.expand(B, L, H, NB)
        K = Pi / (Pi + R).clamp_min(1e-8)
        alpha = 1.0 - K; u = K * nu
        d = assoc_scan_affine_scalar(alpha, u)
        return theta_path + d, Pi, K, R


class _MapFormerWithInEKF(MapFormerWM):
    """Generic MapFormer-WM wrapper around a pluggable InEKF module."""

    INEKF_CLASS = None  # override in subclass

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                 dropout=0.1, grid_size=64, bottleneck_r=2):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout,
                         grid_size, bottleneck_r)
        self.layers = nn.ModuleList([
            WMTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.inekf = self.INEKF_CLASS(d_model, n_heads, self.n_blocks)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.action_to_lie(x)
        cum_delta = torch.cumsum(delta, dim=1)
        theta_path = cum_delta * self.path_integrator.omega.unsqueeze(0).unsqueeze(0)

        theta_hat, Pi, K, R = self.inekf(theta_path, x)
        self.last_theta_path = theta_path.detach()
        self.last_theta_hat = theta_hat.detach()
        self.last_Pi = Pi.detach(); self.last_K = K.detach(); self.last_R = R.detach()

        theta_for_rope = theta_hat.transpose(1, 2)
        cos_a = torch.cos(theta_for_rope); sin_a = torch.sin(theta_for_rope)
        causal_mask = torch.triu(
            torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)
        x = self.out_norm(x)
        return self.out_proj(x)


class MapFormerWM_L15_ConstR(_MapFormerWithInEKF):
    INEKF_CLASS = InEKFLevel15_ConstR


class MapFormerWM_L15_NoMeas(_MapFormerWithInEKF):
    INEKF_CLASS = InEKFLevel15_NoMeas


class MapFormerWM_L15_NoCorr(_MapFormerWithInEKF):
    INEKF_CLASS = InEKFLevel15_NoCorr


class MapFormerWM_L15_DARE(_MapFormerWithInEKF):
    INEKF_CLASS = InEKFLevel15_DARE


ABLATIONS = {
    "L15_ConstR":  MapFormerWM_L15_ConstR,
    "L15_NoMeas":  MapFormerWM_L15_NoMeas,
    "L15_NoCorr":  MapFormerWM_L15_NoCorr,
    "L15_DARE":    MapFormerWM_L15_DARE,
}
