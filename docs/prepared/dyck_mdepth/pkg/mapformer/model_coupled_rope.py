"""RoPE driven by supplied position IDs: the position-coupling oracle (Cho et al. 2024).

Identical to the index RoPE baseline except the rotation angle for token i is pos_ids[i] * inv_freq
instead of i * inv_freq. Cho et al. use learned absolute position embeddings over coupled IDs; this is
the rotary analogue, labelled as such. With pos_ids = arange it is exactly MapFormerWM_RoPE.
"""
import torch

from mapformer.model_baseline_rope import MapFormerWM_RoPE


class MapFormerWM_CoupledRoPE(MapFormerWM_RoPE):
    wants_pos_ids = True

    def forward(self, tokens, pos_ids=None):
        if pos_ids is None:
            return super().forward(tokens)
        B, L = tokens.shape
        x = self.token_emb(tokens)
        ang = pos_ids[:, :L].to(x.dtype).unsqueeze(-1) * self.inv_freq.to(x.dtype)   # (B, L, nb)
        cos_a = ang.cos().unsqueeze(1).expand(B, self.n_heads, L, -1).contiguous()
        sin_a = ang.sin().unsqueeze(1).expand(B, self.n_heads, L, -1).contiguous()
        mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, mask)
        return self.out_proj(self.out_norm(x))
