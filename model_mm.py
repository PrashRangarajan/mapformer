"""Memory Maze arms (MEMORY_MAZE_PREREG.md). Reuses mapformer.model's PathIntegrator and WMTransformerLayer unchanged.

Sequence [f_0, s_1, f_1, ..., s_1000, f_1000] (T = 2001). Every arm shares the same input embedding and retrieval head:
  frame token f_t: LayerNorm(256-d frozen embedding) -> Linear(d); on a blind step a learned blank vector instead;
  step token s_t:  Linear(content feature) where the content feature is the E feature (168-d: one-hot (a_{t-1}, a_t)
                   times [dir_{t-1}, dir_t]) for every arm, plus the odometry d_t (2-d) for the arms given position
                   (P_odo, GPS-PE);
  a token-type embedding (step / frame) is added;
  query q_t = Linear(LayerNorm(out at s_t)), key k_u = Linear(LayerNorm(embedding of f_u)), cosine / TAU.
Frames enter CONTENT ONLY (IMAGE_TORUS design rule): in the path arms the step map reads only the step feature of
step tokens; frame tokens have step 0.

Arms (ARMS):
  P_E    path: step = W_out W_in (E feature), rank 4 shared latent, PathIntegrator (omega geometric, Delta_max 64)
  P_odo  path: step = W_out W_in (d_t), content also gets d_t      (given-odometry ceiling)
  P_raw  path: step = W_out W_in (one-hot a_t)                      (optional secondary)
  R1/R4  index RoPE (canonical base 10000), 1 / 4 layers
  GPS    index RoPE + additive sinusoidal code of cumsum(d_t) (pos_t - pos_0), content also gets d_t
  GRU    one-layer GRU, hidden size chosen so its core matches one transformer layer's parameter count
  P_E_frozen  (Amendment 2, descriptive) P_E with the least-squares E map installed frozen (W_in, W_out, omega fixed);
              its trainable parameter count equals R1's exactly
Probe states (`probe_states`): the final hidden after the last LayerNorm for transformer arms; the GRU's own hidden
state (184-d) for the GRU arm (Amendment 2).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mapformer.model import PathIntegrator, WMTransformerLayer

EMB = 256
E_DIM = 168
TAU = 0.1
ARMS = ("P_E", "P_odo", "P_raw", "R1", "R4", "GPS", "GRU", "P_E_frozen")   # P_E_frozen: Amendment 2, descriptive
GPS_FREQS = 16


def arm_spec(arm):
    """(content feature keys, step feature key or None, kind, n_layers)."""
    return {
        "P_E": (("e",), "e", "path", 1),
        "P_E_frozen": (("e",), "e", "path", 1),   # P_E with the fitted E map installed FROZEN (train_mm does it)
        "P_odo": (("e", "odo"), "odo", "path", 1),
        "P_raw": (("e",), "raw", "path", 1),
        "R1": (("e",), None, "index", 1),
        "R4": (("e",), None, "index", 4),
        "GPS": (("e", "odo"), None, "gps", 1),
        "GRU": (("e",), None, "gru", 1),
    }[arm]


FEAT_DIM = {"e": E_DIM, "odo": 2, "raw": 6}


class IO(nn.Module):
    def __init__(self, d, content_dim, dc=128):
        super().__init__()
        self.frame_in = nn.Sequential(nn.LayerNorm(EMB), nn.Linear(EMB, d))
        self.step_in = nn.Linear(content_dim, d)
        self.blank = nn.Parameter(torch.randn(d) * 0.02)
        self.type_emb = nn.Embedding(2, d)
        self.q = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dc))
        self.k = nn.Sequential(nn.LayerNorm(EMB), nn.Linear(EMB, dc))

    def tokens(self, emb, sighted, content):
        """emb (B, 1001, 256) float, sighted (B, 1001) bool, content (B, 1001, C) -> x (B, 2001, d)."""
        B, T1, _ = emb.shape
        f = self.frame_in(emb)
        f = torch.where(sighted.unsqueeze(-1), f, self.blank.expand_as(f))
        s = self.step_in(content[:, 1:])
        x = torch.empty(B, 2 * T1 - 1, f.shape[-1], device=emb.device, dtype=f.dtype)
        x[:, 0::2] = f
        x[:, 1::2] = s
        typ = torch.zeros(2 * T1 - 1, dtype=torch.long, device=emb.device)
        typ[0::2] = 1
        return x + self.type_emb(typ).unsqueeze(0)

    def logits(self, out, emb):
        """out (B, 2001, d) final hidden; returns (B, 1000, 1001): row t-1 = query at s_t vs every frame u."""
        q = F.normalize(self.q(out[:, 1::2]), dim=-1)
        k = F.normalize(self.k(emb), dim=-1)
        return torch.matmul(q, k.transpose(1, 2)) / TAU


def index_cos_sin(L, n_heads, n_blocks, device, base=10000.0):
    k = torch.arange(n_blocks, dtype=torch.float32, device=device)
    inv = base ** (-k / n_blocks)
    ang = torch.outer(torch.arange(L, dtype=torch.float32, device=device), inv)
    return (torch.cos(ang)[None, None].expand(1, n_heads, L, n_blocks),
            torch.sin(ang)[None, None].expand(1, n_heads, L, n_blocks))


class MMTransformer(nn.Module):
    def __init__(self, arm, d=128, n_heads=2, rank=4, dropout=0.0, grid=64):
        super().__init__()
        self.arm = arm
        ckeys, skey, kind, n_layers = arm_spec(arm)
        self.ckeys, self.skey, self.kind = ckeys, skey, kind
        self.d, self.n_heads = d, n_heads
        self.n_blocks = d // n_heads // 2
        self.io = IO(d, sum(FEAT_DIM[k] for k in ckeys))
        if kind == "path":
            self.w_in = nn.Linear(FEAT_DIM[skey], rank, bias=False)
            self.w_out = nn.Linear(rank, n_heads * self.n_blocks, bias=False)
            self.path = PathIntegrator(n_heads, self.n_blocks, grid)
        if kind == "gps":
            lam = 64.0 ** (torch.arange(GPS_FREQS, dtype=torch.float32) / (GPS_FREQS - 1))   # 1 .. 64 cells
            self.register_buffer("gps_w", 2 * math.pi / lam)
            self.gps_in = nn.Linear(4 * GPS_FREQS, d)
        self.layers = nn.ModuleList([WMTransformerLayer(d, n_heads, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d)

    def step_delta(self, feats):
        """Per-token step (B, 2001, H, nb): step tokens get W_out W_in phi_t, frame tokens 0 (ActOnly)."""
        phi = feats[self.skey][:, 1:]                       # (B, 1000, Fdim): step features of s_1..s_1000
        B, T = phi.shape[:2]
        st = self.w_out(self.w_in(phi)).view(B, T, self.n_heads, self.n_blocks)
        delta = torch.zeros(B, 2 * T + 1, self.n_heads, self.n_blocks, device=phi.device, dtype=st.dtype)
        delta[:, 1::2] = st
        return delta

    def hidden(self, emb, sighted, feats):
        content = torch.cat([feats[k] for k in self.ckeys], dim=-1)
        x = self.io.tokens(emb, sighted, content)
        B, L, _ = x.shape
        if self.kind == "path":
            cos_a, sin_a = self.path(self.step_delta(feats))
        else:
            cos_a, sin_a = index_cos_sin(L, self.n_heads, self.n_blocks, x.device)
            cos_a, sin_a = cos_a.expand(B, -1, -1, -1), sin_a.expand(B, -1, -1, -1)
        if self.kind == "gps":
            p = torch.cumsum(feats["odo"], dim=1)              # (B, 1001, 2): pos_t - pos_0
            ptok = torch.repeat_interleave(p, 2, dim=1)[:, 1:]  # f_0:p_0, s_t:p_t, f_t:p_t
            a = ptok.unsqueeze(-1) * self.gps_w                 # (B, L, 2, K)
            code = torch.cat([torch.sin(a), torch.cos(a)], dim=-1).flatten(2)
            x = x + self.gps_in(code)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, mask)
        return self.out_norm(x)

    def forward(self, emb, sighted, feats):
        out = self.hidden(emb, sighted, feats)
        return self.io.logits(out, emb), out

    def probe_states(self, emb, sighted, feats):
        return self.hidden(emb, sighted, feats)

    @torch.no_grad()
    def install_e_map(self, W, frozen, seed=0):
        """Rule 14: install the least-squares E map (W: (168, 2), e_feature @ W = displacement) as the step map.
        Latent 0/1 = (dx, dy); W_out sends them to each (head, block) along a random unit direction u_k, so the block's
        angle is omega_k * u_k . (pos_t - pos_0): a position code with no aliasing over 15 cells (lowest omega 2 pi/64).
        frozen=True also freezes W_in, W_out and omega."""
        assert self.kind == "path" and self.skey == "e"
        g = torch.Generator().manual_seed(seed)
        self.w_in.weight.zero_()
        self.w_in.weight[0] = torch.as_tensor(W[:, 0], dtype=torch.float32)
        self.w_in.weight[1] = torch.as_tensor(W[:, 1], dtype=torch.float32)
        phi = torch.rand(self.w_out.weight.shape[0], generator=g) * 2 * math.pi
        self.w_out.weight.zero_()
        self.w_out.weight[:, 0] = torch.cos(phi)
        self.w_out.weight[:, 1] = torch.sin(phi)
        if frozen:
            for p in (self.w_in.weight, self.w_out.weight, self.path.omega):
                p.requires_grad_(False)


def transformer_layer_params(d=128, n_heads=2):
    return sum(p.numel() for p in WMTransformerLayer(d, n_heads, 0.0).parameters())


def gru_hidden_for(d=128, n_heads=2):
    """Largest h with GRU(d -> h) + Linear(h -> d) parameters <= one transformer layer's."""
    target = transformer_layer_params(d, n_heads)
    h = 1
    while 3 * (d * h + h * h + 2 * h) + (h * d + d) <= target:
        h += 1
    return h - 1


class MMGRU(nn.Module):
    def __init__(self, arm="GRU", d=128, **kw):
        super().__init__()
        self.arm = arm
        self.ckeys, self.skey, self.kind, _ = arm_spec(arm)
        self.io = IO(d, sum(FEAT_DIM[k] for k in self.ckeys))
        self.h = gru_hidden_for(d)
        self.gru = nn.GRU(d, self.h, batch_first=True)
        self.proj = nn.Linear(self.h, d)
        self.out_norm = nn.LayerNorm(d)

    def hidden(self, emb, sighted, feats):
        content = torch.cat([feats[k] for k in self.ckeys], dim=-1)
        x = self.io.tokens(emb, sighted, content)
        y, _ = self.gru(x)
        return self.out_norm(self.proj(y))

    def probe_states(self, emb, sighted, feats):
        """The GRU's hidden state (184-d), the representation Pasukonis et al. probe for recurrent models."""
        content = torch.cat([feats[k] for k in self.ckeys], dim=-1)
        y, _ = self.gru(self.io.tokens(emb, sighted, content))
        return y

    def forward(self, emb, sighted, feats):
        out = self.hidden(emb, sighted, feats)
        return self.io.logits(out, emb), out


def build(arm, **kw):
    assert arm in ARMS, arm
    return MMGRU(arm, **kw) if arm == "GRU" else MMTransformer(arm, **kw)


def n_params(m, trainable_only=False):
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)
