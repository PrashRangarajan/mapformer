"""Arms for TW_AMBIG_PREREG.md (direction words as actions and as observed content). New classes only; every base
weight equals the context-free arm's at the same seed (extra modules draw no random numbers; checked in
docs/audits/2026-10-08/tw_ambig_checks.py).

  MapWM        model_rank.MapFormerWM_r4, 1 layer: the context-free step (the text world's path arm).
  RoleTag      ORACLE, 1 layer: MapWM whose STEP reads a separate embedding row for a direction word used in a
               non-movement role (12 rows `nm_emb`, initialised as copies of the 12 direction words' rows, so at
               init RoleTag == MapWM exactly); attention / content see the plain word. A context-free step over
               (word, role) pairs: what MapWM could do if it were told the role. Reads tagged streams
               (environment_tw_ambig.TextWorldAmbig(tag_roles=True): non-movement direction word k -> id V + k).
  DirOnlyRole  REFERENCE, 1 layer: steps only on MOVEMENT-role direction words, every other token (including the same
               words in a non-movement role) steps 0 (TW_NORMSTEP's DirOnly, told the role; its cap there was the
               aside nouns, 0.970-0.974). Tagged streams.
  HSR          model_context_step.HiddenStepResWM, 2 layers: Delta_t = W_out W_in (emb(x_t) + alpha LN(h1_t)), alpha a
               learned scalar initialised at 0; h1 = an index-RoPE attention layer; layer 2 is path-integrated.
  CF2          HSR with alpha FIXED at 0 (a buffer): the same 2-layer architecture with a context-free step, the
               one-knob control for HSR's context correction (identical to HSR at init).
  RoPE1, RoPE2 model_baseline_rope.MapFormerWM_RoPE, 1 / 2 layers: index position, no path (the floor).
"""
import torch
import torch.nn as nn

from mapformer.model_baseline_rope import MapFormerWM_RoPE
from mapformer.model_codes import _StepOverride
from mapformer.model_context_step import HiddenStepResWM
from mapformer.model_rank import MapFormerWM_r4

N_DIR = 12


class _Tagged(_StepOverride):
    """MapWM r=4 that reads a tagged stream: ids >= vocab_size are non-movement direction words, mapped to their word
    for the content path. `step(tokens, x)` sees the raw (tagged) ids."""

    def __init__(self, vocab_size, *a, dir_ids=None, **kw):
        super().__init__(vocab_size, *a, **kw)
        assert dir_ids is not None and len(dir_ids) == N_DIR, dir_ids
        base = torch.arange(vocab_size + N_DIR); base[vocab_size:] = torch.as_tensor(dir_ids)
        self.register_buffer("base_id", base)
        self.n_base = vocab_size

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.token_emb(self.base_id[tokens])
        cos_a, sin_a = self.path_integrator(self.step(tokens, x))
        m = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, m)
        return self.out_proj(self.out_norm(x))


class RoleTag(_Tagged):
    def __init__(self, vocab_size, *a, dir_ids=None, **kw):
        super().__init__(vocab_size, *a, dir_ids=dir_ids, **kw)
        self.nm_emb = nn.Parameter(self.token_emb.weight.detach()[torch.as_tensor(dir_ids)].clone())

    def step(self, tokens, x):
        tag = tokens >= self.n_base
        xs = torch.where(tag[..., None], self.nm_emb[(tokens - self.n_base).clamp(min=0, max=N_DIR - 1)], x)
        return self.action_to_lie(xs)


class DirOnlyRole(_Tagged):
    def __init__(self, vocab_size, *a, dir_ids=None, **kw):
        super().__init__(vocab_size, *a, dir_ids=dir_ids, **kw)
        mask = torch.zeros(vocab_size + N_DIR); mask[torch.as_tensor(dir_ids)] = 1.0
        self.register_buffer("step_mask", mask)

    def step(self, tokens, x):
        return self.action_to_lie(x) * self.step_mask[tokens][..., None, None].to(x.dtype)


class CF2(HiddenStepResWM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        del self.ctx_alpha
        self.register_buffer("ctx_alpha", torch.zeros(1))


# arm -> (constructor, layers, reads a tagged stream)
ARMS = {"MapWM": (MapFormerWM_r4, 1, False), "RoleTag": (RoleTag, 1, True), "DirOnlyRole": (DirOnlyRole, 1, True),
        "HSR": (HiddenStepResWM, 2, False), "CF2": (CF2, 2, False), "RoPE1": (MapFormerWM_RoPE, 1, False),
        "RoPE2": (MapFormerWM_RoPE, 2, False)}
PATH_ARMS = ["MapWM", "RoleTag", "DirOnlyRole", "HSR", "CF2"]


def build(arm, env, size=64):
    cls, L, tagged = ARMS[arm]
    assert bool(getattr(env, "tag_roles", False)) == tagged, (arm, "env tag_roles must match the arm")
    kw = dict(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=L, grid_size=size)
    if tagged:
        kw["dir_ids"] = env.dir_word_ids
    m = cls(**kw)
    assert len(m.layers) == L, (arm, len(m.layers), L)                      # rule 17
    return m


def steps(m, tokens):
    """Per-token angle increments (B, L, H, n_blocks), omega not applied; None for index arms."""
    if isinstance(m, MapFormerWM_RoPE):
        return None
    if isinstance(m, _Tagged):
        return m.step(tokens, m.token_emb(m.base_id[tokens]))
    if isinstance(m, HiddenStepResWM):
        return m.step(tokens)[0]
    return m.action_to_lie(m.token_emb(tokens))
