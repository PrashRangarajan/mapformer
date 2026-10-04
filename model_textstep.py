"""Step variants for NormStep on the text world (TW_NORMSTEP_PREREG.md). Built on model_codes._StepOverride
(MapWM r=4 with an overridable step; forward identical to MapFormerWM's), so every base weight equals
Vanilla_r4's at the seed: the extra modules draw no random numbers.

  MapWM_NormStepNB  NormStep with a bias-free LayerNorm: Delta(e) = W (gamma * (e - mean) / std). Removes the
                    beta parameter's step (W beta, the same for every token), which on a stream with a variable
                    number of words per move would be a per-word clock unless cancelled. A shared step can still
                    be built from a shared component of gamma * norm(e) (audit 2026-10-03): this is a one-knob
                    ablation of beta, not a guarantee of no per-word clock.
  MapWM_DirOnly     steps from the 12 direction words only (ids set by `set_step_ids`): the reference with no
                    step on any other word, told which words move it (TEM-t's action-only update; ActOnly's
                    text-world analogue).
"""
import torch
import torch.nn as nn

from mapformer.model_codes import _StepOverride, MapWM_NormStep


class MapWM_NormStepNB(MapWM_NormStep):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.step_ln = nn.LayerNorm(self.d_model, bias=False)


class MapWM_DirOnly(_StepOverride):
    def __init__(self, vocab_size, *a, **kw):
        super().__init__(vocab_size, *a, **kw)
        self.register_buffer("step_mask", torch.zeros(vocab_size))

    def set_step_ids(self, ids):
        self.step_mask.zero_(); self.step_mask[list(ids)] = 1.0

    def step(self, tokens, x):
        return self.action_to_lie(x) * self.step_mask[tokens][..., None, None].to(x.dtype)


def step_of(m, tokens):
    """Per-token angle increments (B, L, H, n_blocks) for any arm, omega not applied."""
    x = m.token_emb(tokens)
    return m.step(tokens, x) if hasattr(m, "step") else m.action_to_lie(x)
