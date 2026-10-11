"""Per-head rank on the 1D ring (RANK_TESTS_PREREG.md, experiment 1).

Rank theory T1 (`docs/theory/2026-10-04/00_PLAN.md`): at per-head rank r = D, any step component shared by every move
(a clock c) is, in every channel, a false drift of the decoded position, so a rank-D head is clean only if c = 0
exactly; at r >= D + 1 a clock can sit off the map. On the ring D = 1. Arms (all built from MapWM's base at the seed,
so every non-bottleneck weight is identical across arms; the bottleneck is created afterwards, as in
`model_rank_perhead.py`):
  R1    per-head rank 1:  Delta_h = W_out^h W_in^h x, W_in^h: d -> 1
  R2    per-head rank 2:  as R1 with W_in^h: d -> 2 (= model_rank_perhead.MapFormerWM_PerHead)
  R1DF  per-head rank 1 with drift FORBIDDEN by construction: the step of the +1 action is W_out^h W_in^h a with
        a = (e(+1) - e(-1)) / sqrt(2) (Amendment 1), the -1 action's is its exact negative, and every other token's step is 0. So c = 0 and
        observation steps are 0 at every point of training: if R1 fails because training cannot find c = 0, R1DF
        solves; if R1DF also fails, drift is not the obstacle.
  R1AO  (Amendment 1) per-head rank 1 with observation and blank steps forced to 0 and both action steps FREE: removes
        the observation steps (the LEAK_RESULTS effect) but leaves the action-common component c free, so R1DF vs R1AO
        isolates c and R1AO vs R1 isolates the observation steps.
R1, R1DF and R1AO share the bottleneck's initial weights (same module, same draw). Amendment 1: R1DF's direction is
a = (e(+1) - e(-1)) / sqrt(2) (was / 2), so its initial step has the variance of a single embedding, as R1's does.
"""
import torch
import torch.nn as nn

from mapformer.model import MapFormerWM
from mapformer.model_rank_perhead import ActionToLieAlgebraPerHead


class _RingPerHead(MapFormerWM):
    R = 1

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, bottleneck_r=2, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size, 2)
        # replaced AFTER the base is built, so the base's RNG draws equal MapWM's (Vanilla) at this seed
        self.action_to_lie = ActionToLieAlgebraPerHead(d_model, n_heads, self.n_blocks, self.R)


class MapFormerWM_Ring_R1(_RingPerHead):
    R = 1


class MapFormerWM_Ring_R2(_RingPerHead):
    R = 2


class MapFormerWM_Ring_R1DF(_RingPerHead):
    """Per-head rank 1, drift forbidden. `action_ids` = (token id of +1, token id of -1)."""
    R = 1

    def __init__(self, vocab_size, *a, action_ids=(0, 1), **kw):
        super().__init__(vocab_size, *a, **kw)
        self.register_buffer("act", torch.tensor(action_ids, dtype=torch.long), persistent=False)

    def step_table(self):
        """(V, H, nb) per-token step before omega: +v for the +1 action, -v for the -1 action, 0 otherwise."""
        e = self.token_emb.weight
        a = (e[self.act[0]] - e[self.act[1]]) / (2 ** 0.5)
        v = self.action_to_lie(a.view(1, 1, -1))[0, 0]                      # (H, nb)
        sign = torch.zeros(e.shape[0], device=e.device, dtype=e.dtype)
        sign[self.act[0]], sign[self.act[1]] = 1.0, -1.0
        return sign[:, None, None] * v[None]

    def get_position_state(self, tokens: torch.Tensor):
        return self.path_integrator(self.step_table()[tokens])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        x = self.token_emb(tokens)
        delta = self.step_table()[tokens]                                    # (B, L, H, nb)
        cos_a, sin_a = self.path_integrator(delta)
        causal_mask = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, cos_a, sin_a, causal_mask)
        return self.out_proj(self.out_norm(x))


class MapFormerWM_Ring_R1AO(MapFormerWM_Ring_R1DF):
    """Per-head rank 1, observation steps forced to 0, action steps free (Amendment 1)."""

    def step_table(self):
        e = self.token_emb.weight
        v = self.action_to_lie(e[self.act].view(1, 2, -1))[0]               # (2, H, nb): +1 and -1 actions, free
        out = torch.zeros(e.shape[0], *v.shape[1:], device=e.device, dtype=v.dtype)
        out[self.act] = v
        return out
