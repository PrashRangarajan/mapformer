"""The gain-phase map on the new-object task (GAIN_PHASE_PREREG.md): a 2x2 of STEP x SCORE.

    STEP   raw       Delta = W_out W_in e              (MapWM's; e = token embedding, objects e = A c_o)
           NormStep  Delta = W_out W_in LN(e)          (model_codes.MapWM_NormStep; LEAK_RESULTS.md)
    SCORE  rotary    score = (R(theta_t) q_t) . (R(theta_s) k_s)   (MapWM's; content can shift the kernel)
           gain      score = mu_q,t mu_k,s (2/sqrt d_h) sum_c A_c cos(theta_t,c - theta_s,c)
                     (model_em_pope.GainKernelLayer, M = 1 = GAIN_GRAIN's GainScalar; mu = softplus(Linear(LN x)) >= 0)

| arm        | step     | score  | class                                   |
|------------|----------|--------|-----------------------------------------|
| MapWM      | raw      | rotary | model_rank.MapFormerWM_r4 (unchanged)   |
| NormStep   | NormStep | rotary | model_codes.MapWM_NormStep (unchanged)  |
| GainRaw    | raw      | gain   | GainWM_Raw (here)                       |
| GainPhase  | NormStep | gain   | GainWM_Phase (here): the gain-phase map |

All four are rank 4, 1 layer, 2 heads, d 128, 32 angles per head, MapWM's PathIntegrator (identical initial omega).

Initial weights. MapWM / NormStep draw their base exactly as MapFormerWM_r4 (NormStep adds only a LayerNorm, no RNG
draw). The gain arms first build the SAME base (same draws, so the global RNG ends in the same state and the trainer's
later use_object_codes draws the same code encoder / readout), then replace each WMTransformerLayer by a GainKernelLayer
built under a forked RNG and copy into it the base layer's v_proj, o_proj, norm1, norm2 and FFN. The only new tensors,
q_gain / k_gain (H x 1 outputs each), are drawn from a separate generator seeded by (initial seed + GAIN_SEED_OFFSET)
with nn.Linear's default distribution; amp_raw is a constant (A_c = 1). So at a seed all four arms share every initial
tensor except the score's content projections (q_proj / k_proj in the rotary arms, q_gain / k_gain in the gain arms),
and leave the global CPU RNG (hence the data stream, ParallelBatchGenerator's base seed = torch.initial_seed()) and the
CUDA RNG untouched. Checked in docs/audits/2026-10-06/gain_phase_equiv.py.

Consequence stated in the prereg: the gain arms have ~33k fewer parameters (q/k projections 2 x 128 x 128 + biases
replaced by 2 x 128 x 2 + biases), exactly as GAIN_GRAIN's GainScalar against MapPoPE-Pair / MapWM.
"""
import math

import torch
import torch.nn as nn

from mapformer.model_codes import _StepOverride, MapWM_NormStep
from mapformer.model_em_pope import GainKernelLayer
from mapformer.model_rank import MapFormerWM_r4

GAIN_SEED_OFFSET = 7_919_000
SHARED_LAYER_PARTS = ("v_proj", "o_proj", "norm1", "norm2", "ffn")


def _linear_default_init_(lin: nn.Linear, gen: torch.Generator):
    """nn.Linear.reset_parameters' distribution (kaiming_uniform a=sqrt(5) -> U(-1/sqrt(fan_in), +), bias the same),
    drawn from `gen` instead of the global RNG."""
    bound = 1.0 / math.sqrt(lin.in_features)
    with torch.no_grad():
        lin.weight.copy_(torch.empty_like(lin.weight).uniform_(-bound, bound, generator=gen))
        if lin.bias is not None:
            lin.bias.copy_(torch.empty_like(lin.bias).uniform_(-bound, bound, generator=gen))


class _GainScore(_StepOverride):
    """MapWM r=4 (step overridable) whose attention is GainKernelLayer with ONE gain per token per head (GainScalar)."""
    N_MODULES = 1

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1, dropout=0.1, grid_size=64, **kw):
        super().__init__(vocab_size, d_model, n_heads, n_layers, dropout, grid_size)
        assert self.n_blocks * 2 == self.d_head and self.action_to_lie.w_in.out_features == 4
        gen = torch.Generator().manual_seed(int(torch.initial_seed()) + GAIN_SEED_OFFSET)
        new = []
        with torch.random.fork_rng(devices=[]):                 # CPU RNG restored on exit; CUDA RNG never touched
            for old in self.layers:
                g = GainKernelLayer(d_model, n_heads, old.dropout.p, self.n_blocks, self.N_MODULES)
                for part in SHARED_LAYER_PARTS:
                    getattr(g, part).load_state_dict(getattr(old, part).state_dict())
                _linear_default_init_(g.q_gain, gen); _linear_default_init_(g.k_gain, gen)
                new.append(g)
        self.layers = nn.ModuleList(new)


class GainWM_Raw(_GainScore):
    """GainRaw: MapWM's raw step (W_out W_in e), gain score."""


class GainWM_Phase(_GainScore):
    """GainPhase, the gain-phase map: NormStep's step (W_out W_in LN(e)) and the gain score. step_ln is created last
    (LayerNorm init draws nothing), as in MapWM_NormStep."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.step_ln = nn.LayerNorm(self.d_model)

    def step(self, tokens, x):
        return self.action_to_lie(self.step_ln(x))


ARMS = {"MapWM": MapFormerWM_r4, "NormStep": MapWM_NormStep, "GainRaw": GainWM_Raw, "GainPhase": GainWM_Phase}
FACTORS = {"MapWM": ("raw", "rotary"), "NormStep": ("norm", "rotary"), "GainRaw": ("raw", "gain"),
           "GainPhase": ("norm", "gain")}


def register(arms: dict) -> dict:
    """Add GainRaw / GainPhase to train_newobj.ARMS (MapWM and NormStep there are already these classes)."""
    for k, v in ARMS.items():
        assert arms.get(k, v) is v, (k, arms.get(k), v)
        arms[k] = v
    return arms
