"""EXPTORUS (EXPTORUS_PREREG.md): the exp and softplus phase increments in ATTENTION on the T = 1024 torus.

The two non-negative arms are model_signroute's `exp` and `softplus` step modes (SIGNROUTE ran them on the ring) with the
torus's four move tokens (environment.GridWorld action ids 0-3: N, S, W, E) instead of the ring's two. Every other token's
step is 0 by construction (the observation-step-free rank-4 class of TORUS_EXIST / GENERALITY E3 / TORUS_LRSWEEP):

  SignedAO4_r4    step(m) = W_out W_in e(m)             (model_wrap_exist; anchor, solves)
  AbsAO4_r4       step(m) = |W_out W_in e(m)|           (model_wrap_exist; anchor, fails)
  ExpAO4_r4       step(m) = exp(W_out W_in e(m))        (LRU-style; strictly positive; this file)
  SoftplusAO4_r4  step(m) = softplus(W_out W_in e(m))   (GRAPE-AP / Mamba-Delta style; this file)

All four are built by the same constructor sequence (MapFormerWM base, then the action_to_lie replaced in place), so at one
torch seed their initial weights are bitwise identical (test_exptorus.py measures it); they differ only in the function
applied to the readout. exp(0) = 1 and softplus(0) = ln 2, so at init the two non-negative arms already step every move by
~1.05 and ~0.71 (SIGNROUTE's measured means on the ring), i.e. they start as a RoPE-like clock with a rescaled omega, and on
the high-omega channels their pair sums omega (a(x) + a(y)) start PAST pi -- the hypothesised route of G-3 A2's exp success
(cancel by wrapping, at 2 pi, rather than through zero). That start is part of the parameterisation, not a confound this
batch removes; the analysis records it per channel.

`register()` adds the two names to train_variant.VARIANT_MAP at runtime, so the existing trainer and evaluators run these
classes without any change to an existing module (train_exptorus.py / eval_exptorus.py call it before their main()).
"""
from mapformer.model_signroute import _signroute_variant

TORUS_MOVES = (0, 1, 2, 3)          # environment.GridWorld: action tokens 0-3 (ACTION_DELTAS N, S, W, E)

_Exp = _signroute_variant("exp")
_Softplus = _signroute_variant("softplus")


class MapFormerWM_ExpAO4_r4(_Exp):
    """Observation-step-free rank-4 MapFormer-WM on the 2D torus, move steps exp(W_out W_in e(move))."""
    def __init__(self, vocab_size, *a, action_ids=TORUS_MOVES, **kw):
        super().__init__(vocab_size, *a, action_ids=action_ids, **kw)


class MapFormerWM_SoftplusAO4_r4(_Softplus):
    """Observation-step-free rank-4 MapFormer-WM on the 2D torus, move steps softplus(W_out W_in e(move))."""
    def __init__(self, vocab_size, *a, action_ids=TORUS_MOVES, **kw):
        super().__init__(vocab_size, *a, action_ids=action_ids, **kw)


NEW_VARIANTS = {"ExpAO4_r4": MapFormerWM_ExpAO4_r4, "SoftplusAO4_r4": MapFormerWM_SoftplusAO4_r4}


def register():
    """Add the two variants to train_variant.VARIANT_MAP (idempotent). Refuses to shadow a different class under the same
    name, so a later edit of train_variant that registers these names differently cannot be silently overridden."""
    from mapformer.train_variant import VARIANT_MAP
    for k, cls in NEW_VARIANTS.items():
        have = VARIANT_MAP.get(k)
        if have is not None and have is not cls:
            raise RuntimeError(f"train_variant.VARIANT_MAP already maps {k!r} to {have!r}; refusing to override")
        VARIANT_MAP[k] = cls
    return VARIANT_MAP
