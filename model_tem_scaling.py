"""TEMFaithful at varying d_g — parameter-scaling sweep.

TEMFaithful's 'model' (excluding vocab I/O) is almost entirely A_a, the
per-action transition matrices, which scale as n_actions * d_g**2:
  d_g=32  -> A_a   4,096
  d_g=64  -> A_a  16,384   (default TEMFaithful)
  d_g=128 -> A_a  65,536
  d_g=256 -> A_a 262,144

Question: is TEM's task performance parameter-saturated? If d_g=256
performs ~ the same as d_g=64, TEM's solving power is the FIXED Hopfield
retrieval algorithm, not parameter count — strong evidence for the
inductive-bias account. If TEM-Big >> TEM, TEM is merely under-
parameterised.

d_x (content dim) is left at the default (d_model // 2 = 64) so the
sweep isolates dynamics capacity (A_a), not I/O capacity.
"""

from __future__ import annotations

from .model_tem_faithful import TEMFaithful


def _make_tem(d_g_val: int):
    class M(TEMFaithful):
        def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=1,
                     dropout=0.1, grid_size=64, **kwargs):
            kwargs.pop("d_g", None)
            super().__init__(vocab_size, d_model=d_model, n_heads=n_heads,
                             n_layers=n_layers, dropout=dropout,
                             grid_size=grid_size, d_g=d_g_val, **kwargs)
    M.__name__ = f"TEMFaithful_dg{d_g_val}"
    M.__qualname__ = M.__name__
    return M


TEMFaithful_dg32 = _make_tem(32)
TEMFaithful_dg128 = _make_tem(128)
TEMFaithful_dg256 = _make_tem(256)
# d_g=64 is the default TEMFaithful; no separate class needed.
