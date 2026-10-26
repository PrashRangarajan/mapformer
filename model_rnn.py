"""Recurrent baselines for RNN_TORUS_PREREG.md: LSTM / GRU next-token predictors on the interleaved torus stream
[a1, o1, a2, o2, ...], parameter-matched to the path model (MapWM Vanilla_r4, 204,757 params) and to index RoPE at
4 layers (798,741 params).

Form: token embedding (dim = hidden) -> nn.LSTM / nn.GRU (n layers, hidden h, batch_first, inter-layer dropout 0.1 when
n > 1, as model_baselines_extra.LSTMBaseline) -> LayerNorm -> Linear(vocab). The same form as the repo's existing
`LSTM` variant (model_baselines_extra.LSTMBaseline, which fixes h = d_model = 128, 137,749 params); here h is chosen per
arm so the parameter count matches the anchor. LSTM forget-gate bias starts at 1 (b_ih forget slice 1, b_hh 0;
Jozefowicz et al. 2015) -- the standard strengthening for long memory; GRU uses PyTorch's default init.

The constructor keeps train_variant's signature (vocab_size, d_model, n_heads, n_layers, grid_size): d_model and n_heads
are IGNORED (the hidden size is fixed by the variant name), n_layers MUST equal the variant's depth (asserted), so a
checkpoint rebuilt by eval_noise_refine / eval_rank_strata from its config gets the trained shape.

`register()` adds the arms to train_variant.VARIANT_MAP (a module-level dict that eval_noise_refine and eval_rank_strata
import by reference), so the existing trainer and evaluators run them unchanged; nothing in train_variant.py changes.
"""
import math

import torch
import torch.nn as nn

VOCAB = 21                    # GridWorld(64, n_obs_types=16): 4 actions + 16 observation types + blank
TARGET_PATH = 204_757         # Vanilla_r4 (MapWM shared rank 4), 1 layer, d 128 -- counted 2026-10-26
TARGET_ROPE4 = 798_741        # RoPE --n-layers 4 (FACTORIAL_N8_PREREG.md)


class RNNBaseline(nn.Module):
    """Embedding(h) -> LSTM/GRU(h, n layers) -> LayerNorm -> Linear(vocab)."""
    CELL = "LSTM"
    HIDDEN = 128
    LAYERS = 1

    def __init__(self, vocab_size, d_model=128, n_heads=2, n_layers=None, grid_size=64, dropout=0.1, **kwargs):
        super().__init__()
        if n_layers is None:
            n_layers = self.LAYERS
        assert n_layers == self.LAYERS, (f"{type(self).__name__} is a {self.LAYERS}-layer arm; got n_layers={n_layers} "
                                         f"(pass --n-layers {self.LAYERS})")
        h = self.HIDDEN
        self.vocab_size, self.hidden, self.n_layers, self.cell = vocab_size, h, n_layers, self.CELL
        self.token_emb = nn.Embedding(vocab_size, h)
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[self.CELL]
        self.rnn = rnn(input_size=h, hidden_size=h, num_layers=n_layers,
                       dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        self.out_norm = nn.LayerNorm(h)
        self.out_proj = nn.Linear(h, vocab_size)
        if self.CELL == "LSTM":                      # gate order i, f, g, o (PyTorch): forget slice = [h, 2h)
            with torch.no_grad():
                for l in range(n_layers):
                    getattr(self.rnn, f"bias_ih_l{l}")[h:2 * h].fill_(1.0)
                    getattr(self.rnn, f"bias_hh_l{l}")[h:2 * h].zero_()

    def forward(self, tokens):
        y, _ = self.rnn(self.token_emb(tokens))
        return self.out_proj(self.out_norm(y))


def n_params(cell, h, layers, vocab=VOCAB):
    """Closed-form parameter count of RNNBaseline (checked against the module in test_rnn_torus.py)."""
    g = {"LSTM": 4, "GRU": 3}[cell]
    rnn = sum(g * (h * h + h * h + 2 * h) for _ in range(layers))   # input size = h at every layer
    return vocab * h + rnn + 2 * h + h * vocab + vocab


def match_hidden(cell, layers, target, vocab=VOCAB):
    """The hidden size whose parameter count is closest to target."""
    return min(range(8, 2048), key=lambda h: (abs(n_params(cell, h, layers, vocab) - target), h))


def make_arm(cell, layers, hidden):
    name = f"{cell}_L{layers}_h{hidden}"
    return name, type(name, (RNNBaseline,), {"CELL": cell, "HIDDEN": hidden, "LAYERS": layers})


# The registered arms (RNN_TORUS_PREREG.md, "Arms"): hidden sizes from match_hidden, frozen here so a later change of the
# matching rule cannot silently change a trained arm's shape. test_rnn_torus.py re-derives and checks them.
ARM_SPECS = {
    # small: ~204,757 (= the path model)
    "LSTM_L1_h157": ("LSTM", 1, 157),
    "GRU_L1_h181": ("GRU", 1, 181),
    "LSTM_L2_h111": ("LSTM", 2, 111),
    "GRU_L2_h128": ("GRU", 2, 128),
    # large: ~798,741 (= index RoPE at 4 layers)
    "LSTM_L1_h313": ("LSTM", 1, 313),
    "GRU_L1_h361": ("GRU", 1, 361),
    "LSTM_L2_h222": ("LSTM", 2, 222),
    "GRU_L2_h256": ("GRU", 2, 256),
}
RNN_VARIANTS = {}
for _n, (_c, _l, _h) in ARM_SPECS.items():
    _name, _cls = make_arm(_c, _l, _h)
    assert _name == _n, (_name, _n)
    RNN_VARIANTS[_n] = _cls
    globals()[_n] = _cls          # picklable by name if ever needed


def register():
    """Add the RNN arms to train_variant.VARIANT_MAP (idempotent). Refuses to shadow an existing variant."""
    from mapformer import train_variant as TV
    for k, v in RNN_VARIANTS.items():
        if k in TV.VARIANT_MAP and TV.VARIANT_MAP[k] is not v:
            raise RuntimeError(f"variant name {k} already exists in train_variant.VARIANT_MAP")
        TV.VARIANT_MAP[k] = v
    return TV.VARIANT_MAP


if __name__ == "__main__":
    for k, (c, l, h) in ARM_SPECS.items():
        m = RNN_VARIANTS[k](VOCAB, n_layers=l)
        n = sum(p.numel() for p in m.parameters())
        tgt = TARGET_PATH if n < 500_000 else TARGET_ROPE4
        print(f"{k:14s} params {n:,} (target {tgt:,}, {100 * (n - tgt) / tgt:+.2f}%; match_hidden -> "
              f"{match_hidden(c, l, tgt)}; state floats per step {h * l * (2 if c == 'LSTM' else 1)})")
