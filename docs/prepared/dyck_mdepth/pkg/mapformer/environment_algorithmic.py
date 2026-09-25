"""The looped-transformer literature's own tasks, so our loop results can be
compared to it rather than sitting beside it.

WHY. Everything this project has measured about looping is on navigation:
Match-Query and the torus. Match-Query has no known ITERATIVE solution -- answering
it is compute-position, match, read, a few COMPOSED operations rather than N
iterations of one -- so the loop's win there is depth substitution, i.e. the
ALBERT/MoR parameter-efficiency claim. The looped literature's actual claim is
different: looping supplies a variable number of ALGORITHMIC STEPS, and its
benchmarks are tasks with a known iterative solution evaluated for LENGTH
GENERALIZATION (arXiv 2409.15647: parity, binary addition, copying).

Two tasks here, chosen to point in opposite directions:

  PARITY  running parity of a bit string. The canonical iterative task, and the one
          where path integration has a SHARP mechanistic prediction. MapFormer's
          angle is theta = omega * cumsum(Delta(x_t)) wrapped mod 2*pi; parity is a
          running sum mod 2. Delta = pi for '1' and 0 for '0' puts the answer
          directly in the rotation. If path integration ever helps on a standard
          algorithmic task, this is it.

  COPY    reproduce the input after a separator. Positional retrieval with NO
          iterative structure and nothing to accumulate. Index position should be
          sufficient, so path integration should buy little. This is the control
          that stops "path integration helps" being a claim about our pipeline
          rather than about the task.

Both are scored at held-out positions only, both report their measured chance rate,
and both are gated by validate_algorithmic.py before any training.

Length generalization is the point: train short, test long. The generators take the
length as an argument so eval lengths need no retraining.
"""
import numpy as np
import torch


class ParityWorld:
    """Running parity. tokens are bits; target at t is parity of bits[0..t]."""
    name = "parity"
    vocab_size = 2          # {0, 1}; targets reuse the same two ids
    chance = 0.5

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    def batch(self, batch_size: int, length: int, rng=None):
        r = rng if rng is not None else self.rng
        bits = r.randint(0, 2, size=(batch_size, length))
        tgt = np.cumsum(bits, axis=1) % 2
        # every position is scored except the first, which is trivially the bit
        mask = np.ones((batch_size, length), dtype=bool)
        mask[:, 0] = False
        return (torch.from_numpy(bits).long(), torch.from_numpy(tgt).long(),
                torch.from_numpy(mask))


class CopyWorld:
    """Reproduce the input after a separator. No iterative structure."""
    name = "copy"
    chance = None           # 1/n_symbols, set in __init__

    def __init__(self, n_symbols: int = 8, seed: int = 0):
        self.n_symbols = n_symbols
        self.sep = n_symbols
        self.vocab_size = n_symbols + 1
        self.chance = 1.0 / n_symbols
        self.rng = np.random.RandomState(seed)

    def batch(self, batch_size: int, length: int, rng=None):
        r = rng if rng is not None else self.rng
        src = r.randint(0, self.n_symbols, size=(batch_size, length))
        sep = np.full((batch_size, 1), self.sep)
        # input:  s_1..s_n SEP s_1..s_{n-1}   (teacher forcing)
        # target:            s_1 s_2 .. s_n   at the positions after SEP
        toks = np.concatenate([src, sep, src[:, :-1]], axis=1)
        tgt = np.zeros_like(toks)
        tgt[:, length:] = src
        mask = np.zeros(toks.shape, dtype=bool)
        mask[:, length:] = True         # score only the reproduction half
        return (torch.from_numpy(toks).long(), torch.from_numpy(tgt).long(),
                torch.from_numpy(mask))


WORLDS = {"parity": ParityWorld, "copy": CopyWorld}
