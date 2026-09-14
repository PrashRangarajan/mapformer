"""Multi-digit addition, in the format of Cho et al. 2024 (position coupling), for asking
whether a learned path integrator can DISCOVER position coupling (ADDITION_DESIGN.md).

Sequence (zero-padded operands, MSB first; sum reversed, LSB first; n = operand length):

    $  a_{n-1} .. a_0  +  b_{n-1} .. b_0  =  s_0 s_1 .. s_n  $

The model is trained to predict s_0..s_n and the closing '$' (next-token loss on those positions only).

Two formats share one vocabulary so embedding tables are parameter-identical:
  shared  every digit uses tokens 4..13 (the literature's format)
  role    first operand uses 4..13, second operand 14..23, sum 24..33

Coupled position IDs (for the CoupledRoPE oracle, Cho et al. sec 3.1): digit of significance j in
either operand and in the sum gets start + (n-1-j); the sum's extra top digit (j = n) gets start - 1;
'+' and '=' get start + n; '$' and PAD get 0. `start` is random in training, fixed at 1 in evaluation.
"""
from __future__ import annotations

import numpy as np
import torch

PAD, EOS, PLUS, EQ = 0, 1, 2, 3
D_SHARED, D_A, D_B, D_S = 4, 4, 14, 24
VOCAB = 34
MAX_POS = 512


class AdditionWorld:
    def __init__(self, fmt: str = "shared"):
        assert fmt in ("shared", "role"), fmt
        self.fmt = fmt
        self.vocab_size = VOCAB

    def _off(self, role):
        if self.fmt == "shared":
            return D_SHARED
        return {"a": D_A, "b": D_B, "s": D_S}[role]

    @staticmethod
    def sample_operand(rng, n_digits):
        """Uniform among numbers with exactly n_digits digits (0-9 for n_digits == 1)."""
        if n_digits == 1:
            return [int(rng.randint(0, 10))]
        d = [int(rng.randint(1, 10))] + [int(x) for x in rng.randint(0, 10, size=n_digits - 1)]
        return d  # MSB first

    @staticmethod
    def add_digits(a, b):
        """a, b: MSB-first digit lists of equal length n. Returns sum digits LSB first, length n+1."""
        n = len(a); out = []; carry = 0
        for j in range(n):
            t = a[n - 1 - j] + b[n - 1 - j] + carry
            out.append(t % 10); carry = t // 10
        out.append(carry)
        return out

    def encode(self, a, b, start=1):
        """-> tokens, loss mask (True where the NEXT token is a sum digit or the closing '$'), pos ids."""
        n = max(len(a), len(b))
        a = [0] * (n - len(a)) + a; b = [0] * (n - len(b)) + b
        s = self.add_digits(a, b)
        toks = [EOS] + [self._off("a") + d for d in a] + [PLUS] + [self._off("b") + d for d in b] + [EQ] \
            + [self._off("s") + d for d in s] + [EOS]
        pos = [0] + [start + (n - 1 - j) for j in range(n - 1, -1, -1)] + [start + n] \
            + [start + (n - 1 - j) for j in range(n - 1, -1, -1)] + [start + n] \
            + [start + (n - 1 - j) for j in range(n)] + [start - 1] + [0]
        first_sum = 1 + n + 1 + n + 1          # index of s_0 in toks
        tgt = np.zeros(len(toks), dtype=bool)
        tgt[first_sum:] = True                 # s_0..s_n and closing '$' are predicted
        return toks, tgt, pos, s

    def batch(self, batch_size, rng, dmax=None, n_digits=None, random_start=False):
        """Balanced sampling over digit counts 1..dmax (both operands independently), or exactly
        n_digits for evaluation. Right-padded. Returns tokens (B, L), target mask for tokens (B, L)
        (True at positions whose token is a target), pos ids (B, L), and sum digit lists."""
        seqs, tgts, poss, sums = [], [], [], []
        for _ in range(batch_size):
            if n_digits is not None:
                la = lb = n_digits
            else:
                la, lb = int(rng.randint(1, dmax + 1)), int(rng.randint(1, dmax + 1))
            a, b = self.sample_operand(rng, la), self.sample_operand(rng, lb)
            n = max(la, lb)
            start = int(rng.randint(1, MAX_POS - n - 2)) if random_start else 1
            t, m, p, s = self.encode(a, b, start)
            seqs.append(t); tgts.append(m); poss.append(p); sums.append(s)
        L = max(len(t) for t in seqs)
        T = torch.full((batch_size, L), PAD, dtype=torch.long)
        M = torch.zeros((batch_size, L), dtype=torch.bool)
        P = torch.zeros((batch_size, L), dtype=torch.long)
        for i, (t, m, p) in enumerate(zip(seqs, tgts, poss)):
            T[i, :len(t)] = torch.tensor(t); M[i, :len(m)] = torch.from_numpy(m); P[i, :len(p)] = torch.tensor(p)
        return T, M, P, sums
