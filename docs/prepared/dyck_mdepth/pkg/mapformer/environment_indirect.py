"""Indirect Indexing (PoPE paper, arXiv:2509.10534, sec 5.1 and App B.1).

Verbatim from App B.1: "requires the model to locate a target character in a variable-length
source string that is at a certain relative distance (left or right) from a source character ...
We generate source strings of length between 20 and 40 characters from the set of uppercase [A-Z]
and lowercase [a-z] letters by uniform sampling without replacement. Then, we randomly pick a
source character given a randomly sampled source string. Next, we uniformly sample shifts in the
range [-15, +15] ... We use character-level tokenization, i.e. all uppercase and lowercase letters,
individual digits, the delimiter symbol, plus and minus signs are separate tokens. The format of
each example is: <source string>, <source character>, <shift>, <target character>".
Their example: `TzbkWoKDyscBepYvfwxEVQtgPa, c, -8, b`. Splits 1M / 10k / 10k.

Loss and accuracy are on the final (target) token only.

Not stated, chosen here: the shift is written sign-then-digits ("-8", "+12"); a target that would
fall outside the string is rejected and the shift resampled (the paper's examples all land inside);
sequences are left-padded to a fixed block so a batch is rectangular, and padding is masked out.
"""
import numpy as np
import torch

LETTERS = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
DIGITS = [str(i) for i in range(10)]
SYMS = [",", " ", "+", "-"]
PAD = "<pad>"
VOCAB = [PAD] + LETTERS + DIGITS + SYMS
STOI = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BLOCK = 56                      # 40 source chars + ", c, -15, " and padding


class IndirectWorld:
    vocab_size = VOCAB_SIZE
    block = BLOCK

    def __init__(self, min_len=20, max_len=40, max_shift=15):
        self.min_len, self.max_len, self.max_shift = min_len, max_len, max_shift

    def sample(self, n, rng):
        X = np.zeros((n, BLOCK), np.int64)      # inputs, left-padded
        Y = np.zeros(n, np.int64)               # target character
        for i in range(n):
            L = int(rng.integers(self.min_len, self.max_len + 1))
            s = list(rng.choice(len(LETTERS), size=L, replace=False))
            while True:
                j = int(rng.integers(0, L))
                k = int(rng.integers(-self.max_shift, self.max_shift + 1))
                if 0 <= j + k < L:
                    break
            src = LETTERS[s[j]]
            tgt = LETTERS[s[j + k]]
            txt = "".join(LETTERS[c] for c in s) + ", " + src + ", " + ("+" if k >= 0 else "-") + str(abs(k)) + ", "
            ids = [STOI[c] for c in txt]
            X[i, BLOCK - len(ids):] = ids
            Y[i] = STOI[tgt]
        return torch.from_numpy(X), torch.from_numpy(Y)

    def batch(self, n, rng):
        return self.sample(n, rng)


def decode(row):
    return "".join(VOCAB[i] for i in row if i != 0)
