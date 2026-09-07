"""Flip-Flop language modelling (Liu et al. 2023) -- the PUBLISHED external check.

Why this exists. `environment_recency.py` is our own probe, and its headline (a
fixed index code cannot do contextual counting) reproduces CoPE's published claim
in a different architecture. That is worth anchoring on the field's own benchmark
rather than only on a task we designed.

The task, per the definition in CoPE sec 5.1 (arXiv:2405.11582, read first-hand
in papers/txt/cope.txt), which cites Liu et al. 2023:

    alternating (instruction, bit) pairs, instructions {w, i, r}
    w = write the following bit    i = ignore it    r = read
    every string starts with w and ends with r
    at r, emit the bit of the most recent w

    "w0i1r0w1i0i1i1r"  ->  1

OOD is defined by raising the ignore density, which lengthens the distance back
to the last write. Standard split: train p_i = 0.8, OOD-dense p_i = 0.98,
OOD-sparse p_i = 0.1.

Relation to our own task: this is `environment_recency.py` at k = 1, with `w`
tokens as the counted class and `i` as the filler. So it tests the CONTENT GATE --
select the write instructions, attend to the most recent -- but not the graded
offset that gives the per-k mechanism readout. That is the trade: a published
benchmark, one k.

A shortcut this task has BY CONSTRUCTION, which we measure rather than remove:
two consecutive reads with no write between them have the SAME answer, and at
p_w = p_r = 0.1 that happens about half the time. It is a property of the
published task, so `score_final_only` exists to report both readings rather than
to quietly redefine the benchmark. See validate_flipflop.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

W, I, R = 0, 1, 2


class FlipFlopWorld:
    def __init__(self, p_write: float = 0.1, p_ignore: float = 0.8,
                 p_read: float = 0.1, score_final_only: bool = False,
                 seed: Optional[int] = None):
        tot = p_write + p_ignore + p_read
        self.p = np.array([p_write, p_ignore, p_read]) / tot
        self.score_final_only = score_final_only
        # vocab: [w i r][bit0 bit1]
        self.instr_offset = 0
        self.bit_offset = 3
        self.unified_vocab_size = 5
        self._rng = np.random.RandomState(seed)

    def generate_episode(self, T: int = 512, rng=None):
        """T is the token budget; the stream is (instruction, bit) pairs.

        Returns (tokens, score_pos, answers, info). `score_pos[j]` is the index of
        the READ token; the logit there is scored against the bit that follows,
        matching train_recency's convention.
        """
        if rng is None:
            rng = self._rng
        n_pairs = T // 2
        if n_pairs < 2:
            raise ValueError("T too small")
        ops = rng.choice(3, size=n_pairs, p=self.p)
        ops[0] = W                     # every string starts with a write
        ops[-1] = R                    # and ends with a read
        bits = rng.randint(0, 2, size=n_pairs)

        tokens, score_pos, answers, mem = [], [], [], None
        for j, (o, b) in enumerate(zip(ops, bits)):
            if o == R:
                # a read's bit is not free: it IS the remembered bit
                b = mem if mem is not None else 0
                score_pos.append(len(tokens))
            tokens.append(int(o) + self.instr_offset)
            tokens.append(int(b) + self.bit_offset)
            if o == W:
                mem = int(b)
            if o == R:
                answers.append(int(b) + self.bit_offset)

        if self.score_final_only and score_pos:
            score_pos, answers = score_pos[-1:], answers[-1:]
        toks = torch.tensor(tokens[:T], dtype=torch.long)
        keep = [j for j, p in enumerate(score_pos) if p + 1 < len(toks)]
        info = {"n_pairs": n_pairs, "T": int(toks.shape[0]),
                "n_scored": len(keep), "ops": ops}
        return (toks, [score_pos[j] for j in keep],
                [answers[j] for j in keep], info)

    def generate_batch(self, batch_size: int, T: int = 512, rng=None):
        toks, sps, ans, infos = [], [], [], []
        for _ in range(batch_size):
            t, sp, a, i = self.generate_episode(T, rng)
            toks.append(t); sps.append(sp); ans.append(a); infos.append(i)
        return torch.stack(toks), sps, ans, infos
