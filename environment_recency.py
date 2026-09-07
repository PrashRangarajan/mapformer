"""Recency task: retrieve the k-th most recent symbol. The CLOCK half of the 2x2.

Why this task exists
--------------------
Every task in this repository is a MAP task -- torus revisit, Match-Query,
compositional, family tree, MiniGrid, MiniWorld, and `environment_clock.py`
(which despite the name uses signed +/-1/+/-2 ticks, so it is modular NAVIGATION).
On all of them a signed increment beats a monotone one, and the review's central
dichotomy -- signed accumulator measures net displacement (a MAP), monotone
measures elapsed path (a CLOCK), mutually exclusive, each correct for one job --
has therefore only ever been tested on one side. A dichotomy that only ever points
one way is not a dichotomy; it is "signed is better".

This is the missing cell. The answer here depends on ELAPSED COUNT, not on net
displacement, so the theory says the ordering must INVERT.

Design, and why it is a MATCH and not a DECODE
----------------------------------------------
`environment_map_query.py` asked the model to DECODE absolute position and got
0.121 against a chance of 0.016, because MapFormer's position code exists to make
positions COMPARABLE, never readable. Asking "how many steps ago?" would repeat
that error and produce a floor on every arm rather than a crossover.

So the query names the offset and the model retrieves the CONTENT there:

    ... s s s [q_k] <MASK> s s [q_k'] <MASK> s ...

At each `q_k` the model predicts the k-th most recent SYMBOL token. That is the
same operation the paper's revisit task performs -- attend to the key at a known
relative position and read off its content -- moved from the space axis to the
time axis. Nothing has to be decoded.

Why a monotone increment should win here
----------------------------------------
theta_t - theta_s is the accumulator difference between query and key.

  monotone  Delta >= 0, so theta is strictly increasing in the symbol count and
            theta_t - theta_s is a bijection with "how many symbols back". The
            k-back key is uniquely addressed. A model can reach this by learning
            Delta = 1 on symbols and Delta = 0 on query/mask tokens -- i.e. a
            CONTENT-DEPENDENT COUNTER, which is exactly what CoPE does and the
            regime the CoPE / CARoPE / GRAPE-AP family was built for.

  signed    theta is a random walk over the symbol stream, so it revisits values
            and many candidate keys share one theta. The difference no longer
            addresses a unique position. G7 below MEASURES this collision rate
            rather than assuming it.

Note the arms are not symmetric in what they are allowed to learn. `Abs` / `Pos` /
`CARoPE` are CONSTRAINED monotone. `Signed` is UNCONSTRAINED -- it may learn a
monotone code if that is what the task rewards. So there are two nested
predictions, and the second is the more interesting:

  (1) constraining to monotone costs nothing here, where on the torus it costs
      -0.215 / -0.280 at T=512/1024 -> a crossover INTERACTION.
  (2) the unconstrained Signed arm LEARNS a monotone code here (opposition score
      high) and a cancelling one on the torus (opposition 0.11) -> the accumulator
      exponent alpha, measured on a trained model, READS OFF what the task
      demanded. That would make alpha a diagnostic rather than a description.

An index code (RoPE) should also do well here, since k-back is what index RoPE
natively encodes -- so the whole position axis should invert too, not just the
sign. That is a stronger claim than the sign result alone and is the reason to
keep RoPE / PlainFlat arms in the batch.

Shortcut risks, all gated in validate_recency.py -- NONE of them assumed:
  - answer n-gram: two queries in a row with no symbol between them have related
    answers, and consecutive same-k queries repeat outright. `min_gap` exists to
    control this and the validator sweeps it; the Match-Query dedup was found
    necessary the same way, by a gate failing.
  - "most recent symbol always": ignore k entirely and answer s_{t-1}. Solves
    every k=1 query for free.
  - marginal: symbols are drawn uniformly so this should sit at chance, but a
    skewed draw would break that and it is measured, not assumed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class RecencyWorld:
    """Stream of symbols with interleaved k-back queries."""

    def __init__(self, n_symbols: int = 16, k_max: int = 8,
                 p_query: float = 0.25, min_gap: Optional[int] = None,
                 seed: Optional[int] = None):
        """`min_gap` = symbols that must be emitted between consecutive queries.

        DEFAULT IS k_max, and that is a structural guarantee rather than a tuned
        number. Two adjacent queries return the SAME symbol exactly when
        `k_2 == k_1 + g`, where g is the number of symbols between them: query 2
        counts back from a history that is g longer. So the answer stream is
        auto-correlated at rate `(k_max - g) / k_max^2`, and once `g >= k_max`
        the relation needs `k_2 > k_max` and is impossible.

        Measured in `RECENCY_GATES.md`, and the arithmetic is exact -- predicted
        order-1 n-gram 0.165 / 0.121 / 0.0625 at g = 1 / 4 / 8, observed
        0.147 / 0.070 / 0.060 against a chance of 0.0625. It is a parameter
        rather than a hardcoded value so the validator can re-measure the damage
        for any (k_max, p_query) instead of the design assuming it away; this is
        the same failure Match-Query's cell dedup exists to prevent.

        Cost of the guarantee: the scored rate falls from 0.16 to 0.077 per
        token, i.e. ~20 scored queries per 256-token episode.
        """
        if min_gap is None:
            min_gap = k_max
        if not 1 <= k_max:
            raise ValueError("k_max must be >= 1")
        self.n_symbols = n_symbols
        self.k_max = k_max
        self.p_query = p_query
        self.min_gap = min_gap

        # vocab layout: [symbols][query offsets q_1..q_kmax][MASK]
        self.sym_offset = 0
        self.query_offset = n_symbols
        self.mask_tok = n_symbols + k_max
        self.unified_vocab_size = n_symbols + k_max + 1

        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    def generate_episode(self, T: int = 256, rng=None):
        """Emit ~T tokens. Returns (tokens, score_pos, answers, info).

        `score_pos[i]` is the index of the q_k token, matching Match-Query's
        convention: logits at that index are scored against `answers[i]`, so a
        trainer slices the symbol block out of the logits exactly as
        `train_match_query._match_logits` does.
        """
        if rng is None:
            rng = self._rng

        tokens: list[int] = []
        sym_positions: list[int] = []   # token indices holding a symbol
        sym_values: list[int] = []      # the symbol at each of those
        score_pos: list[int] = []
        answers: list[int] = []
        offsets: list[int] = []
        gap = self.min_gap              # symbols emitted since the last query

        while len(tokens) < T:
            can_query = (len(sym_values) >= self.k_max
                         and gap >= self.min_gap
                         and len(tokens) + 2 <= T)
            if can_query and rng.random() < self.p_query:
                k = int(rng.randint(1, self.k_max + 1))
                score_pos.append(len(tokens))
                tokens.append(self.query_offset + (k - 1))
                tokens.append(self.mask_tok)
                # k-th most recent symbol: k=1 is the immediately preceding one
                answers.append(sym_values[-k] + self.sym_offset)
                offsets.append(k)
                gap = 0
            else:
                s = int(rng.randint(0, self.n_symbols))
                sym_positions.append(len(tokens))
                sym_values.append(s)
                tokens.append(s + self.sym_offset)
                gap += 1

        toks = torch.tensor(tokens[:T], dtype=torch.long)
        # a query pair truncated by the T cap must not be scored
        keep = [i for i, p in enumerate(score_pos) if p + 1 < T]
        info = {"sym_positions": sym_positions, "sym_values": sym_values,
                "offsets": [offsets[i] for i in keep], "T": int(toks.shape[0]),
                "n_scored": len(keep)}
        return (toks, [score_pos[i] for i in keep],
                [answers[i] for i in keep], info)

    def generate_batch(self, batch_size: int, T: int = 256, rng=None):
        toks, sps, ans, infos = [], [], [], []
        for _ in range(batch_size):
            t, sp, a, info = self.generate_episode(T, rng)
            toks.append(t); sps.append(sp); ans.append(a); infos.append(info)
        return torch.stack(toks), sps, ans, infos
