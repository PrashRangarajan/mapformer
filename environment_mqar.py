"""MQAR -- Multi-Query Associative Recall (Arora et al. 2023), the field's standard
synthetic for this architecture family.

Construction, from the descriptions in the local corpus rather than from memory:
a multi-query version of the induction-head task in which "a model has to recall the
token following a query token multiple times". Reported settings there are
sequence length 256/512 and 16/64/256 key-value pairs.

    [ k1 v1  k2 v2  ...  kN vN ][ filler ... q  ... q ... ]
                                              ^        ^
                                        predict v(q)  predict v(q)

Keys and values are drawn from disjoint blocks so a query is unambiguous, keys within
an episode are distinct, and every query names a key that appeared in the KV block.
Scored at query positions only; chance is 1/n_values.

DEVIATION FROM THE PUBLISHED SETTING, stated rather than hidden: the standard sweep
uses a vocabulary of 8192. At d_model=128 that embedding table alone would be
1.05M parameters against a 204K model -- the vocabulary would BE the model. This uses
a smaller vocabulary by default. Arms remain comparable to each other; none of these
numbers is comparable to a published MQAR figure, and the pre-registration does not
ask them to be.

MQAR is expected to be at ceiling here. See MQAR_PREREG.md: its difficulty knob is
STATE SIZE, and it exists to measure how close a sub-quadratic model gets to softmax
attention -- which every arm here already is.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class MQARWorld:
    def __init__(self, n_keys: int = 128, n_values: int = 128, n_kv: int = 16,
                 n_queries: int = 16, seed: Optional[int] = None):
        if n_kv > n_keys:
            raise ValueError("n_kv cannot exceed n_keys (keys are distinct)")
        self.n_keys, self.n_values = n_keys, n_values
        self.n_kv, self.n_queries = n_kv, n_queries
        # vocab: [keys][values][filler]
        self.key_offset = 0
        self.val_offset = n_keys
        self.filler_tok = n_keys + n_values
        self.unified_vocab_size = n_keys + n_values + 1
        self._rng = np.random.RandomState(seed)

    def generate_episode(self, T: int = 256, rng=None):
        if rng is None:
            rng = self._rng
        need = 2 * self.n_kv + 2 * self.n_queries
        if T < need:
            raise ValueError(f"T={T} too small for {self.n_kv} pairs + {self.n_queries} queries")

        keys = rng.choice(self.n_keys, size=self.n_kv, replace=False)
        vals = rng.randint(0, self.n_values, size=self.n_kv)
        kv = dict(zip(keys.tolist(), vals.tolist()))

        toks = []
        for k, v in zip(keys, vals):
            toks.append(int(k) + self.key_offset)
            toks.append(int(v) + self.val_offset)

        # queries scattered through filler, each naming a key from the block
        qs = rng.choice(self.n_kv, size=self.n_queries, replace=False)
        tail = T - len(toks)
        slots = sorted(rng.choice(tail - 1, size=self.n_queries, replace=False))
        score_pos, answers = [], []
        cur = 0
        for slot, qi in zip(slots, qs):
            while cur < slot:
                toks.append(self.filler_tok); cur += 1
            score_pos.append(len(toks))              # predict AT the query token
            toks.append(int(keys[qi]) + self.key_offset)
            answers.append(int(kv[int(keys[qi])]) + self.val_offset)
            cur += 1
        while len(toks) < T:
            toks.append(self.filler_tok)

        t = torch.tensor(toks[:T], dtype=torch.long)
        keep = [i for i, p in enumerate(score_pos) if p + 1 < T]
        return (t, [score_pos[i] for i in keep], [answers[i] for i in keep],
                {"T": int(t.shape[0]), "n_scored": len(keep), "n_kv": self.n_kv})

    def generate_batch(self, batch_size: int, T: int = 256, rng=None):
        toks, sps, ans, infos = [], [], [], []
        for _ in range(batch_size):
            x, sp, a, i = self.generate_episode(T, rng)
            toks.append(x); sps.append(sp); ans.append(a); infos.append(i)
        return torch.stack(toks), sps, ans, infos
