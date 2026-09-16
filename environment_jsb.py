"""Bach Chorales (JSB), as used by the PoPE paper (arXiv:2509.10534, App B.1).

Verbatim: "This dataset (JSB) consists of 4-part scored choral music, which are represented as a
matrix with rows corresponding to voices and columns to time discretized to 16th notes ... We
serialize this matrix in raster-scan fashion by first going down the rows and then moving right
through the columns as in prior work (Huang et al., 2019). We use the variant of the dataset with
16th note temporal 'quantizations' where silence is represented by a pitch of -1 rather than NaN,
available in JSON file format [github.com/czhuang/JSB-Chorales-dataset] ... maximum sequence length
of 2048 for training with 229/76/77 sequences present in the train/validation/test sets. We use a
vocabulary size of 90 which includes the MIDI notes, silence and padding tokens."

Our copy of that JSON has exactly 229/76/77 pieces, pitches in [-1, 81]. Token ids: 0 = PAD,
1 = silence (-1), pitch p -> p + 2. Vocabulary padded to 90 as stated. Pieces longer than 2048
tokens are truncated (mean piece is 242 timesteps = 968 tokens, so this is rare).
"""
import json
from pathlib import Path

import numpy as np
import torch

PATH = Path(__file__).resolve().parent / "data" / "Jsb16thSeparated.json"
PAD, SIL, VOCAB_SIZE, MAXLEN = 0, 1, 90, 2048


def _tok(piece):
    ids = [SIL if v < 0 else int(v) + 2 for step in piece for v in step]   # raster scan: voices, then time
    return ids[:MAXLEN]


def load(split):
    d = json.load(open(PATH))[split]
    X = torch.zeros(len(d), MAXLEN, dtype=torch.long)
    M = torch.zeros(len(d), MAXLEN, dtype=torch.bool)
    for i, p in enumerate(d):
        ids = _tok(p)
        X[i, :len(ids)] = torch.tensor(ids); M[i, :len(ids)] = True
    return X, M


def splits():
    return {k: load(k) for k in ("train", "valid", "test")}
