"""Floors for GAIN_PHASE's readout (object-identity accuracy at revisits whose answer is an object, test pool, LEAK's
eval stream: leak_eval.sequences('test'), 200 sequences, T = 1024). CPU, calls the task code (rule 11).
  uniform-16        1/16: a guess among the sequence's 16 objects
  most-frequent     the object seen most often so far in the sequence (ties: most recent)
  last-object       the most recent object observation
  retrace           gate_newobj's retrace predictor (while a run reverses the previous one, the observation j steps
                    back), scored on object targets; falls back to most-frequent when retrace has no object guess
Output: gain_phase_floor_out.txt"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/prashr")
from mapformer.environment_newobj import N_SPECIAL
from mapformer.leak_eval import sequences

toks, revs = sequences("test")
hit = Counter(); tot = 0
for tok, rev in zip(toks.numpy(), revs.numpy()):
    a, o, r = tok[0::2], tok[1::2], rev[1::2]
    T = len(a); cnt = Counter(); last = None; prev_len = cur = k = 0
    for t in range(T):
        if t > 0 and a[t] == a[t - 1]:
            cur += 1
        else:
            prev_len = cur if (t > 0 and (a[t] ^ 1) == a[t - 1]) else 0; cur = 1; k = 0
        if prev_len > 0:
            k += 1
        if r[t] and o[t] >= N_SPECIAL:
            tot += 1
            mf = max(cnt.items(), key=lambda kv: (kv[1], kv[0] == last))[0] if cnt else None
            hit["most-frequent"] += mf == o[t]; hit["last-object"] += last == o[t]
            g = o[t - 2 * k] if (prev_len > 0 and k <= prev_len and t - 2 * k >= 0) else None
            g = g if (g is not None and g >= N_SPECIAL) else mf
            hit["retrace"] += g == o[t]
        if o[t] >= N_SPECIAL:
            cnt[o[t]] += 1; last = o[t]
print(f"object revisit targets: {tot} over {len(toks)} sequences")
print(f"  uniform-16    {1 / 16:.4f}")
for kname in ("most-frequent", "last-object", "retrace"):
    print(f"  {kname:13s} {hit[kname] / tot:.4f}")
