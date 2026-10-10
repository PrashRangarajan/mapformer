"""TinyStories word-level gate (rule 11, adapted to language modelling; CPU). (1) Decode round trip: the first val
story rebuilt from token ids equals the tokenizer applied to the raw text; every id < vocab. (2) Floors in nats/token
on the val stream: unigram, and interpolated bigram / trigram (Jelinek-Mercer, weights chosen on a held-out train slice,
counts from the first 100M train tokens). The LM pilot is read beside the best of these."""
import json, numpy as np
from mapformer.tinystories_data import tokenize, VALID, SEP

D = "/home/prashr/mapformer/data/tinystories"
itos = json.load(open(f"{D}/vocab.json")); V = len(itos); idx = {w: i for i, w in enumerate(itos)}
tr = np.memmap(f"{D}/train.bin", dtype=np.uint16, mode="r"); va = np.array(np.memmap(f"{D}/val.bin", dtype=np.uint16, mode="r"), dtype=np.int64)
assert tr[:50_000_000].max() < V and va.max() < V

raw = open(VALID, encoding="utf-8").read(200_000).split(SEP)[0]
ref = [idx.get(t, idx["<unk>"]) for t in tokenize(raw)]
print("round trip first val story:", "PASS" if list(va[: len(ref)]) == ref else "FAIL", f"({len(ref)} tokens)")
print("decoded:", " ".join(itos[i] for i in va[:40]).replace("\n", "\\n"))

N = 100_000_000
t = np.array(tr[:N], dtype=np.int64); h = np.array(tr[N:N + 2_000_000], dtype=np.int64)
uni = np.bincount(t, minlength=V) + 0.5; puni = uni / uni.sum()

def table(keys):
    u, c = np.unique(keys, return_counts=True); return u, c
def lookup(u, c, q):
    i = np.searchsorted(u, q); i = np.minimum(i, len(u) - 1); return np.where(u[i] == q, c[i], 0)

bk, bc = table(t[:-1] * V + t[1:]); ck, cc = table(t[:-2] * V * V + t[1:-1] * V + t[2:])
ctx1 = np.bincount(t[:-1], minlength=V); c2k, c2c = table(t[:-1] * V + t[1:])     # bigram counts = trigram contexts

def probs(s):
    a, b, c = s[:-2], s[1:-1], s[2:]
    p1 = puni[c]
    p2 = lookup(bk, bc, b * V + c) / np.maximum(ctx1[b], 1)
    n3 = lookup(c2k, c2c, a * V + b)
    p3 = lookup(ck, cc, a * V * V + b * V + c) / np.maximum(n3, 1)
    return p1, p2, p3, ctx1[b] > 0, n3 > 0

def nll(s, l2, l3):
    p1, p2, p3, h2, h3 = probs(s)
    w3 = np.where(h3, l3, 0.0); w2 = np.where(h2, l2, 0.0) * (1 - w3)
    return float(-np.log(w3 * p3 + w2 * p2 + (1 - w3 - w2) * p1).mean())

best2 = min((nll(h, l2, 0.0), l2) for l2 in np.linspace(0.5, 0.98, 13))
best3 = min((nll(h, l2, l3), l2, l3) for l2 in (0.7, 0.8, 0.9, 0.95) for l3 in np.linspace(0.3, 0.9, 13))
print(f"val floors (nats/token): unigram {float(-np.log(puni[va]).mean()):.4f} | bigram {nll(va, best2[1], 0):.4f} "
      f"(l2 {best2[1]:.2f}) | trigram {nll(va, best3[1], best3[2]):.4f} (l2 {best3[1]:.2f}, l3 {best3[2]:.2f})")
print(f"val: {len(va):,} tokens; <unk> {float((va == idx['<unk>']).mean()):.5f}; vocab {V}")
