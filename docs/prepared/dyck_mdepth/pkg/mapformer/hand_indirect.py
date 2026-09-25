"""Existence check: can Indirect Indexing be solved by ARITHMETIC rather than lookup?

Trained models memorise it -- 31 separate lookups, one per shift, collapsing to
chance on any shift they were not shown (INDIRECT_OOD.md). Before building a model
that might learn it properly, construct a solution BY HAND and check it generalises.
This project's rule: existence before mechanism. A solution that exists need not be
found by training, but one that does not exist cannot be.

The construction is two attention hops with hand-set weights and NO per-shift
parameters at all -- every shift, seen or unseen, goes through the same operations:

  hop 1 (content)   the answer query attends to the string letter matching the
                    source letter; the VALUE returns that letter's POSITION as
                    features e(p) = (cos p*th_i, sin p*th_i)
  shift             each digit contributes an angle scaled by its place value, and
                    angles ADD: rotating by a then b equals rotating by a+b, so
                    "16" becomes 16*th exactly even if 16 never appeared
  hop 2 (position)  the query is e(p) rotated by the shift angle, i.e. e(p+k);
                    string keys are e(j); score(j) = sum_i cos((p+k-j) th_i),
                    which peaks exactly at j = p+k

Three ingredients, each ablated below to show what it buys:
  A. position readable as a VALUE        (standard RoPE and MapFormer: no -- position
                                           lives only in the query-key rotation)
  B. the 2nd hop's rotation set from what the 1st hop RETRIEVED
                                         (MapFormer: no -- its angle is fixed from
                                           token identity before any attention runs)
  C. the shift as ADDITIVE angles        (a learned model: whatever it finds)

Examples come from the task's own generator (rule 7), with the module's block
widened only so longer strings and larger shifts fit.
"""
import numpy as np
import torch
import mapformer.environment_indirect as E

TH = 10000.0 ** (-np.arange(32) / 32)          # RoPE-style frequency ladder, 32 blocks


def feats(pos):
    """e(p): absolute position features, (..., 64)."""
    a = np.multiply.outer(np.asarray(pos, float), TH)
    return np.concatenate([np.cos(a), np.sin(a)], -1)


def rotate(e, ang):
    """Rotate e(p) by angle ang per block -> e(p + ang/TH). Exact angle addition."""
    c, s = e[..., :32], e[..., 32:]
    ca, sa = np.cos(ang), np.sin(ang)
    return np.concatenate([c * ca - s * sa, s * ca + c * sa], -1)


def parse(row):
    """Split one task sequence into (string ids, source id, sign, digit ids)."""
    toks = [E.VOCAB[i] for i in row if i != 0]
    txt = "".join(toks)
    string, src, shift, _ = txt.split(", ")
    return [E.STOI[c] for c in string], E.STOI[src], (-1 if shift[0] == "-" else 1), [int(d) for d in shift[1:]]


def solve(row, *, pos_as_value=True, additive_digits=True, lookup=None, beta=50.0):
    s_ids, src, sign, digits = parse(row)
    L = len(s_ids)
    string_pos = np.arange(L)

    # ---- hop 1: content match against the source letter ----
    match = np.array([1.0 if t == src else 0.0 for t in s_ids])
    w1 = np.exp(beta * match); w1 /= w1.sum()
    if pos_as_value:
        e_p = w1 @ feats(string_pos)           # value = position features -> e(p)
    else:
        e_p = feats(0) * 0 + feats(L)          # position NOT carried back: the query
                                               # only knows where IT is (the end)

    # ---- the shift, as an angle ----
    if additive_digits:
        k = sign * sum(d * 10 ** m for m, d in enumerate(reversed(digits)))
        ang = k * TH                            # compositional: digits' angles add
    else:
        key = (sign, tuple(digits))
        if key not in lookup:                   # a lookup table: only shifts it holds
            return None
        ang = lookup[key]

    # ---- hop 2: attend by position to e(p) rotated by the shift ----
    q = rotate(e_p, ang)
    scores = feats(string_pos) @ q
    j = int(np.argmax(scores))
    return s_ids[j]


def run(world, n, seed, **kw):
    X, Y = world.sample(n, np.random.default_rng(seed))
    hit = tot = miss = 0
    for row, y in zip(X.numpy(), Y.numpy()):
        out = solve(row, **kw)
        if out is None:
            miss += 1; continue
        tot += 1; hit += int(out == y)
    return hit / max(tot, 1), tot, miss


def shift_of(row):
    _, _, sign, digits = parse(row)
    return sign * int("".join(map(str, digits)))


def main():
    E.BLOCK = 80                                # widen only so long strings / big shifts fit
    chance = 1 / 52
    print(f"chance = 1/52 = {chance:.3f}\n")

    trained = E.IndirectWorld(min_len=20, max_len=40, max_shift=15)
    extended = E.IndirectWorld(min_len=40, max_len=52, max_shift=30)

    print("THE CONSTRUCTION -- no per-shift parameters anywhere")
    a, n, _ = run(trained, 4000, 1)
    print(f"  shifts +/-15, strings 20-40 (the training distribution)  acc {a:.4f}  (n={n})")
    X, Y = extended.sample(4000, np.random.default_rng(2))
    big = [(r, y) for r, y in zip(X.numpy(), Y.numpy()) if abs(shift_of(r)) > 15]
    acc = np.mean([solve(r) == y for r, y in big])
    print(f"  shifts |k| 16-30, never 'seen', strings 40-52          acc {acc:.4f}  (n={len(big)})")

    print("\nABLATIONS -- which ingredient buys what")
    a, n, _ = run(trained, 4000, 1, pos_as_value=False)
    print(f"  A removed: hop 1 returns no position (standard RoPE / MapFormer)  acc {a:.4f}")

    # C replaced by a lookup table built only from EVEN shifts
    lut = {}
    for k in range(-14, 15, 2):                # EVEN shifts -14..14 (range(-15,16,2) is ODD)
        sign = -1 if k < 0 else 1
        lut[(sign, tuple(int(d) for d in str(abs(k))))] = k * TH
    Xe, Ye = trained.sample(6000, np.random.default_rng(3))
    ev = [(r, y) for r, y in zip(Xe.numpy(), Ye.numpy()) if shift_of(r) % 2 == 0]
    od = [(r, y) for r, y in zip(Xe.numpy(), Ye.numpy()) if shift_of(r) % 2 != 0]
    ev_acc = np.mean([solve(r, additive_digits=False, lookup=lut) == y for r, y in ev])
    od_out = [solve(r, additive_digits=False, lookup=lut) for r, y in od]
    od_acc = np.mean([(o == y) if o is not None else False for o, (r, y) in zip(od_out, od)])
    print(f"  C replaced by a lookup table of EVEN shifts: even {ev_acc:.4f}  |  odd {od_acc:.4f}  "
          f"<- memorisation's signature")
    ev_add = np.mean([solve(r) == y for r, y in ev]); od_add = np.mean([solve(r) == y for r, y in od])
    print(f"  C kept (additive digit angles), same split: even {ev_add:.4f}  |  odd {od_add:.4f}")

    # B: the best any SINGLE position kernel anchored at the query could do
    print("\nB -- MapFormer's placement: the query's position is fixed by the prefix")
    X, Y = trained.sample(4000, np.random.default_rng(4))
    offs = []
    for row in X.numpy():
        s_ids, src, sign, digits = parse(row)
        L = len(s_ids); p = s_ids.index(src); k = shift_of(row)
        offs.append(L - 1 - (p + k))            # string tokens between target and end
    offs = np.array(offs)
    best = np.bincount(offs).max() / len(offs)
    print(f"  distance from the end of the string to the target varies {offs.min()}-{offs.max()} "
          f"(sd {offs.std():.1f} positions)")
    print(f"  so ONE offset from the query can be right at most {best:.3f} of the time "
          f"-- even with a perfect clock")


if __name__ == "__main__":
    main()
