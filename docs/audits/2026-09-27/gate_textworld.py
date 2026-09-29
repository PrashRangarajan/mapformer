"""Rule-11 gate for TEXTWORLD_PREREG.md, calling environment_textworld.TextWorld.
Held-out map (seed 10000), T tokens: vocabulary, steps per sequence, revisit rate at object slots,
and the floor at the scored slots: best constant, and an n-gram over preceding WORDS (orders 1-5)
fit on the training map (seed 0). A render sample is printed."""
from collections import Counter, defaultdict
import numpy as np
from mapformer.environment_textworld import TextWorld

T, NTR, NTE = 1024, 1500, 500
np.random.seed(1)
tr, te = TextWorld(seed=0), TextWorld(seed=10000)
print("vocab", tr.unified_vocab_size, tr.vocab)
t, m, r = te.generate_trajectory(T)
print("sample:", " ".join(te.vocab[i] for i in t[:60].tolist()))
TR = [tr.generate_trajectory(T) for _ in range(NTR)]; TE = [te.generate_trajectory(T) for _ in range(NTE)]
steps = np.mean([int(m.sum()) for _, m, _ in TE]); rev = np.mean([int(r.sum()) / max(1, int(m.sum())) for _, m, r in TE])
assert max(int(t.max()) for t, _, _ in TE) < te.unified_vocab_size
tgt = [(tok.tolist(), i) for tok, _, r in TE for i in np.nonzero(r.numpy())[0]]
ys = [tok[i] for tok, i in tgt]
const = Counter(ys).most_common(1)[0][1] / len(ys)
res = {}
for n in range(1, 6):
    tab = defaultdict(Counter)
    for tok, _, r in TR:
        tl = tok.tolist()
        for i in np.nonzero(r.numpy())[0]:
            tab[tuple(tl[i - n:i])][tl[i]] += 1
    res[n] = np.mean([(tab[tuple(tok[i - n:i])].most_common(1)[0][0] if tab[tuple(tok[i - n:i])] else ys[0]) == tok[i] for tok, i in tgt])
print(f"T={T}: object slots/seq {steps:.1f}, revisit fraction {rev:.3f}, scored targets {len(ys)}")
print(f"floor: const {const:.3f} | n-gram " + " ".join(f"{n}:{v:.3f}" for n, v in res.items()))
