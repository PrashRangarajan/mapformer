"""Floors for TW_NORMSTEP_PREREG.md on its eval stream (held-out map 10000, np seed 10**6, 200 trials, T=1024):
analyze_textworld_secondary.floors with the seed changed (best constant; reversal-copy rule)."""
import numpy as np
import mapformer.analyze_textworld_secondary as S

_seed = np.random.seed
np.random.seed = lambda s: _seed(10**6 if s == 0 else s)     # floors() seeds with 0; redirect to the eval seed
const, rcopy, n = S.floors()
np.random.seed = _seed
print(f"eval seed 10**6: best constant {const:.4f}; reversal-copy {rcopy:.4f}; {n} targets")
