"""95% permutation-inversion CI for MAPPOPE_PAIR's rank-2 T=128 contrasts (descriptive, after the registered verdict).
Output: mappope_pair_ci_out.txt."""
import json, numpy as np
from mapformer.stats_core import perm2_ci
J = json.load(open("/home/prashr/mapformer/MAPPOPE_PAIR_R2.json"))
a = {k: np.array([x[1] for x in sorted(J[f"0.0|{k}|128"])]) for k in ("Vanilla", "MapPoPE-Pair", "MapPoPE-Flat")}
for n, x, y in (("SCORE", "Vanilla", "MapPoPE-Pair"), ("COUNT", "MapPoPE-Pair", "MapPoPE-Flat"), ("TOTAL", "Vanilla", "MapPoPE-Flat")):
    ci = perm2_ci(a[x], a[y], step=0.0001, n_mc=20000)
    print(f"{n}: d {a[y].mean() - a[x].mean():+.4f}, 95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]")
