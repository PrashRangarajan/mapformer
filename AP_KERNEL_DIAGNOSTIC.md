# A_P kernel geometry at zero displacement

Measured on same-cell revisit pairs. A well-formed position kernel is positive and maximal at zero displacement.

| model | same-cell A_P | % negative | % row-max | n pairs |
|---|---|---|---|---|
| EM-sep vocab16 [EM WINS] | -0.0884 | 100.0% | 0.0% | 1159 |
| EM-p0  vocab16 [EM WINS] | +0.5617 | 0.0% | 38.5% | 1071 |
| EM-sep vocab256 [EM LOSES] | -0.0424 | 46.4% | 1.6% | 1251 |
| EM-p0  vocab256 s0 [good 0.910] | +0.6543 | 0.0% | 1.4% | 1294 |
| EM-p0  vocab256 s1 [failed 0.502] | +0.3443 | 27.0% | 6.5% | 1162 |
| EM-sep paper-task | -0.1677 | 77.3% | 1.5% | 1193 |
| EM-p0  paper-task | +0.3974 | 0.0% | 45.5% | 1218 |

## Verdict: the parameterization hypothesis is FALSIFIED

The script pre-registered the falsification condition: "If kernel quality is
equally bad in regimes where EM WINS (n_obs=16), the parameterization story is
wrong." The result is worse than that -- kernel quality is *inversely* related
to accuracy across configs:

| config | same-cell A_P | % negative | EM vs WM (T=512) |
|---|---|---|---|
| EM-sep, n_obs=16 | -0.0884 | **100.0%** | EM **WINS** +0.027 (3/3 seeds) |
| EM-sep, n_obs=256 | -0.0424 | 46.4% | EM **LOSES** -0.086 |

The configuration with the *worst possible* kernel -- negative at zero
displacement on 100% of revisit pairs -- is the one where EM beats WM on every
seed. A_P kernel geometry does NOT explain the vocab-specific deficit.

### What the kernel measurement DOES show

It cleanly separates the two parameterizations, consistently and in every
regime: separate q0/k0 is negative at zero displacement (46-100% of pairs,
mean -0.04 to -0.17); single-p_0 is positive on 100% of pairs (mean +0.34 to
+0.65), as the autocorrelation form guarantees. Within single-p_0 it also
tracks the seed collapse (good seed 0% negative, failed seed 27%).

So the geometric claim is true and measurable -- it just is not what drives
accuracy.

### Correction to earlier reasoning

I argued that "a position kernel should be maximal at zero displacement" was
checkable mathematics rather than interpretation, and used it to call the
separate-q0/k0 form defective. The mathematics is right for single-p_0 and the
measurement confirms it, but the inference to "therefore it performs worse" was
unjustified: models whose kernel is negative at zero displacement on 100% of
revisit pairs still beat WM on 3/3 seeds.

A_P does not need to be a "same place" detector. Inside softmax(A_X o A_P) it
only needs to be a CONSISTENT, discriminative function of displacement -- an
inverted but reliable kernel carries the same information. The paper's stated
suspicion that separation "would create sparser attention values" is not
refuted by kernel geometry, and our earlier dismissal of it went beyond the
evidence.
