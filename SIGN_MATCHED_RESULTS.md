# The sign ablation at matched length -- results (2026-09-28)

Pre-registration `SIGN_MATCHED_PREREG.md`; runs `runs/sign_matched`; full output
`SIGN_MATCHED_ANALYSIS.txt` (`python3 -m mapformer.analyze_sign_matched`); eval `SIGN_MATCHED.md` /
`.json`; strata `SIGN_MATCHED_STRATA.json`; probe `SIGN_MATCHED_PROBE.md` (registered, constrained
arms) and `SIGN_MATCHED_PROBE_SIGNED.md` (the signed arm, run afterwards as that probe's own reading
note requires). Torus, trained AND tested at T=1024, 900 epochs, 8 seeds, r=4 shared, one batch.

## Registered verdict: SIGN IS CAPABILITY

| arm | Delta | SOLVED | **T=1024 (training length)** | T=2048 |
|---|---|---|---|---|
| `Signed_r4` | `W_out W_in x` | **8/8** | **0.998 +/- 0.004** | 0.969 |
| `Abs_r4` | `\|W_out W_in x\|` (primary) | 0/8 | 0.821 +/- 0.120 | 0.725 |
| `Pos_r4` | `softplus(.)` (GRAPE-AP) | 1/8 | 0.781 +/- 0.193 | 0.693 |
| `RoPE` | token index | 0/8 | 0.731 +/- 0.008 | 0.552 |

| contrast at T=1024 | difference | exact permutation p | SOLVED, Fisher p |
|---|---|---|---|
| **Abs - Signed (primary)** | **-0.177** (MDE ~0.14) | **0.0002** | 0/8 vs 8/8, **0.0002** |
| Pos - Signed | -0.218 | 0.0003 | 1/8 vs 8/8, 0.0014 |
| Signed - RoPE (position effect, matched length) | +0.267 | 0.0002 | 8/8 vs 0/8, 0.0002 |

**The sign effect survives the matched-length control.** Trained at T=128 (`SIGN_ABLATION.md`) it
was -0.054 and UNMEASURED at the training length, -0.363 only when extrapolating to 1024; trained
AND tested at 1024 it is -0.177, 8/8 vs 0/8, in distribution. Of the robustness-not-capability
claims CLAUDE.md listed as never controlled (InEKF, forget gate, PoPE-wrapping, sign,
rotate/allocentric), sign is the first to get the control, and it passes.

**The mechanism is visible in the learned code, not only in accuracy.** Opposition
`||Delta(+x) + Delta(-x)|| / mean||Delta||` (0 = opposite actions cancel, 2 = identical): Signed
**0.062 / 0.056** (x / y), Abs 1.924 / 1.919, Pos 1.966 / 1.971. Constraint integrity: every Abs/Pos
Delta >= 0 after training (fraction negative 0.000). The signed arm uses its sign (half its entries
negative); the constrained arms cannot, and they fail.

## Caveats
- **Budget-scoped.** The monotone arms did not converge: Abs 5 STALLED + 3 DESCENDING, Pos 4 + 3 (+1
  SOLVED). Mostly flat, but the claim is "within 900 epochs at this recipe", not "cannot ever".
- **Accuracy is loss here** (r(final loss, acc) = -0.992 over 32 runs), as expected at matched length;
  the verdict is on the raw contrast as registered.
- Monotone still beats index (Abs - RoPE +0.090 at T=1024; unregistered, reported only): a monotone
  content-dependent phase carries SOME position information, just not net displacement.
- It remains a **replication in a new regime** (Sarrof et al. 2405.17394, Grazzi et al., Selective
  RoPE 2511.17388 on sign/negative eigenvalues), now at matched length on navigation.
- Scope: torus, T=1024, n_heads 2, d 128, r=4 shared, one recipe, 900 epochs, n=8. Pos_r4 carries
  the original's init confound.
