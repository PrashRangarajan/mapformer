# DYCK_DECAY_PREREG -- the test that separates a reliability prior from a state variable

`DECAY_RESULTS.md` showed a 48-parameter decay envelope repairing PoPE's length extrapolation on Bach
as well as a 786k per-token phase does, on BOTH the index and the path-integrated row. But the learned
rates say what it bought: every head kept its ALiBi decay (lambda 0.498 .. 0.0045, horizons 2 to 222
tokens on pieces up to 2048). **The winning model on Bach is a local model**, so on that task
"repairing the collapse" and "becoming local" are the same operation, and the result is partly a fact
about chorales rather than about positional encoding.

Dyck-2 separates them, because its dependencies are explicitly NOT local: the bracket to be closed can
sit arbitrarily far back, and 6% of scored positions have it beyond 32 tokens. A reliability prior
that suppresses far pairs must damage exactly those.

## Arms and recipe

`DYCK_PREREG.md` verbatim (1 layer, 1 head, 560k sequences at L=32 D=4, 8 seeds), adding
`MapPoPE_decay` and `PoPE_decay`. Baselines in hand: MapPoPE-1L 0.927 at L128 D12 with closer accuracy
0.730 at distance 33+; PoPE-1L 0.615 and 0.569; the no-stack n-gram floor 0.884 / 0.511.

## Registered verdicts

- **E1 (the prediction)** `MapPoPE_decay` closer accuracy at **distance 33+** is BELOW MapPoPE's 0.730
  and detectably so. This is the claim: a decay envelope destroys long-range stack tracking.
- **E2** The damage is graded in distance -- little or nothing at 1-2, most at 33+. A flat loss across
  distances would mean the envelope hurt the model generally rather than by suppressing far pairs.
- **E3** `PoPE_decay` at distance 33+ against PoPE's 0.569: predicted below, but the floor is 0.5 and
  PoPE is already near it, so this arm has little room and is reported as a secondary check.
- **E4** F1 at the training cell (L32 D4): predicted approximately unchanged for both. Dyck's training
  cell is short enough that a 222-token horizon costs nothing there; a loss HERE would mean the
  envelope is damaging the model rather than its reach.

**Falsification**: if the envelope leaves distance-33+ accuracy intact, then a decay prior is
compatible with stack tracking, "decay and path integration do different jobs" is wrong as stated, and
the Bach result generalises further than I claimed.
