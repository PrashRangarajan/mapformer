# Guards: the methodological lessons as tested code

Three modules plus a test file (and, since 2026-09-24, `stats_core` and the operational guards
in the last six rows). They exist because every retraction of 2026-08-26..09-11
came from skipping one of these checks, not from getting the arithmetic wrong.
Run the tests with `python3 -m mapformer.test_guards` from `/home/prashr`
(CPU only, about 2 s). Every test recomputes a number that is already committed.

| utility | lesson | learned in |
|---|---|---|
| `stats_guard.paired` / `from_diffs` / `table` | Pair by seed and report the MDE, `2.8*sd/sqrt(n)`. Say DETECTABLE only when \|delta\| > MDE, otherwise "unmeasured", never "null" | rule 11 |
| `stats_guard.interaction` | Interaction MDE `2.8*sqrt(v1/n1+v2/n2)`, or paired diff-of-diffs | LOOP_HEADROOM |
| `stats_guard.rule9` / `loss_matched` | Check r(loss, acc) first. Loss-match with one pooled fit. At \|r\|>0.98 the loss-matched contrast warns that it conditions on a mediator | rule 9, AUDIT finding 7 |
| `stats_guard.replication_split` | Show the pooled estimate, the first-k seeds and the fresh seeds together. Flags a sign flip, and flags a fresh estimate that falls under its own MDE | rule 6, D5 E4, AUDIT finding 3 |
| `ckpt_guard.compare_checkpoints` | Bitwise, NaN-aware check of weights and loss curves. An equal pair is labelled DETERMINISM, not replication. Handles both checkpoint formats | rule 27, AUDIT finding 10 |
| `ckpt_guard.require_checkpoints` | Fails loudly: shows tried paths, the files actually present, and which layout would have matched | eval_noise_refine dashes (x2) |
| `ckpt_guard.assert_frozen_unchanged` / `assert_moved` / `assert_zero` | Checks in code that an intervention did what it claims. `assert_moved` is the positive control | NOLEAK_PREREG, WARM |
| `ckpt_guard.assert_no_rng_consumed` | Checks torch (CPU and CUDA), numpy global, python `random`, and any generators you pass | v4 RNG control, unfreeze recording |
| `ckpt_guard.assert_same_function_at_init` | A new arm must equal its comparator at init | MAGONLY M4, NOLEAK |
| `probe_rewind` | Rewind slope, pooled and per (head, block). A_P selection over symbol keys. For warm starts, the full latent-pathway slope (both coordinates) | AUDIT finding 2, UNFREEZE correction |
| `stats_guard.Contrast.row_full` / `.p_t` / `.p_signflip` / `.mde_t` / `mde_t()` / `signflip_p()` + `TABLE_HEADER_FULL` | The house verdict \|d\| > 2.8 sd/sqrt(n) is \|t\| > 2.8: alpha 0.107 at n=3, 0.027 at n=8; the exact 80%-power multiplier is 5.36 / 3.26. These report the exact-t MDE, t-test p, sign-flip p and an n<6 / sd=0 note BESIDE the unchanged verdict | audit 2026-09-24 A1 |
| `stats_core` | The one implementation of: exact-t MDE (`mde`, `mde_multiplier`, `verdict_alpha`), `paired_p`, `signflip_p`, vectorised exact two-sample permutation `perm2_p` / `perm2_ci` (fixed grid reproduces the committed rank readouts; `grid=None` cannot clip), `fisher_solved`, Amendment-2 `classify_run` with RISING split out, `bimodality`, `report` | audit 2026-09-24 A1, B5 |
| `ckpt_guard.check_not_stale` | Refuses a train_hourglass checkpoint whose val_bpc disagrees with its run JSON (a duplicate launch's early `.best.pt`); warns if the JSON is missing. Called by the code readers | 2026-09-22 stale NoSigma s2 |
| `experiment_audit` | Exits 2 (not 0) on a layout it cannot read, after a convergence check on the checkpoints alone; prints the registered windowed classes beside the slope rule | audit 2026-09-24 A2 |
| `safe_clear.sh` | Refuses a run dir with any completion marker (repo marker naming it or a prefix of it, any `.*done*` inside) or a live python3 naming it. Fails CLOSED | rm of a finished batch; audit A3 |
| `lib_driver.sh` | Driver helpers: comm-matched trainer count, least-loaded picker (2 jobs/GPU), flock, md5 guard, setsid launch, wait-for-batch, marker only after the artifacts exist | rules 13, 20-24 |

## Usage

```python
from mapformer.stats_guard import load_arms, paired, loss_matched, replication_split, table
acc, loss = load_arms("_D5_N24.json")                 # relative -> REPO, not cwd
c = paired(acc["EMDoF_alignlock"], acc["VanillaEM_P0_r4"], "AlignLock - P0")
print(table([c]))
print(replication_split(c, first_k=8).report())       # FLAG sign flip: +0.120 -> -0.109
lm, r9 = loss_matched(acc, loss, "EMDoF_alignlock", "VanillaEM_P0_r4")
print(r9)                                             # r = -0.986 -> mediator WARNING

from mapformer.ckpt_guard import compare_checkpoints, require_checkpoints
print(compare_checkpoints(a_pt, b_pt).report())       # DETERMINISM / DIFFERENT
found = require_checkpoints("runs/dof/recency", "recency", ["VanillaEM_P0_r4"], range(24))

from mapformer.ckpt_guard import snapshot, assert_frozen_unchanged, assert_moved, assert_no_rng_consumed
before = snapshot(model); train_a_few_steps(model)
assert_frozen_unchanged(before, model, ["p0_pos", "action_to_lie.w_in.weight"])
assert_moved(before, model, ["layers.0.q_content.weight"])   # else the freeze check is vacuous
assert_no_rng_consumed(model._record, 0)
```

```python
from mapformer.stats_guard import paired, TABLE_HEADER_FULL
c = paired(acc_b, acc_a, "b - a")
print(TABLE_HEADER_FULL); print(c.row_full())   # house columns + exact-t MDE, t-test p, sign-flip p
from mapformer.stats_core import perm2_p, perm2_ci, classify_run, report
perm2_p(a, b)["p"]; perm2_ci(a, b, step=0.005)      # unclipped CI; grid=(-0.6, 0.6) = the registered one
classify_run(losses)["registered"]                  # SOLVED / STALLED / DESCENDING (cls adds RISING)
```

```bash
python3 -m mapformer.experiment_audit --runs-dir mapformer/runs/rank_matched_e900   # exit 2 + windowed classes
bash mapformer/safe_clear.sh mapformer/runs/<dir>                                  # refuses completed batches
python3 -m mapformer.probe_rewind --runs-dir runs/dof/recency --arms VanillaEM_P0_r4 --seeds 0-7
python3 -m mapformer.probe_rewind --ckpt runs/warm/EMWarm_freeze_s0/EMWarm_freeze_recency.pt
```

## What the tests pin

- **MagOnly:** `AlignFree - MagOnly` = +0.146, sd 0.150, MDE 0.086, 22/24. Fresh seeds give +0.113 (14/16). r = -0.978.
- **D5:** `AlignLock - P0` is +0.120 on seeds 0-7 and -0.109 on seeds 8-23, which is flagged as a sign flip. Loss-matched +0.001 and -0.009. r = -0.986.
- **AUDIT finding 3:** `sep - P0` is +0.073 on fresh seeds (unmeasured). Phase freedom is +0.173 on fresh seeds (replicates).
- **Determinism pairs:** `runs/dof/recency` vs `runs/recency_em` P0 s0 is bitwise identical, and so is the MagOnly repro. U1 holds on 27 shared tensors. NoLeak at init shows the NaN false alarm: `torch.equal` says unequal, the NaN-aware check says equal.
- **Rewind slopes:** all 56 slopes in `_REWIND_PROBE.json` are reproduced to 1.2e-7, covering all three origin forms. `EMWarm_freeze` s0 gives -1.000 with A_P selection 1.000. `VanillaEM_P0_r4` s0 gives -0.007.
- **UNFREEZE correction table:** effective -0.602 and latent pathway -0.866. The "coordinate 0" value -0.989 is the recorded `traj_lat` at the last epoch boundary; the endpoint recompute is -0.988.

Scope: `probe_rewind` covers MapFormer-EM on recency only. Its A_P selection readout uses 32 held-out episodes, so it differs from the unsaved sample behind `_REWIND_PROBE.json` column 2. The slope does not depend on sampling.
