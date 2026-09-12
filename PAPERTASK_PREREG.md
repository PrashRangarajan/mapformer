# PAPERTASK RERUN -- does EM's length advantage survive a converged recipe?

Pre-registered 2026-09-12, before any arm was trained. `EM_WM_THEORY.md` P6.

## Why

The one EM-beats-WM cell on a MAP task comes from a batch (`run_seed_scaleup.sh`) that ran
**16 epochs on the LinearLR default** -- the recipe standing rule 10 exists to warn about -- and
whose 48 training logs and checkpoints have since been **deleted**, so rule 9 cannot be applied
to it at all. Its numbers reproduce from the committed per-seed json, and floor-normalising
them (floors now measured, `PAPER_TASK_FLOORS.md`) makes the gap LARGER: EM - WM = +0.174 /
+0.352 / **+0.430** at l=512 / 1024 / 2048, all detectable, 8/8 at both extended lengths. But
"converged" was never established: WM's IID there is 0.969 +/- 0.037 with a worst seed of 0.898,
against the paper's 0.99.

So the effect is either real and large, or a convergence artifact -- and the batch that would
tell us was thrown away. This re-runs it properly.

## Arms and recipe

`Vanilla` (WM), `VanillaEM_P0`, `MapPoPE-Flat`, **8 seeds each, ONE batch, all trained fresh**
into `runs/paper_task_rerun/` (nothing reused; the old directory no longer exists).

    train_variant --epochs 50 --schedule cosine --n-batches 98 --batch-size 128 --n-steps 128
                  --n-layers 1 --n-heads 2 --d-model 128 --n-landmarks 0

Two deliberate changes from the original: **cosine (5% warmup, decay to 10%) instead of the
LinearLR default**, and **50 epochs instead of 16**. **Training logs are kept** -- that is the
point of the re-run as much as the recipe is.

Evaluation: `eval_paper_ood --extended --n-batches 8 --batch-size 32` (the protocol the original
used), into `PAPER_OOD_RERUN.md/.json`.

## Readouts

**Primary: floor-normalised accuracy `(acc - floor) / (1 - floor)` at ext-s l=2048**, paired by
seed, floors from `PAPER_TASK_FLOORS.md` (measured with no model, on exactly the events
`revisit_accuracy` scores): 0.522 IID, 0.216 OOD-d, 0.799 / 0.801 / 0.802 at l=512 / 1024 / 2048.
Raw accuracy is reported beside it and is NOT the primary, because at pe=0.8 four fifths of the
scale is floor.

Secondary: l=1024 and l=512, IID and OOD-d, raw and normalised; final training loss per run.

## Convergence check, before any verdict is read

Every arm must reach **IID >= 0.99** (the paper's own level; the 16-epoch batch gave WM 0.969).
If WM does not, the recipe still has not converged and P1/P2 are not interpreted -- the run
becomes a budget-curve datapoint and nothing more.

## Predictions

- **P1 (the effect is real).** EM - WM floor-normalised at l=2048 is **>= +0.20 and DETECTABLE**
  (it is +0.430 at 16 epochs). CONFIRMED if both hold.
- **P2 (not a convergence artifact).** With logs kept, `stats_guard.rule9` over the 24 runs, then
  the loss-matched EM - WM residual at l=2048: predicted still DETECTABLE. This is the check the
  deleted logs made impossible. If r(loss, acc) exceeds 0.98 in magnitude, the loss-matched
  reading is reported with the mediator warning (`AUDIT_2026-09-10.md` #7).
- **P3 (the per-pair counterexample).** MapPoPE - EM floor-normalised at l=2048 stays
  **UNMEASURED** (it is +0.159, 7/8, inside MDE at 16 epochs). If it becomes DETECTABLE, the
  best arm on this task is a per-pair kernel by the repo's own standard, and `EM_WM_THEORY.md`
  2a's second objection is upgraded from directional to established.
- **P4 (the falsifier).** If WM reaches IID >= 0.99 AND the l=2048 gap falls below its MDE, the
  extended-length EM advantage was a budget artifact. Then 2a's demotion becomes a retraction,
  and the "EM wins at length" line comes out of every summary file.

## Analysis discipline

`stats_guard.paired` with an MDE beside every contrast; floors quoted beside every raw number
(rule 4); `stats_guard.rule9` before any loss-matched reading; all three arms in one batch
(rule 3); logs retained so the batch can be re-analysed without re-running it.
