# Audits of 2026-09-24

Five read-only audits run while the rank continuation and stability batches trained, plus
the end-to-end review of the rank matched-length line. Nothing here is applied unless a
commit says so. Patches are against the tree as of commit ba42ec5.

| audit | report | what is in it |
|---|---|---|
| context / tokens | `context/REPORT.md` | CLAUDE.md classification; APPLIED in 670ad10 (183 KB -> 21 KB, log in `docs/LOG.md`) and the memory merge b277599 / c0b2d31 / ab05f91 |
| experiments | `experiments/LEDGER.md`, `PROPAGATION.md`, `EXPERIMENTS.md` | 58 claims by status; 44 places in 11 documents still citing retracted results; cheapest decisive experiments |
| theory / documents | `theory_REPORT.md` | withdrawn theory still asserted, over-claims, novelty claims the corpus contradicts, per-head vs shared bottleneck, minimal edits per document |
| efficiency | `efficiency/REPORT.md`, `efficiency/*.patch`, `efficiency/proposed/`, `efficiency/verify_*.py` + `.out` | vectorised walk (byte-identical, 23x), SDPA without TF32 (new series only), scale fold, sync-free loop, vectorised evaluators and permutation test; `verify_gpu_bitexact.py` NOT yet run |
| correctness / stats / hygiene | `hygiene/REPORT.md`, `hygiene/patches/01..13` | small-n MDE calibration, experiment_audit no-op, safe_clear fails open, stale checkpoint (quarantined), rank-pipeline latent risks, relative output paths, pgrep -f |
| rank matched-length review | `rank_review.md` | the review behind Amendment 2 |

Apply order (from the reports): after the running batches finish and their drivers exit --
env fast walk + permutation vectorisation (byte-identical), evaluators, then scale fold and
sync-free loop after the GPU bit-exact check, then SDPA for new series only. Hygiene
patches 03/05/06/09/11 touch files the running drivers use; apply after they exit.
