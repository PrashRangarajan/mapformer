# MapFormer: path-integrated rotary position, and what decides whether training finds the map

A study that began as a faithful reproduction of Rambaud et al., *MapFormer* (arXiv:2511.19279): a transformer whose
rotary angle is a path-integrated, content-dependent phase, `theta = omega * cumsum(W_out W_in x_t)`, instead of the
token index. It is now a study of positional encoding as the mechanism by which a model learns a relational "where"
kept separate from the "what", measured on navigation (torus walks, also told in English, from pixel observations, and
in Memory Maze), algorithmic tasks and sequences. Every navigation evaluation redraws the observation map, so every
navigation number is a transfer measurement.

## Read in this order

1. [`CLAUDE.md`](CLAUDE.md) -- the authority: conventions, the citable results with numbers and files, the withdrawn
   claims, the standing rules (each bought by a failure) and the paper-faithful invariants.
2. [`docs/WHERE_THINGS_STAND.md`](docs/WHERE_THINGS_STAND.md) -- one-page orientation: thesis, what survives (with
   status), what is open, what is stale.
3. [`.claude-memory/project_state.md`](.claude-memory/project_state.md) -- live state: what is running, the last results,
   decisions pending.
4. The write-up: main paper draft 2, [`docs/paper/MAIN_DRAFT2.md`](docs/paper/MAIN_DRAFT2.md) +
   [`MAIN_APPENDIX2.md`](docs/paper/MAIN_APPENDIX2.md), and the sign companion, draft 6,
   [`docs/paper/DRAFT6_workshop.md`](docs/paper/DRAFT6_workshop.md) + [`APPENDIX_draft6.md`](docs/paper/APPENDIX_draft6.md);
   PDFs in [`docs/paper/pdf/`](docs/paper/pdf/); claim map
   [`docs/paper/CLAIM_MAP_clock_map.md`](docs/paper/CLAIM_MAP_clock_map.md).
5. [`RESULTS_INDEX.md`](RESULTS_INDEX.md) -- catalogue of every results file by line of work, with status markers.

## In one paragraph

On navigation whose revisits cannot be served by a local rule (1024-move torus walks), one attention layer whose rotary
phase path-integrates a learned step learns the map (0.998, 8/8 seeds) that index RoPE does not (0.731, 0/8), not at
four layers, not looped, and not as an LSTM / GRU; the same holds from pixel observations and when the walk is told in
English. Whether training finds that map is decided by two choices of the step map: the sign of the increment (`|.|`
and softplus steps represent the map exactly yet fail to learn it) and the per-head rank of the content-to-angle map
(rank D+1, with weight decay, where rank D fails because of a drifting clock component). Both are costs of search, not
of representation. Many earlier effects were robustness past the training length, not capability, and closed under
matched-length controls; the April-August InEKF / Level 1.5 extensions ended as negatives. Numbers, scope and status:
`CLAUDE.md` and `docs/WHERE_THINGS_STAND.md`.

## Quick start

```bash
pip install -r requirements.txt            # check torch.cuda.is_available(): pip can fall back to a CPU wheel
cd /home/prashr                            # modules run as python3 -m mapformer.X from the parent directory
python3 -m mapformer.test_guards           # guard tests (CPU, seconds)
python3 -m mapformer.main --device cuda --epochs 16 --n-batches 98   # reproduce MapFormer v4 Table 2 (2D torus, T=128)
```

Each registered batch has a pre-registration (`<NAME>_PREREG.md`), a driver (`run_<name>.sh`), a trainer, an analysis
(`analyze_<name>.py`, most with `--selftest`) and a results file (`<NAME>_RESULTS.md`). Model variants are in
`train_variant.py::VARIANT_MAP` (MapWM-Flat = `Vanilla`, MapEM-Flat = `VanillaEM`; `RoPE` is the index baseline).
An archived batch is reproducible from its pre-registration commit, not from HEAD.

## Layout

| path | what |
|---|---|
| `*.py`, `*.sh` (top level) | the package (`mapformer`): environments, models, trainers, drivers, analyses |
| `*_PREREG.md`, `*_RESULTS.md`, `*.json` (top level) | pre-registrations, results, per-arm outputs (scripts read and write them here by path) |
| `docs/paper/` | drafts 2 and 6, appendices, figures (`figs/`), PDFs (`pdf/`), superseded drafts (`old/`) |
| `docs/audits/<date>/` | audits, verification agents' reports and their scripts (cited by pre-registrations; do not move) |
| `docs/lit/`, `papers/` | literature reviews and the paper corpus (`papers/INDEX.md`) |
| `runs/` | run directories (checkpoints gitignored; per-run JSON, logs and md5 records) |
| `archive/` | withdrawn, superseded, orphaned and Level 1.5-era results, dead code; `archive/MOVED.md` maps old paths to new |
| `report/`, `*.tex` | the September documents (stale; see `docs/WHERE_THINGS_STAND.md`) and the shared report source |

## Conventions

Single-author commits; no emojis; honest reporting (failures with their reason, "unmeasured" below the MDE, the measured
floor beside every headline). Details and the standing rules: `CLAUDE.md`. Known code bugs: `KNOWN_BUGS.md`; guards:
`GUARDS.md`; the Hourglass hierarchy code: `HOURGLASS_README.md`.

## Reference

Rambaud, Mascarenhas and Lakretz, *MapFormers: Self-Supervised Learning of Cognitive Maps with Input-Dependent
Positional Encoding*, arXiv:2511.19279 (reproduction target: v4, 10 May 2026; text in `papers/txt/mapformer.txt`).
