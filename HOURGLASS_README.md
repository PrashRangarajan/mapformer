# Hourglass-MapFormer + compositional experiments — setup & run

Self-contained instructions to reproduce the Hourglass line on any machine.
The Hourglass/compositional code itself uses only torch + numpy, but importing
the `mapformer` package runs its `__init__.py`, which pulls in the rest of
`requirements.txt` (matplotlib/scipy/scikit-learn via `evaluate.py`). So any
`python3 -m mapformer.X` invocation needs the full requirements installed — use
`pip install -r requirements.txt` below, not a torch+numpy subset.

## What this is

- **Hourglass scaffold** (`model_hourglass.py`): the Hourglass Transformer
  (Nawrot et al. 2021) used as a hierarchy, with MapFormer-WM layers swapped in.
  Verified faithful to the reference (lucidrains) implementation; causality is
  numerically checked.
- **Plain scaffold** (`hourglass_plain.py`): the same shorten/upsample ops with
  ordinary transformer layers, for a scaffold sanity-check on enwik8.
- **Compositional env** (`environment_compositional.py`): repeatable room
  MOTIFS at different locations, to test whether a motif-collapsing hierarchy
  earns its keep. See `COMPOSITIONAL_EXPERIMENT.md` for the full design.

## Setup

```bash
# 1. Get the code (repo folder MUST be named `mapformer`)
git clone git@github.com:PrashRangarajan/mapformer.git
cd mapformer

# 2. Deps. IMPORTANT: `pip install torch` may grab a wheel built for a newer
#    CUDA than the driver supports (e.g. torch+cu130 on a CUDA-12.4 driver),
#    which silently leaves torch.cuda.is_available()==False and runs on CPU.
#    Check the driver's CUDA version with `nvidia-smi` (top-right) and install a
#    matching wheel FIRST, then the rest of requirements.
pip install torch --index-url https://download.pytorch.org/whl/cu124   # e.g. for CUDA 12.x
pip install -r requirements.txt                                        # numpy, matplotlib, scipy, sklearn
# verify CUDA is actually live before spending GPU time:
python3 -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"

# 3. enwik8 data (only needed for the scaffold check; compositional is synthetic)
mkdir -p data && cd data
curl -O http://mattmahoney.net/dc/enwik8.zip
python3 -c "import zipfile; zipfile.ZipFile('enwik8.zip').extractall('.')"
cd ..
```

## IMPORTANT: how to invoke

Everything runs as a **module from the repo's PARENT directory**:

```bash
cd ..                 # now sit ONE LEVEL ABOVE the `mapformer/` folder
python3 -m mapformer.<script> ...
```

All default paths (enwik8 data, output dirs, result files) are derived from the
package location, so there is nothing to edit for a new machine.

## Verify the port (no GPU needed)

```bash
python3 -m mapformer.test_hourglass_causal      # scaffold: zero future-leak, param parity
python3 -m mapformer.validate_compositional     # task validity: labels, lags, baselines
```

## Run: enwik8 scaffold / efficiency check

Hourglass and Flat-10 have identical params; Hourglass runs its valley at
seq/shorten, so it is cheaper. The efficiency win grows with sequence length.

```bash
# long-sequence efficiency confirmation (attention dominates -> savings are real)
python3 -m mapformer.train_hourglass_enwik8 --model hourglass --shorten 4 \
    --seq-len 2048 --batch-size 6 --iters 8000 --device cuda:0
python3 -m mapformer.train_hourglass_enwik8 --model flat10 \
    --seq-len 2048 --batch-size 6 --iters 8000 --device cuda:0
# outputs: mapformer/hourglass_enwik8/{hourglass,flat10}.json  (val_bpc vs iter + wall)
```

Flags: `--model {hourglass,flat10,flat9}`, `--shorten`, `--seq-len`,
`--batch-size`, `--iters`, `--lr`, `--device`, `--out`, `--data`.

## Run: compositional Phase 1 + eval

```bash
for V in Vanilla VanillaEM Hourglass_k2 HourglassFlat3; do
  python3 -m mapformer.train_compositional --variant $V --target motif \
      --n-steps 256 --epochs 50 --n-batches 156 --n-layers 3 \
      --device cuda:0 --output-dir mapformer/runs/comp_phase1 --seed 0
done

python3 -m mapformer.eval_compositional \
    --checkpoints mapformer/runs/comp_phase1/{Vanilla,VanillaEM,Hourglass_k2,HourglassFlat3}.pt \
    --lengths 256 512 1024 2048 --n-traj 200 --device cuda:0
# outputs: mapformer/COMPOSITIONAL_RESULTS.md
```

`--target`: `motif` (compositional objective), `cross` (transfer-only),
`exact` (paper-standard fine target). Registered variants that path-integrate
work as `--variant` (Vanilla, VanillaEM, Hourglass_k2, Hourglass_k4,
Hourglass_k2_deep, HourglassFlat3, ...), plus two **non-MapFormer controls**
(ordinary index-RoPE transformers, no path integration): `PlainHourglass`,
`PlainFlat`.

Clear backbone-structure **aliases** (non-breaking — old keys still work):
`MapWM-Flat`=Vanilla, `MapEM-Flat`=VanillaEM, `MapWM-Hier`=Hourglass_k2,
`MapWM-FlatHG`=HourglassFlat3, `Plain-Hier`=PlainHourglass,
`Plain-Flat`=PlainFlat.

## Run: compositional multi-seed (the authoritative comparison)

```bash
# 6 variants x seeds {0,1,2}, split across both GPUs, then aggregate mean+-std.
# Resumable (skips existing checkpoints); writes COMPOSITIONAL_MULTISEED.md/.json.
nohup bash mapformer/run_comp_multiseed.sh >/dev/null 2>&1 &
tail -f mapformer/compositional_multiseed.log
```

To (re-)aggregate on-disk checkpoints without retraining (renders clean names):

```bash
python3 -m mapformer.agg_comp_multiseed \
    --runs-dir mapformer/runs/comp_multiseed --seeds 0 1 2 \
    --variants Vanilla VanillaEM Hourglass_k2 HourglassFlat3 PlainHourglass PlainFlat \
    --lengths 256 512 1024 2048 --n-traj 200 --batch 16 --device cuda:0
```

Headline (n=3): the **hierarchy** helps in BOTH the MapFormer and plain
families, but **MapFormer barely beats a plain transformer**; MapWM-Hier's lead
is high-variance (one lucky seed). Full reading in `COMPOSITIONAL_EXPERIMENT.md`.
For a long eval `--n-steps`, add `--batch 16` to `eval_compositional` to fit
T=2048 in one 24 GB GPU.

## Run: both, unattended (waits for idle GPUs)

```bash
nohup bash mapformer/run_hourglass_experiments.sh >/dev/null 2>&1 &
tail -f mapformer/hourglass_experiments.log
```

Waits until both GPUs have >20 GB free, then runs the long-seq enwik8 chain on
GPU0 and the compositional Phase-1 chain on GPU1, and writes the result files.
Touches `mapformer/.hourglass_experiments_done` when finished. Does not commit.

## File map

| File | Role |
|---|---|
| `model_hourglass.py` | Hourglass with MapFormer layers (+ registered variants) |
| `hourglass_plain.py` | plain-layer Hourglass + flat baselines (scaffold check) |
| `train_hourglass_enwik8.py` | enwik8 char-LM trainer (bpc) |
| `test_hourglass_causal.py` | causality + param-parity unit test |
| `environment_compositional.py` | repeatable-motif grid env |
| `validate_compositional.py` | task-validity gate (run before training) |
| `train_compositional.py` | compositional-task trainer |
| `eval_compositional.py` | exact vs cross-instance acc/NLL at length (`--batch` for long T) |
| `run_hourglass_experiments.sh` | GPU-waiting launcher for the single-seed Phase 1 + enwik8 |
| `run_comp_multiseed.sh` | multi-seed (6 variants x 3 seeds) 2-GPU launcher + aggregate |
| `agg_comp_multiseed.py` | multi-seed aggregator (mean±std, clean display names) |
| `COMPOSITIONAL_EXPERIMENT.md` | full experiment design + hypotheses + RESULTS |
| `COMPOSITIONAL_MULTISEED.md` | authoritative multi-seed result table |
