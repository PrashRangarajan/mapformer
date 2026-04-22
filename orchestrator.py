#!/usr/bin/env python3
"""
Orchestrator for the full paper-ready experiment suite.

Launches all training runs across:
  - Multi-seed main variants
  - Level 1.5 ablations
  - RoPE baseline
on available GPUs, writes results as they finish. After all trainings,
runs evaluation and commits results.

Run in background:
  cd /home/prashr
  nohup python3 -u -m mapformer.orchestrator > mapformer/orchestrator.log 2>&1 &
"""

import subprocess
import time
import os
import json
from pathlib import Path
from datetime import datetime


REPO = Path("/home/prashr/mapformer")
RUNS_DIR = REPO / "runs"
RUNS_DIR.mkdir(exist_ok=True)

N_GPUS = 2
GPU_STATE = [None, None]  # per-GPU currently-running Popen, or None

EXPERIMENTS = []

# Main variants, 3 seeds × 2 configs
MAIN_VARIANTS = ["Vanilla", "Level1", "Level15", "PC", "RoPE"]
CONFIGS = [
    {"tag": "clean",    "p_action_noise": 0.0,  "n_landmarks": 0},
    {"tag": "noise",    "p_action_noise": 0.10, "n_landmarks": 0},
    {"tag": "lm200",    "p_action_noise": 0.10, "n_landmarks": 200},
]
SEEDS = [0, 1, 2]

for variant in MAIN_VARIANTS:
    for cfg in CONFIGS:
        for seed in SEEDS:
            tag = cfg["tag"]
            out_dir = RUNS_DIR / f"{variant}_{tag}" / f"seed{seed}"
            EXPERIMENTS.append({
                "variant": variant,
                "seed": seed,
                "p_action_noise": cfg["p_action_noise"],
                "n_landmarks": cfg["n_landmarks"],
                "output_dir": str(out_dir),
                "ckpt_path": out_dir / f"{variant}.pt",
                "n_epochs": 50,
                "n_batches": 156,
                "tag": tag,
            })

# Level 1.5 ablations, seed=0 only, 2 configs (clean + lm200)
ABLATIONS = ["L15_ConstR", "L15_NoMeas", "L15_NoCorr", "L15_DARE"]
for abl in ABLATIONS:
    for cfg in [CONFIGS[0], CONFIGS[2]]:  # clean and lm200 only
        tag = cfg["tag"]
        out_dir = RUNS_DIR / f"{abl}_{tag}" / f"seed0"
        EXPERIMENTS.append({
            "variant": abl,
            "seed": 0,
            "p_action_noise": cfg["p_action_noise"],
            "n_landmarks": cfg["n_landmarks"],
            "output_dir": str(out_dir),
            "ckpt_path": out_dir / f"{abl}.pt",
            "n_epochs": 50,
            "n_batches": 156,
            "tag": tag,
        })

# Skip experiments whose checkpoint already exists
REMAINING = [e for e in EXPERIMENTS if not e["ckpt_path"].exists()]

print(f"[{datetime.now()}] Orchestrator starting")
print(f"  Total experiments: {len(EXPERIMENTS)}")
print(f"  Already complete:  {len(EXPERIMENTS) - len(REMAINING)}")
print(f"  To run:            {len(REMAINING)}")
print(f"  Expected walltime: ~{len(REMAINING) * 13 / N_GPUS / 60:.1f} hours")
print("")


def launch_experiment(gpu_id, exp):
    """Launch training for one experiment on the given GPU."""
    out_dir = Path(exp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python3", "-u", "-m", "mapformer.train_variant",
        "--variant", exp["variant"],
        "--seed", str(exp["seed"]),
        "--n-landmarks", str(exp["n_landmarks"]),
        "--p-action-noise", str(exp["p_action_noise"]),
        "--epochs", str(exp["n_epochs"]),
        "--n-batches", str(exp["n_batches"]),
        "--device", "cuda",
        "--output-dir", exp["output_dir"],
    ]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd="/home/prashr", env=env, stdout=log_f, stderr=subprocess.STDOUT
    )
    exp["proc"] = proc
    exp["log_f"] = log_f
    exp["start_time"] = time.time()
    print(f"[{datetime.now()}] GPU{gpu_id}: launched {exp['variant']} "
          f"tag={exp['tag']} seed={exp['seed']} (pid {proc.pid})")


def experiment_done(exp):
    if "proc" not in exp: return False
    return exp["proc"].poll() is not None


# Orchestration loop
queue = list(REMAINING)
running = []

while queue or running:
    # Poll running experiments
    still_running = []
    for exp in running:
        if experiment_done(exp):
            exp["log_f"].close()
            retcode = exp["proc"].returncode
            ok = exp["ckpt_path"].exists()
            status = "OK" if (retcode == 0 and ok) else f"FAIL(rc={retcode},ckpt={'exists' if ok else 'missing'})"
            dt = time.time() - exp["start_time"]
            print(f"[{datetime.now()}] GPU{exp['gpu_id']}: {exp['variant']} "
                  f"tag={exp['tag']} seed={exp['seed']} -> {status} ({dt/60:.1f} min)")
            GPU_STATE[exp["gpu_id"]] = None
        else:
            still_running.append(exp)
    running = still_running

    # Launch on free GPUs
    for gpu in range(N_GPUS):
        if GPU_STATE[gpu] is None and queue:
            exp = queue.pop(0)
            exp["gpu_id"] = gpu
            GPU_STATE[gpu] = exp
            launch_experiment(gpu, exp)
            running.append(exp)

    time.sleep(10)

print(f"[{datetime.now()}] All training complete.")

# Final evaluation + commit handled by followup script
os.chdir(REPO)
subprocess.run(["bash", "orchestrator_finalize.sh"], cwd=REPO)
