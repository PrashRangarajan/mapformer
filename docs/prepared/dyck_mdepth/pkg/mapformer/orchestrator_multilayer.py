#!/usr/bin/env python3
"""Multi-layer scaling study for Level 1.5.

Paper only tested 1 layer. We test 1, 2, 4 layers for both Vanilla and
Level 1.5 to see if the gain scales. Kicked off separately once the main
orchestrator has freed up compute.

Usage:
  nohup python3 -u -m mapformer.orchestrator_multilayer \
      > mapformer/orchestrator_multilayer.log 2>&1 &
"""

import subprocess
import time
import os
from pathlib import Path
from datetime import datetime


REPO = Path("/home/prashr/mapformer")
RUNS_DIR = REPO / "runs"

N_GPUS = 2
GPU_STATE = [None, None]

# Multi-layer variants
# 1 layer already done in main orchestrator, so just 2 and 4
EXPERIMENTS = []
for variant in ["Vanilla", "Level15"]:
    for n_layers in [2, 4]:
        for config in [("clean", 0), ("lm200", 200)]:
            tag, n_lm = config
            for seed in [0]:  # single seed at multi-layer (cost)
                out_dir = RUNS_DIR / f"{variant}_L{n_layers}_{tag}" / f"seed{seed}"
                EXPERIMENTS.append({
                    "variant": variant,
                    "seed": seed,
                    "n_landmarks": n_lm,
                    "p_action_noise": 0.10 if tag != "clean" else 0.0,
                    "n_layers": n_layers,
                    "output_dir": str(out_dir),
                    "ckpt_path": out_dir / f"{variant}.pt",
                    "n_epochs": 50,
                    "n_batches": 156,
                    "tag": tag,
                })

REMAINING = [e for e in EXPERIMENTS if not e["ckpt_path"].exists()]

print(f"[{datetime.now()}] Multi-layer orchestrator starting")
print(f"  Total experiments: {len(EXPERIMENTS)}")
print(f"  Already complete:  {len(EXPERIMENTS) - len(REMAINING)}")
print(f"  To run:            {len(REMAINING)}")
print(f"  Expected walltime: ~{len(REMAINING) * 15 / N_GPUS / 60:.1f} hours")


def launch_experiment(gpu_id, exp):
    out_dir = Path(exp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # We need to pass --n-layers, so modify train_variant to accept it
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
        "--n-layers", str(exp["n_layers"]),
    ]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd="/home/prashr", env=env, stdout=log_f, stderr=subprocess.STDOUT)
    exp["proc"] = proc; exp["log_f"] = log_f; exp["start_time"] = time.time()
    print(f"[{datetime.now()}] GPU{gpu_id}: {exp['variant']} L={exp['n_layers']} "
          f"{exp['tag']} seed={exp['seed']} (pid {proc.pid})")


def experiment_done(exp):
    if "proc" not in exp: return False
    return exp["proc"].poll() is not None


queue = list(REMAINING); running = []

while queue or running:
    still_running = []
    for exp in running:
        if experiment_done(exp):
            exp["log_f"].close()
            retcode = exp["proc"].returncode
            ok = exp["ckpt_path"].exists()
            status = "OK" if (retcode == 0 and ok) else f"FAIL(rc={retcode})"
            dt = time.time() - exp["start_time"]
            print(f"[{datetime.now()}] GPU{exp['gpu_id']}: {exp['variant']} "
                  f"L={exp['n_layers']} {exp['tag']} -> {status} ({dt/60:.1f} min)")
            GPU_STATE[exp["gpu_id"]] = None
        else:
            still_running.append(exp)
    running = still_running

    for gpu in range(N_GPUS):
        if GPU_STATE[gpu] is None and queue:
            exp = queue.pop(0)
            exp["gpu_id"] = gpu
            GPU_STATE[gpu] = exp
            launch_experiment(gpu, exp)
            running.append(exp)

    time.sleep(10)

print(f"[{datetime.now()}] Multi-layer experiments complete.")
