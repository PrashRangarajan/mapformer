#!/usr/bin/env python3
"""Level15PC multi-seed orchestrator.

Combines Level 1.5 InEKF + Predictive Coding aux loss on the standard
MapFormer-WM backbone (no Grid). Tests whether forward-model PC and
inverse-model Kalman are complementary.

Note: this orchestrator EXCLUDES (clean s0, noise s0) which are
launched separately in the parent shell (training in flight).
Picks up the remaining 7 runs to complete 3 configs × 3 seeds.

Usage:
  nohup python3 -u -m mapformer.orchestrator_level15pc \
      > mapformer/logs/orchestrator_level15pc.log 2>&1 &
"""

import subprocess, time, os
from pathlib import Path
from datetime import datetime


REPO = Path("/home/prashr/mapformer")
RUNS_DIR = REPO / "runs"
LOGS = REPO / "logs"; LOGS.mkdir(exist_ok=True)

N_GPUS = 2
GPU_STATE = [None, None]

EXPERIMENTS = []
for cfg in [("clean", 0.0, 0), ("noise", 0.10, 0), ("lm200", 0.10, 200)]:
    tag, p_noise, n_lm = cfg
    for seed in [0, 1, 2]:
        out_dir = RUNS_DIR / f"Level15PC_{tag}" / f"seed{seed}"
        EXPERIMENTS.append({
            "variant": "Level15PC", "seed": seed, "n_landmarks": n_lm,
            "p_action_noise": p_noise, "output_dir": str(out_dir),
            "ckpt_path": out_dir / "Level15PC.pt",
            "n_epochs": 50, "n_batches": 156, "aux_coef": 0.1, "tag": tag,
        })

REMAINING = [e for e in EXPERIMENTS if not e["ckpt_path"].exists()]
print(f"[{datetime.now()}] Level15PC orchestrator: {len(REMAINING)}/{len(EXPERIMENTS)} to run")


def launch(gpu, exp):
    out_dir = Path(exp["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"Level15PC_{exp['tag']}_s{exp['seed']}.log"
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = ["python3", "-u", "-m", "mapformer.train_variant",
           "--variant", exp["variant"], "--seed", str(exp["seed"]),
           "--n-landmarks", str(exp["n_landmarks"]),
           "--p-action-noise", str(exp["p_action_noise"]),
           "--epochs", str(exp["n_epochs"]),
           "--n-batches", str(exp["n_batches"]),
           "--aux-coef", str(exp["aux_coef"]),
           "--device", "cuda", "--output-dir", exp["output_dir"]]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd="/home/prashr", env=env,
                            stdout=log_f, stderr=subprocess.STDOUT)
    exp["proc"] = proc; exp["log_f"] = log_f; exp["start_time"] = time.time()
    print(f"[{datetime.now()}] GPU{gpu}: {exp['variant']} {exp['tag']} s{exp['seed']} (pid {proc.pid})")


def done(exp): return "proc" in exp and exp["proc"].poll() is not None


queue = list(REMAINING); running = []
while queue or running:
    still = []
    for exp in running:
        if done(exp):
            exp["log_f"].close()
            ok = exp["ckpt_path"].exists(); rc = exp["proc"].returncode
            dt = time.time() - exp["start_time"]
            status = "OK" if (rc == 0 and ok) else f"FAIL(rc={rc})"
            print(f"[{datetime.now()}] GPU{exp['gpu_id']}: "
                  f"{exp['variant']} {exp['tag']} s{exp['seed']} -> {status} ({dt/60:.1f}m)")
            GPU_STATE[exp["gpu_id"]] = None
        else:
            still.append(exp)
    running = still
    for gpu in range(N_GPUS):
        if GPU_STATE[gpu] is None and queue:
            exp = queue.pop(0); exp["gpu_id"] = gpu
            GPU_STATE[gpu] = exp; launch(gpu, exp); running.append(exp)
    time.sleep(10)

print(f"[{datetime.now()}] Level15PC orchestrator done.")
