#!/usr/bin/env python3
"""MambaLike-only orchestrator — highest-priority baseline.

Skips LSTM (already done multi-seed) and CoPE (4/9 done, rest deferred
due to CoPE's extreme per-epoch cost on our setup — ~500 min/run vs
LSTM's 8 min/run). The point of this orchestrator is to answer the
"does generic Mamba-family SSM subsume MapFormer+L1.5?" question fast.

9 runs (3 configs x 3 seeds). Each expected ~10-15 min.

Usage:
  nohup python3 -u -m mapformer.orchestrator_mambalike \
      > mapformer/orchestrator_mambalike.log 2>&1 &
"""

import subprocess, time, os
from pathlib import Path
from datetime import datetime

REPO = Path("/home/prashr/mapformer")
RUNS_DIR = REPO / "runs"
N_GPUS = 2
GPU_STATE = [None, None]

EXPERIMENTS = []
for cfg in [("clean", 0.0, 0), ("noise", 0.10, 0), ("lm200", 0.10, 200)]:
    tag, p_noise, n_lm = cfg
    for seed in [0, 1, 2]:
        out_dir = RUNS_DIR / f"MambaLike_{tag}" / f"seed{seed}"
        EXPERIMENTS.append({
            "variant": "MambaLike", "seed": seed, "n_landmarks": n_lm,
            "p_action_noise": p_noise, "output_dir": str(out_dir),
            "ckpt_path": out_dir / "MambaLike.pt",
            "n_epochs": 50, "n_batches": 156, "tag": tag,
        })

REMAINING = [e for e in EXPERIMENTS if not e["ckpt_path"].exists()]
print(f"[{datetime.now()}] MambaLike orchestrator: {len(REMAINING)}/{len(EXPERIMENTS)} to run")


def launch(gpu, exp):
    out_dir = Path(exp["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = ["python3", "-u", "-m", "mapformer.train_variant",
           "--variant", exp["variant"], "--seed", str(exp["seed"]),
           "--n-landmarks", str(exp["n_landmarks"]),
           "--p-action-noise", str(exp["p_action_noise"]),
           "--epochs", str(exp["n_epochs"]),
           "--n-batches", str(exp["n_batches"]),
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

print(f"[{datetime.now()}] MambaLike orchestrator done.")
