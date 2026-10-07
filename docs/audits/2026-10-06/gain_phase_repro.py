"""Reproduction check for GAIN_PHASE_PREREG.md (pilot part a): through the batch's entry point (train_gain_phase, which
also imports model_gain_phase), MapWM s0 and NormStep s0 at the LEAK recipe exactly (run_leak.sh's flags: 900-epoch
cosine schedule, T=1024, batch 16, 98 batches, lr 1e-3, data-workers 3) must reproduce the stored LEAK runs'
per-epoch training losses BIT FOR BIT. Training is stopped after K epochs by a recording loss module (the schedule is
the 900-epoch one, so the first K epochs are the stored run's first K); each epoch loss is re-accumulated exactly as
train.py does (float64 sum of the per-batch losses in order, / n_batches) and compared with ==.
Usage: python3 gain_phase_repro.py [K] [device]  -> gain_phase_repro_out.txt"""
import os
import subprocess
import tempfile
import sys

import torch

sys.path.insert(0, "/home/prashr")
REPO = "/home/prashr/mapformer"
NB = 98


class Stop(Exception):
    pass


def child(arm, k, dev):
    rec = []
    Base = torch.nn.CrossEntropyLoss

    class Recorder(Base):
        def forward(self, a, b):
            out = super().forward(a, b); rec.append(out.detach().to(torch.float64))
            if len(rec) == k * NB:
                raise Stop
            return out
    torch.nn.CrossEntropyLoss = Recorder
    from mapformer import train_gain_phase
    sys.argv = ["train_gain_phase", "--variant", arm, "--seed", "0", "--epochs", "900", "--n-steps", "1024",
                "--batch-size", "16", "--device", dev, "--output-dir", tempfile.mkdtemp(prefix=f"gain_phase_repro_{arm}_")]   # nothing is written: stopped before saving
    try:
        train_gain_phase.train_newobj.main()
    except Stop:
        pass
    ref = torch.load(f"{REPO}/runs/leak/p0/{arm}_s0/{arm}.pt", map_location="cpu", weights_only=False)["losses"]
    n_eq = 0
    for e in range(k):
        acc = torch.zeros((), dtype=torch.float64, device=rec[0].device)
        for l in rec[e * NB:(e + 1) * NB]:
            acc += l
        mine = float(acc) / NB
        n_eq += mine == ref[e]
        print(f"  {arm} epoch {e + 1}: wrapper {mine!r} stored {ref[e]!r} {'EQUAL' if mine == ref[e] else 'DIFFERENT'}")
    print(f"{arm} s0: {n_eq}/{k} epochs bitwise equal -> {'PASS' if n_eq == k else 'FAIL'}", flush=True)
    import multiprocessing as mp                  # the generator is never closed (training stopped by exception):
    for c in mp.active_children():                # terminate its workers explicitly; os._exit would orphan them
        c.terminate()
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 5             # parsed here: spawned data workers re-import this file
    dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    print(f"torch {torch.__version__}; K = {K} epochs of the 900-epoch LEAK schedule; device {dev}", flush=True)
    for arm in ("MapWM", "NormStep"):
        subprocess.run([sys.executable, "-u", __file__, "--child", arm, str(K), dev], check=False)
