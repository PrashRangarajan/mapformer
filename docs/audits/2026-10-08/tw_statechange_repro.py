"""Reproduction check for TW_STATECHANGE_PREREG.md (pilot part a; GAIN_PHASE's gain_phase_repro.py pattern): through the
batch's entry point train_tw_statechange at p_take = p_drop = 0 with TextWorld's vocabulary (--no-state-vocab), the
TW_NORMSTEP recipe exactly (900-epoch cosine schedule, T = 1024, batch 16, 98 batches, lr 1e-3, data-workers 3) must
reproduce the stored runs/tw_normstep/p0/{arm}_s10 per-epoch training losses BIT FOR BIT. Training is stopped after K
epochs by a recording loss module (the schedule is the 900-epoch one, so the first K epochs are the stored run's first K);
each epoch loss is re-accumulated exactly as train.py does (float64 sum of the per-batch losses in order, / n_batches)
and compared with ==. The stored runs were trained on GPU: a CPU run is a test of the script, not of reproduction.
Usage: python3 tw_statechange_repro.py [K] [device]"""
import os
import subprocess
import sys
import tempfile

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
    from mapformer import train_tw_statechange
    sys.argv = ["train_tw_statechange", "--arm", arm, "--seed", "10", "--epochs", "900", "--n-steps", "1024",
                "--batch-size", "16", "--p-take", "0", "--p-drop", "0", "--no-state-vocab", "--device", dev,
                "--output-dir", tempfile.mkdtemp(prefix=f"twsc_repro_{arm}_")]      # nothing is written: stopped first
    try:
        train_tw_statechange.main()
    except Stop:
        pass
    ref = torch.load(f"{REPO}/runs/tw_normstep/p0/{arm}_s10/{arm}.pt", map_location="cpu", weights_only=False)["losses"]
    n_eq = 0
    for e in range(k):
        acc = torch.zeros((), dtype=torch.float64, device=rec[0].device)
        for l in rec[e * NB:(e + 1) * NB]:
            acc += l
        mine = float(acc) / NB
        n_eq += mine == ref[e]
        print(f"  {arm} epoch {e + 1}: wrapper {mine!r} stored {ref[e]!r} {'EQUAL' if mine == ref[e] else 'DIFFERENT'}")
    print(f"{arm} s10: {n_eq}/{k} epochs bitwise equal -> {'PASS' if n_eq == k else 'FAIL'}", flush=True)
    import multiprocessing as mp                  # the generator is never closed (training stopped by exception)
    for c in mp.active_children():
        c.terminate()
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    print(f"torch {torch.__version__}; K = {K} epochs of the 900-epoch TW_NORMSTEP schedule; device {dev}", flush=True)
    for arm in ("MapWM", "NormStep", "DirOnly"):
        subprocess.run([sys.executable, "-u", __file__, "--child", arm, str(K), dev], check=False)
