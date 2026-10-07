"""GAIN_PHASE pilot, part 2 (after the 30-epoch pilot showed the gain arms far behind the rotary arms): the batch's
REAL schedule (900-epoch cosine, 45-epoch warmup), stopped after K epochs, for pilot seed 110 -- does a gain arm learn
under the batch's schedule? Prints the per-epoch training loss (train.py's own log line every 5 epochs) and, at the end,
the epoch-K loss; the comparators are the LEAK logs of MapWM / NormStep (seeds 0-7, same schedule), read at the same
epochs. Nothing is saved (stopped before the checkpoint). Usage: python3 gain_phase_pilot_long.py ARM SEED K DEVICE"""
import multiprocessing as mp
import os
import sys
import tempfile

import torch

sys.path.insert(0, "/home/prashr")
NB = 98


class Stop(Exception):
    pass


def run(arm, seed, k, dev):
    rec = []
    Base = torch.nn.CrossEntropyLoss

    class Recorder(Base):
        def forward(self, a, b):
            out = super().forward(a, b); rec.append(1)
            if len(rec) == k * NB:
                raise Stop
            return out
    torch.nn.CrossEntropyLoss = Recorder
    from mapformer import train_gain_phase
    sys.argv = ["train_gain_phase", "--variant", arm, "--seed", str(seed), "--epochs", "900", "--n-steps", "1024",
                "--batch-size", "16", "--device", dev, "--output-dir", tempfile.mkdtemp(prefix=f"gain_phase_long_{arm}_")]
    try:
        train_gain_phase.train_newobj.main()
    except Stop:
        print(f"stopped after {k} epochs of the 900-epoch schedule", flush=True)
    for c in mp.active_children():
        c.terminate()
    os._exit(0)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
