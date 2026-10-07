"""Evaluation step of GAIN_PHASE_PREREG.md: gain_phase_eval.evaluate_run on every run of the batch (one shared eval
stream: leak_eval.sequences, test and train pools), written to GAIN_PHASE_EVAL.json keyed '<arm>|<seed>'.
`python3 -m mapformer.eval_gain_phase [device]`"""
import json
import sys
import time

from mapformer.analyze_gain_phase import ARMS, SEEDS, R, REPO
from mapformer.gain_phase_eval import evaluate_run, sequences


def main(dev="cuda:0", runs=R, seeds=SEEDS, out=f"{REPO}/GAIN_PHASE_EVAL.json", arms=ARMS):
    data = {pool: sequences(pool) for pool in ("test", "train")}
    J = {}
    for s in seeds:
        for a in arms:
            t0 = time.time()
            J[f"{a}|{s}"] = r = evaluate_run(f"{runs}/{a}_s{s}/{a}.pt", a, dev, data)
            print(f"{a:9s} s{s}: acc {r['acc']:.4f} L_ms {r['L_ms']:+.4f} L_zero {r['L_zero']:+.4f} S_id {r['S_id']:.4f} "
                  f"x2 {r['x2']:.4f} x4 {r['x4']:.4f} train {r['train_x1']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    json.dump(J, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
