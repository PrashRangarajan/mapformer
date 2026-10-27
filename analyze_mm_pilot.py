"""Pilot readout for the Memory Maze batch (MEMORY_MAZE_PREREG.md, Pilots). Reads <root>/*/results.json of --pilot runs.

  python3 -m mapformer.analyze_mm_pilot --root <pilot runs>

Per run: arm, seed/tag, parameters, seconds per epoch, peak GiB, final training loss, the validation curve (R1 on
non-local blind re-entries of train-part0 1500-1599: training-split episodes never trained on, never a scoring split),
the plateau epoch (first epoch after which every later validation point is within 0.01 of the final one), and the
probe calibration (model-free constant baseline on eval vs the published 64.8 / 78.3).
"""
import argparse
import glob
import json
import os

import numpy as np


def plateau_epoch(val, tol=0.01):
    if not val:
        return None
    ep = [v[0] for v in val]; r = np.array([v[1] for v in val])
    for i in range(len(r)):
        if np.all(np.abs(r[i:] - r[-1]) <= tol):
            return int(ep[i])
    return int(ep[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for f in sorted(glob.glob(os.path.join(a.root, "*", "results.json"))):
        name = os.path.basename(os.path.dirname(f))
        r = json.load(open(f))
        c = r["config"]
        val = r.get("val", [])
        row = dict(arm=c["arm"], n_params=c["n_params"], n_trainable=c["n_trainable"], sec_per_epoch=r["sec_per_epoch"],
                   peak_gib=r["peak_gib"], epochs=c["epochs"], final_loss=r["hist"][-1] if r["hist"] else None,
                   val_final=val[-1][1] if val else None, val_max=max(v[1] for v in val) if val else None,
                   val_all_final=val[-1][2] if val else None, plateau_epoch=plateau_epoch(val),
                   val_curve=[(int(e), round(x, 4)) for e, x, _ in val], probe=r.get("probe"),
                   score_sec=r.get("score_sec"), probe_sec=r.get("probe_sec"), env=r.get("env"))
        out[name] = row
        print(f"{name:28s} params {row['n_params']:7d} {row['sec_per_epoch']:.2f}s/ep peak {row['peak_gib']} GiB "
              f"loss {row['final_loss']:.4f} val final {row['val_final']} max {row['val_max']} plateau@{row['plateau_epoch']}")
        print("    curve " + " ".join(f"{e}:{x:.3f}" for e, x in row["val_curve"]))
        if row["probe"]:
            print("    probe " + json.dumps(row["probe"]))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
