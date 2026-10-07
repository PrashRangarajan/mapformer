"""Pilot reproduction check for RANK_NOWRAP_PREREG.md: the batch's run path (train_rank_nowrap, arm
Vanilla_r2ph_om32 on the 32-torus) must reproduce a stored RANK_ND run (runs/rank_nd/D2/Vanilla_r2ph_s0, trained by
run_rank_nd.sh with the same flags) BITWISE for the first K epochs. train.train() keeps per-epoch losses in a local
list and prints only every 5th epoch to 4 decimals, so the module-global name `float` of mapformer.train (used
exactly once per epoch: `epoch_loss = float(epoch_loss_t)`) is shadowed by a recorder that returns the builtin's
value unchanged; after K epochs it writes the sums and exits (daemon data workers terminate with the process).
Usage (from /home/prashr):
  python3 mapformer/docs/audits/2026-10-06/rank_nowrap_repro.py K OUT.json <train_rank_nowrap CLI args...>
"""
import builtins
import json
import sys

sys.path.insert(0, "/home/prashr")
import mapformer.train                           # noqa: E402,F401
from mapformer import train_rank_nowrap as W      # noqa: E402

# mapformer/__init__.py does `from .train import train`, so the attribute mapformer.train is the FUNCTION; the module
# whose globals train() reads is sys.modules["mapformer.train"] (the first version patched the function and was inert)
T = sys.modules["mapformer.train"]
assert hasattr(T, "train") and callable(T.train) and T.train is W.train_variant.train

def main():
    K, OUT = int(sys.argv[1]), sys.argv[2]
    n_batches = int(sys.argv[sys.argv.index("--n-batches") + 1])
    rec = []

    def _float(x):
        v = builtins.float(x)
        rec.append(v / n_batches)                 # avg_loss = epoch_loss / n_batches, as train() computes it
        if len(rec) >= K:
            json.dump({"epochs": K, "losses": rec, "argv": sys.argv[1:]}, open(OUT, "w"), indent=1)
            raise SystemExit(0)
        return v

    T.float = _float
    sys.argv = ["mapformer.train_rank_nowrap"] + sys.argv[3:]
    W.train_variant.main()


if __name__ == "__main__":          # the guard matters: spawned data workers re-import this file as __mp_main__
    main()
