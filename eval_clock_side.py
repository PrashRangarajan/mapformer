"""CLOCK_SIDE_PREREG.md Amendment 2: re-evaluation of every saved checkpoint on a larger held-out stream (the primary
readout). train_recency's in-training evaluation scores ~213 targets per run at T = 1024 (per-seed SE ~0.03), too few
for the registered NO DIFFERENCE branch to be reachable (audit M1). Here each run is scored on 64 x 8 = 512 sequences
(~1,700 targets) at T = 1024 from a fresh stream (np RandomState(7000 + seed), RecencyWorld(seed=10000), the same items
for every arm at a seed), with train_recency.evaluate unchanged, in eval mode; then, in the same process after
`rescore_hook.install("auto")`, the dropout-scale re-scored accuracy (audit M4). Explicit attention (no SDPA) at eval;
--fast-attn logits agree to <= 1e-6 (audit). Writes CLOCK_SIDE_EVAL.json.
  python3 -m mapformer.eval_clock_side --device cuda:0
"""
import argparse, json
import torch

from mapformer.environment_recency import RecencyWorld
from mapformer.train_recency import evaluate
from mapformer.train_variant import VARIANT_MAP
from mapformer import rescore_hook

REPO = "/home/prashr/mapformer"; R = f"{REPO}/runs/clock_side/p0"
ARMS = ("Signed_r4", "Abs_r4", "CARoPE_r4", "Pos_r4", "RoPE"); SEEDS = list(range(200, 208)); T = 1024


def load(arm, s, dev):
    b = torch.load(f"{R}/{arm}_s{s}/{arm}_recency.pt", map_location="cpu", weights_only=False)
    m = VARIANT_MAP[arm](vocab_size=b["vocab_size"], d_model=b["d_model"], n_heads=b["n_heads"], n_layers=b["n_layers"],
                         grid_size=b["grid_size"])
    m.load_state_dict(b["model_state"]); return m.to(dev), b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    out = {}
    for rescored in (False, True):
        if rescored:
            rescore_hook.install("auto")
        for arm in ARMS:
            for s in SEEDS:
                m, b = load(arm, s, a.device)
                env = RecencyWorld(seed=10000, n_symbols=b["n_symbols"], k_max=b["k_max"], p_query=0.25, min_gap=b["min_gap"])
                with torch.no_grad():
                    acc, nll, n, _ = evaluate(m, env, T, 64, 8, a.device, seed=7000 + s)
                key = f"{arm}|{s}"; out.setdefault(key, {})
                out[key]["acc_rs" if rescored else "acc"] = acc; out[key]["n"] = n
                print(f"{'re-scored' if rescored else 'eval mode'} {key}: acc {acc:.4f} (n {n})", flush=True)
    json.dump(out, open(f"{REPO}/CLOCK_SIDE_EVAL.json", "w"), indent=1)
    rescore_hook._report()                                   # hooked layers and skipped classes


if __name__ == "__main__":
    main()
