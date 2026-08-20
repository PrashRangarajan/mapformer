"""The paper's OWN OOD protocol for 1D-2D grid navigation (Table 2).

Protocol taken verbatim from Appendix B:

    "For 1D/2D grid navigation, we trained all models on sequences of l = 128
     steps, a grid size of lgrid = 64, and a probability of placing an object at
     a specific location pempty = 0.5. We tested length generalization on two
     OOD datasets - OOD-d: l = 64, pempty = 0.2, lgrid = 32 and OOD-s: l = 512,
     pempty = 0.8, lgrid = 128."

Note "dense"/"sparse" refer to OBJECT density (p_empty), not visit density --
OOD-d places objects on 80% of cells, OOD-s on 20%. Both also change sequence
length AND grid width simultaneously, so these are joint length+scale shifts.

DISCREPANCY IN THE PAPER (not resolved silently): the Table 2 caption says
"OOD: dense (D) 64/32, sparse (S) 256/128", i.e. l=256 for OOD-s, while
Appendix B says l=512. Both are evaluated here and reported as separate rows.

UNSTATED PARAMETER: the paper does not give K (number of object types) for the
navigation experiment. We use our trained value K=16; it is fixed across IID
and OOD here, and cannot be changed at eval anyway without a vocab mismatch.

The model is held fixed (including path-integrator omega, which was initialised
for grid_size=64 and then trained); only the environment changes. That is the
OOD test.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import torch

from mapformer.environment import GridWorld
from mapformer.train_variant import VARIANT_MAP
from mapformer.eval_paper_task import revisit_accuracy

_REPO = Path(__file__).resolve().parent

# (label, seq_len l, grid width, p_empty)
CONDITIONS = [
    ("IID  l=128 g=64  pe=0.5", 128, 64, 0.5),
    ("OOD-d l=64 g=32  pe=0.2", 64, 32, 0.2),
    ("OOD-s l=256 g=128 pe=0.8", 256, 128, 0.8),   # Table 2 caption
    ("OOD-s l=512 g=128 pe=0.8", 512, 128, 0.8),   # Appendix B
]

# EXTENSION beyond the paper's protocol. The published benchmark saturates at
# 0.96-1.0, so "best" there is a 1-3pp claim on a ceiling and cannot separate
# models. These push the OOD-s condition further in length only -- grid width and
# p_empty stay at the paper's OOD-s values -- to find where the curves actually
# diverge. Reported separately from the protocol rows above; they are OURS, not
# the paper's, and no published number exists to compare them against.
EXTENDED = [
    ("ext-s l=1024 g=128 pe=0.8", 1024, 128, 0.8),
    ("ext-s l=2048 g=128 pe=0.8", 2048, 128, 0.8),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(_REPO / "runs/paper_task"))
    ap.add_argument("--variants", nargs="+",
                    default=["Vanilla", "VanillaEM", "VanillaEM_P0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--env-seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "PAPER_OOD_PROTOCOL.md"))
    ap.add_argument("--extended", action="store_true",
                    help="add length-only extensions past the paper's protocol")
    args = ap.parse_args()

    dev = torch.device(args.device)
    CONDS = CONDITIONS + (EXTENDED if args.extended else [])
    res = {v: {c[0]: [] for c in CONDS} for v in args.variants}

    for v in args.variants:
        cls = VARIANT_MAP[v]
        for s in args.seeds:
            ckpt = Path(args.runs_dir) / f"{v}_s{s}" / f"{v}.pt"
            if not ckpt.exists():
                print(f"MISSING {ckpt} -- skipping")
                continue
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd, cfg = blob["model_state_dict"], blob["config"]

            for label, L, g, pe in CONDS:
                env = GridWorld(size=g, n_obs_types=cfg["n_obs_types"],
                                p_empty=pe, n_landmarks=0, seed=args.env_seed)
                assert env.unified_vocab_size == cfg["vocab_size"], (
                    f"vocab mismatch {env.unified_vocab_size} vs {cfg['vocab_size']}")
                # grid_size stays at the TRAINED value: the model is unchanged,
                # only the world it is evaluated in differs.
                model = cls(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
                            grid_size=cfg["grid_size"]).to(dev)
                model.load_state_dict(sd)
                acc, nll, n = revisit_accuracy(model, env, args.n_batches,
                                               args.batch_size, L, dev)
                res[v][label].append(acc)
                print(f"[{v} s{s}] {label}  acc={acc:.4f} nll={nll:.4f} (n={n})",
                      flush=True)

    def cell(xs):
        if not xs:
            return "n/a"
        return f"{st.mean(xs):.3f}" + (f" ± {st.stdev(xs):.3f}" if len(xs) > 1 else "")

    lines = [
        "# Paper's own OOD protocol (Table 2, 1D-2D grid navigation)",
        "",
        "Appendix B verbatim: trained at l=128, lgrid=64, pempty=0.5; "
        "**OOD-d**: l=64, pempty=0.2, lgrid=32; **OOD-s**: l=512, pempty=0.8, "
        "lgrid=128.",
        "'dense'/'sparse' = OBJECT density (pempty), not visit density.",
        "",
        "The Table 2 caption instead gives OOD-s as l=256; both lengths are "
        "reported since the paper is internally inconsistent.",
        "",
        "Paper's 2D results -- MapWM: IID 0.99, OOD-d 0.99, OOD-s 0.96. "
        "MapEM-os: IID 1.0, OOD-d 0.99, OOD-s 0.97.",
        "",
        "| variant | " + " | ".join(c[0] for c in CONDS) + " |",
        "|---" * (len(CONDS) + 1) + "|",
    ]
    for v in args.variants:
        lines.append(f"| {v} | " + " | ".join(cell(res[v][c[0]]) for c in CONDS) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
