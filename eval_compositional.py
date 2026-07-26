"""
Evaluate compositional-task checkpoints: exact-revisit vs cross-instance
accuracy + NLL, overall and on non-blank cells, across sequence lengths.

Fresh env per eval (new templates+assignment every episode, distinct seed from
training) so nothing is memorised.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.abspath(__file__))

from mapformer.environment_compositional import CompositionalGridWorld
from mapformer.train_variant import VARIANT_MAP


@torch.no_grad()
def eval_ckpt(ckpt_path, lengths, n_traj, device, seed=10000,
              room_size=8, n_templates=4, grid_size=64, batch_size=64):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    variant = ck["variant"]
    model = VARIANT_MAP[variant](
        vocab_size=ck["vocab_size"], d_model=ck["d_model"],
        n_heads=ck["n_heads"], n_layers=ck["n_layers"], grid_size=grid_size,
    ).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    wants_seg = getattr(model, "wants_seg_id", False)  # oracle room segmentation
    blank = grid_size and None
    results = {}
    for T in lengths:
        env = CompositionalGridWorld(size=grid_size, room_size=room_size,
                                     n_templates=n_templates, seed=seed)
        blank_tok = env.unified_blank
        agg = {k: [0, 0, 0.0] for k in ["exact", "cross", "cross_nb"]}  # [correct,n,nll_sum]
        done = 0
        while done < n_traj:
            b = min(batch_size, n_traj - done)
            batch = env.generate_batch(b, T)
            tokens = batch[0].to(device)
            if wants_seg:
                nr = torch.tensor(batch[5]["new_room"], dtype=torch.long, device=device)
                seg = (torch.cumsum(nr, dim=1) - 1).repeat_interleave(2, dim=1)
                model._batch_seg_id = seg[:, :-1]
            exact_m = batch[2][:, 1:].to(device)
            cross_m = batch[4][:, 1:].to(device)
            inp = tokens[:, :-1]; tgt = tokens[:, 1:]
            logits = model(inp)
            logp = F.log_softmax(logits, dim=-1)
            pred = logits.argmax(-1)
            for name, m in [("exact", exact_m), ("cross", cross_m)]:
                if m.sum() > 0:
                    corr = (pred[m] == tgt[m]).sum().item()
                    nll = -logp[m].gather(-1, tgt[m].unsqueeze(-1)).sum().item()
                    agg[name][0] += corr; agg[name][1] += int(m.sum()); agg[name][2] += nll
            nb = cross_m & (tgt != blank_tok)
            if nb.sum() > 0:
                corr = (pred[nb] == tgt[nb]).sum().item()
                nll = -logp[nb].gather(-1, tgt[nb].unsqueeze(-1)).sum().item()
                agg["cross_nb"][0] += corr; agg["cross_nb"][1] += int(nb.sum()); agg["cross_nb"][2] += nll
            done += b
        row = {}
        for k, (c, n, s) in agg.items():
            row[f"{k}_acc"] = c / max(n, 1)
            row[f"{k}_nll"] = s / max(n, 1)
            row[f"{k}_n"] = n
        results[T] = row
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return variant, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--lengths", nargs="+", type=int, default=[256, 512, 1024, 2048])
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64,
                    help="trajectories per forward pass (lower to fit long T in GPU memory; "
                         "does not change results)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(_REPO, "COMPOSITIONAL_RESULTS.md"))
    args = ap.parse_args()

    all_rows = {}
    for cp in args.checkpoints:
        v, res = eval_ckpt(cp, args.lengths, args.n_traj, args.device, batch_size=args.batch)
        all_rows[v] = res
        print(f"\n=== {v} ===")
        for T, r in res.items():
            print(f" T={T:5d} exact_acc={r['exact_acc']:.3f} "
                  f"cross_acc={r['cross_acc']:.3f} cross_nb_acc={r['cross_nb_acc']:.3f} "
                  f"cross_nll={r['cross_nll']:.3f}")

    lines = ["# Compositional-motif results\n",
             "Cross-instance = compositional target (motif seen elsewhere). "
             "cross_nb = non-blank subset. Fresh env, OOD length.\n"]
    for T in args.lengths:
        lines.append(f"\n## T={T}\n")
        lines.append("| variant | exact_acc | cross_acc | cross_nb_acc | cross_nll |")
        lines.append("|---|---|---|---|---|")
        for v, res in all_rows.items():
            r = res[T]
            lines.append(f"| {v} | {r['exact_acc']:.3f} | {r['cross_acc']:.3f} "
                         f"| {r['cross_nb_acc']:.3f} | {r['cross_nll']:.3f} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
