"""Evaluate the D x r rank-threshold batch.

Metric is eval_noise_refine's, unchanged: held-out revisit accuracy on a FRESH
environment (env-seed 10000, so the observation map differs from training), scored
only at revisited observation positions, plus NLL.

Reports, per D:
  - mean +/- sd per arm at each length
  - the paired contrast against r=2 with its MDE (2.8*sd/sqrt(n)) and sign count
  - final training loss per arm, because rule 10 says convergence is verified
    before a table is read and rule 9 says an accuracy gap that is only a loss gap
    is not an architecture result
  - the loss-matched residual, since r(loss, acc) is reported alongside

Per-seed numbers are written to a JSON beside the markdown so any headline here
can be re-derived without rerunning.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from mapformer.environment_nd import GridWorldND
from mapformer.train_variant import VARIANT_MAP


@torch.no_grad()
def evaluate(model, env, T, n_trials, dev, seed):
    np.random.seed(seed)
    ok = tot = 0; nll = 0.0
    for _ in range(n_trials):
        tok, _om, rev = env.generate_trajectory(T)
        tok = tok.unsqueeze(0).to(dev)
        lp = F.log_softmax(model(tok[:, :-1]).float(), dim=-1)
        pred = lp.argmax(-1)[0]; tgt = tok[0, 1:]; m = rev[1:].to(dev)
        if m.sum() == 0:
            continue
        ok += (pred[m] == tgt[m]).sum().item(); tot += int(m.sum())
        idx = torch.arange(lp.shape[1], device=dev)[m]
        nll += float(-lp[0, idx, tgt[m]].sum())
    return (ok / tot, nll / tot) if tot else (None, None)


def mde(x):
    x = np.asarray(x, dtype=float)
    return 2.8 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="D:size:arm1,arm2,... one per dimensionality")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    ap.add_argument("--lengths", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--n-obs-types", type=int, default=16)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--env-seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)

    raw, o = {}, ["# The rank threshold across dimension", "",
                  "Held-out revisit accuracy on a fresh environment "
                  f"(env-seed {a.env_seed}), {a.n_trials} trajectories, "
                  f"{len(a.seeds)} seeds, one batch.", ""]

    for spec in a.configs:
        D, N, arms = spec.split(":"); D, N = int(D), int(N)
        arms = arms.split(",")
        env = GridWorldND(dims=D, size=N, n_obs_types=a.n_obs_types,
                          seed=a.env_seed)
        o += [f"## D = {D}, grid {N} ({N**D} cells, {2*D} actions, "
              f"vocab {env.unified_vocab_size})", ""]
        acc = {arm: {T: {} for T in a.lengths} for arm in arms}
        loss = {arm: {} for arm in arms}
        for arm in arms:
            for s in a.seeds:
                cp = Path(a.runs_dir) / f"D{D}" / f"{arm}_s{s}" / f"{arm}.pt"
                if not cp.exists():
                    continue
                b = torch.load(cp, map_location="cpu", weights_only=False)
                c = b["config"]
                m = VARIANT_MAP[arm](vocab_size=c["vocab_size"],
                                     d_model=c["d_model"], n_heads=c["n_heads"],
                                     n_layers=c["n_layers"],
                                     grid_size=c["grid_size"])
                m.load_state_dict(b["model_state_dict"]); m.to(dev).eval()
                loss[arm][s] = float(b["losses"][-1])
                for T in a.lengths:
                    r, _n = evaluate(m, env, T, a.n_trials, dev, a.env_seed + s)
                    if r is not None:
                        acc[arm][T][s] = r
                del m
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
        raw[f"D{D}"] = {"grid": N, "acc": acc, "final_loss": loss}

        o += ["| arm | final loss | " + " | ".join(f"T={T}" for T in a.lengths) + " |",
              "|---|---|" + "---|" * len(a.lengths)]
        for arm in arms:
            L = list(loss[arm].values())
            cells = []
            for T in a.lengths:
                v = list(acc[arm][T].values())
                cells.append(f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f} (n={len(v)})"
                             if len(v) > 1 else "—")
            o.append(f"| `{arm}` | {np.mean(L):.4f} |" if L else f"| `{arm}` | — |")
            o[-1] += " " + " | ".join(cells) + " |"
        o.append("")

        base = arms[0]
        o += [f"Paired against `{base}`:", "",
              "| arm | length | delta | sd | MDE | seeds + | verdict |",
              "|---|---|---|---|---|---|---|"]
        for arm in arms[1:]:
            for T in a.lengths:
                common = sorted(set(acc[base][T]) & set(acc[arm][T]))
                if len(common) < 2:
                    continue
                d = np.array([acc[arm][T][s] - acc[base][T][s] for s in common])
                M = mde(d); pos = int((d > 0).sum())
                verdict = ("DETECTABLE" if abs(d.mean()) > M else "unmeasured")
                if abs(d.mean()) > M and d.mean() < 0:
                    verdict = "DETECTABLE NEGATIVE"
                o.append(f"| `{arm}` | {T} | {d.mean():+.3f} | {d.std(ddof=1):.3f} "
                         f"| {M:.3f} | {pos}/{len(common)} | {verdict} |")
        o.append("")

        # rule 9: is the accuracy contrast just the training loss?
        for T in a.lengths:
            xs, ys = [], []
            for arm in arms:
                for s in acc[arm][T]:
                    if s in loss[arm]:
                        xs.append(loss[arm][s]); ys.append(acc[arm][T][s])
            if len(xs) > 3:
                r = float(np.corrcoef(xs, ys)[0, 1])
                o.append(f"- r(final loss, accuracy) at T={T}: **{r:+.3f}** "
                         f"over {len(xs)} runs"
                         + ("  — |r| > 0.98, the held-out eval carries no "
                            "information the loss does not" if abs(r) > 0.98 else ""))
        o.append("")

    Path(a.out).write_text("\n".join(o) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(raw, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
