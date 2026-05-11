"""GSF mode-weight diagnostic: is the multi-modal Bayesian filter actually multi-modal,
or does it collapse to a single dominant mode early?

Loads each trained Level15GSF checkpoint, runs eval trajectories, extracts the
mixture log-weights stashed in `model.last_log_w` (shape (B, K, L)), and reports:

- per-time entropy of the K-way mixture (averaged over trajectories)
- effective number of modes = exp(entropy) per timestep
- per-trajectory: which mode wins at the end, and how often modes change winner

If entropy collapses to ~0 within a few steps → K=8 is overkill, K=2 would do.
If entropy stays high → genuine multi-modal posterior, K=8 is justified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mapformer.environment import GridWorld
from mapformer.model_inekf_gsf import MapFormerWM_Level15GSF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-md", default="GSF_MODES_DIAGNOSTIC.md")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--T", type=int, default=512)
    args = ap.parse_args()

    env = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=200, seed=1000)

    lines = []
    lines.append("# GSF mode-weight diagnostic — is Level15GSF actually multi-modal?\n")
    lines.append("Loads trained Level15GSF checkpoints (3 seeds on lm200), runs eval")
    lines.append("trajectories at T=512, extracts the K=8 mixture weights over time, reports")
    lines.append("entropy + effective-mode-count + winner-mode statistics.\n")
    lines.append("Reading guide:")
    lines.append("- `entropy(t)` close to log(K)=2.08 → uniform → all modes active")
    lines.append("- `entropy(t)` close to 0 → one mode dominates → mixture is degenerate")
    lines.append("- `eff_modes(t) = exp(entropy(t))` is a nicer 'how many modes are alive' number\n")

    for s in [0, 1, 2]:
        ckpt = Path(f"mapformer/runs/Level15GSF_lm200/seed{s}/Level15GSF.pt")
        if not ckpt.exists():
            lines.append(f"## seed {s}: ckpt missing\n"); continue
        c = torch.load(ckpt, map_location="cuda", weights_only=False)
        cfg = c["config"]
        m = MapFormerWM_Level15GSF(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
            grid_size=cfg["grid_size"], n_modes=8,
        ).cuda().eval()
        m.load_state_dict(c["model_state_dict"])

        torch.manual_seed(seed:=2000 + s); np.random.seed(seed)
        all_entropies = []   # (B*trials, L)
        winner_modes = []    # final-step winning mode per trajectory
        changeover_counts = []
        with torch.no_grad():
            for _ in range(args.n_trials):
                tokens, _, _ = env.generate_trajectory(args.T)
                _ = m(tokens.unsqueeze(0).cuda())
                log_w = m.last_log_w           # (1, K, L)
                log_w_norm = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
                w = log_w_norm.exp()           # (1, K, L)
                ent = -(w * log_w_norm).sum(dim=1).squeeze(0).cpu().numpy()  # (L,)
                winners = w.argmax(dim=1).squeeze(0).cpu().numpy()          # (L,)
                all_entropies.append(ent)
                winner_modes.append(int(winners[-1]))
                # Count winner changes: positions where winner[t] != winner[t-1]
                changes = int((winners[1:] != winners[:-1]).sum())
                changeover_counts.append(changes)

        ent_arr = np.stack(all_entropies)  # (n_trials, L)
        mean_ent = ent_arr.mean(axis=0)
        max_ent = float(np.log(8))
        # Summary stats at three time points
        lines.append(f"## seed {s}\n")
        lines.append(f"- max possible entropy log(K=8) = {max_ent:.3f}")
        lines.append(f"- entropy at t=16: mean {mean_ent[16]:.3f} (eff_modes {np.exp(mean_ent[16]):.2f})")
        lines.append(f"- entropy at t=128: mean {mean_ent[128]:.3f} (eff_modes {np.exp(mean_ent[128]):.2f})")
        lines.append(f"- entropy at t=512: mean {mean_ent[-1]:.3f} (eff_modes {np.exp(mean_ent[-1]):.2f})")
        winner_counts = np.bincount(winner_modes, minlength=8).tolist()
        lines.append(f"- final-step winner mode distribution (over {args.n_trials} trajectories): {winner_counts}")
        lines.append(f"- mean winner-changes per trajectory: {np.mean(changeover_counts):.1f}\n")
        del m; torch.cuda.empty_cache()

    lines.append("## Interpretation\n")
    lines.append("If eff_modes drops to ~1 by t=16 and stays there:")
    lines.append("- GSF is essentially K-way ensemble that collapses → K=2 would be enough")
    lines.append("- Win over Level15 is from training-time diversity, not test-time multimodality\n")
    lines.append("If eff_modes stays > 2 even at t=512:")
    lines.append("- Real multi-modal posterior across trajectory")
    lines.append("- K=8 justified; aliased-obs ambiguity persists across long sequences\n")
    lines.append("If winner-changes per trajectory > 5:")
    lines.append("- Different modes win at different times → posterior is dynamic")
    lines.append("- The mixture is doing real work, not just ensembling\n")
    lines.append("*Auto-generated by probe_gsf_modes.py*\n")

    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
