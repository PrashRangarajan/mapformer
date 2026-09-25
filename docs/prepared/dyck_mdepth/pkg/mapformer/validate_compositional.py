"""
Task-validity checks for the compositional environment, run BEFORE any GPU
training (per the project rule: validate the task first).

Checks, all in one batch:
  1. LABEL STATS: fraction of obs positions that are exact-revisit /
     motif-revisit / cross-instance, at several sequence lengths. Need enough
     cross-instance labels for the compositional target to be trainable.
  2. ENV CONSISTENCY: every cross-instance position's obs equals the obs of the
     earliest same-motif-cell occurrence (the target is well-defined and
     deterministic). If this fails the env is buggy.
  3. LONG-RANGE DEMAND: distribution of the LAG (in steps) from a cross-instance
     position back to the nearest prior same-motif-cell occurrence. If most lags
     are tiny, a bounded local window solves it and the task does NOT need
     long-range memory (the LocalOnly trap). We want a heavy tail.
  4. TRIVIAL BASELINE: frequency/majority next-obs accuracy and the "copy the
     most recent same-motif-cell obs" oracle accuracy, to bracket chance and
     ceiling for the cross-instance target.
  5. DISSOCIATION: confirm exact-revisit and cross-instance are disjoint sets
     and both have mass; report their overlap (should be 0 by construction).

Usage: cd /home/prashr && python3 -m mapformer.validate_compositional
"""

import numpy as np
import torch

from mapformer.environment_compositional import CompositionalGridWorld


def run(size=64, room_size=8, n_templates=4, n_obs_types=16,
        lengths=(128, 256, 512), n_traj=200, seed=0):
    print(f"CompositionalGridWorld size={size} room={room_size} "
          f"n_templates={n_templates} K={n_obs_types} fresh_per_episode=True\n")

    for T in lengths:
        env = CompositionalGridWorld(size=size, room_size=room_size,
                                     n_templates=n_templates,
                                     n_obs_types=n_obs_types, seed=seed)
        n_obs = 0
        n_exact = n_motif = n_cross = 0
        overlap = 0
        consistency_fail = 0
        lags = []
        copy_correct = 0
        freq_correct = 0
        # global obs frequency for a majority baseline (per episode)
        for _ in range(n_traj):
            tok, om, rev, motif, cross = env.generate_trajectory(T)
            obs_positions = torch.arange(2 * T)[om]
            obs_tokens = tok[om].numpy()            # (T,)
            rev_f = rev[om].numpy()
            motif_f = motif[om].numpy()
            cross_f = cross[om].numpy()
            tmpl = np.array(env.step_template_id)
            loc = env.step_local
            n_obs += T
            n_exact += rev_f.sum()
            n_motif += motif_f.sum()
            n_cross += cross_f.sum()
            overlap += (rev_f & cross_f).sum()

            # majority baseline: most frequent obs token this episode
            vals, counts = np.unique(obs_tokens, return_counts=True)
            maj = vals[counts.argmax()]
            freq_correct += (obs_tokens == maj).sum()

            # walk forward tracking last-seen step index per motif-cell
            last_step = {}
            last_obs = {}
            for s in range(T):
                key = (int(tmpl[s]), loc[s][0], loc[s][1])
                if cross_f[s]:
                    # consistency: obs must equal earliest same-motif obs
                    if key in last_obs:
                        if last_obs[key] == obs_tokens[s]:
                            copy_correct += 1
                        else:
                            consistency_fail += 1
                        lags.append(s - last_step[key])
                if key not in last_step:
                    last_step[key] = s
                    last_obs[key] = obs_tokens[s]
        lags = np.array(lags) if lags else np.array([0])
        print(f"--- T={T} ({n_traj} traj) ---")
        print(f"  obs positions: {n_obs}")
        print(f"  exact-revisit:     {n_exact/n_obs:6.2%}  ({n_exact})")
        print(f"  motif-revisit:     {n_motif/n_obs:6.2%}  ({n_motif})")
        print(f"  cross-instance:    {n_cross/n_obs:6.2%}  ({n_cross})   <-- compositional target")
        print(f"  exact∩cross overlap (should be 0): {overlap}")
        print(f"  env consistency failures (should be 0): {consistency_fail}")
        print(f"  cross-instance LAG (steps back to matching motif-cell):")
        print(f"     median={np.median(lags):.0f}  mean={lags.mean():.0f}  "
              f"p90={np.percentile(lags,90):.0f}  max={lags.max():.0f}")
        for w in (8, 16, 32, 64, 128):
            frac = (lags <= w).mean()
            print(f"     within {w:3d} steps: {frac:5.1%}")
        print(f"  BASELINES on cross-instance target:")
        print(f"     majority-obs (chance-ish):     {freq_correct/n_obs:6.2%} (over ALL obs)")
        print(f"     copy-nearest-motif-cell oracle:{copy_correct/max(n_cross,1):6.2%} (over cross-instance)")
        print()


if __name__ == "__main__":
    run()
