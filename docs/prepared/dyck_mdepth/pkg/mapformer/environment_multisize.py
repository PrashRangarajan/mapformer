"""Multi-size, multi-environment GridWorld for cross-scale cognitive-map generalization.

Each training batch samples a trajectory from a random environment, where
the environment can be 32x32, 64x64, or 128x128 torus. All envs share the
same obs vocabulary (16 obs types + blank + n_landmarks), only their grid
size and content layout (obs_map, landmark positions) differs.

This is the cross-scale extension of MultiEnvGridWorld: tests whether the
cognitive-map machinery learns scale-invariant or scale-conditional strategies
across genuinely different env sizes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .environment import GridWorld


class MultiSizeMultiEnvGridWorld:
    """Mix of torus envs at different sizes; per-trajectory sample size + instance."""

    N_ACTIONS = GridWorld.N_ACTIONS

    def __init__(
        self,
        sizes=(32, 64, 128),
        n_envs_per_size: int = 20,
        n_test_envs_per_size: int = 20,
        n_obs_types: int = 16,
        p_empty: float = 0.5,
        n_landmarks: int = 200,
        seed: int = 0,
    ):
        self.sizes = list(sizes)
        self.n_envs_per_size = n_envs_per_size
        self.n_test_envs_per_size = n_test_envs_per_size
        self.n_obs_types = n_obs_types
        self.p_empty = p_empty
        self.n_landmarks = n_landmarks

        # Train env pool: n_envs_per_size envs per size
        self.train_envs_by_size = {}
        seed_offset = 1
        for size in sizes:
            envs = []
            for i in range(n_envs_per_size):
                env = GridWorld(size=size, n_obs_types=n_obs_types,
                               p_empty=p_empty, n_landmarks=n_landmarks,
                               seed=seed + seed_offset)
                envs.append(env)
                seed_offset += 1
            self.train_envs_by_size[size] = envs

        # Held-out test pool: different envs (much higher seeds)
        self.test_envs_by_size = {}
        seed_offset = 100000
        for size in sizes:
            envs = []
            for i in range(n_test_envs_per_size):
                env = GridWorld(size=size, n_obs_types=n_obs_types,
                               p_empty=p_empty, n_landmarks=n_landmarks,
                               seed=seed + seed_offset)
                envs.append(env)
                seed_offset += 1
            self.test_envs_by_size[size] = envs

        # Forward unified vocab (all envs share)
        sample_env = self.train_envs_by_size[sizes[0]][0]
        self.unified_vocab_size = sample_env.unified_vocab_size

    def generate_trajectory(
        self, n_steps: int = 128, train: bool = True, size: Optional[int] = None,
        p_transition_noise: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        """Sample a (size, env) pair from the pool, generate one trajectory.

        If `size` is None, sample uniformly across sizes. If `size` is set,
        sample only from that size (useful for size-specific eval)."""
        if rng is None: rng = np.random
        pool = self.train_envs_by_size if train else self.test_envs_by_size

        if size is None:
            # Sample size uniformly
            size = self.sizes[int(rng.randint(0, len(self.sizes)))]

        env = pool[size][int(rng.randint(0, len(pool[size])))]
        tokens, obs_mask, rev_mask = env.generate_trajectory(
            n_steps, p_transition_noise=p_transition_noise,
        )
        return tokens, obs_mask, rev_mask, size

    def generate_batch(
        self, batch_size: int, n_steps: int = 128, train: bool = True,
        size: Optional[int] = None,
        p_transition_noise: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        toks, oms, rms, sizes_used = [], [], [], []
        for _ in range(batch_size):
            t, om, rm, sz = self.generate_trajectory(
                n_steps, train=train, size=size,
                p_transition_noise=p_transition_noise, rng=rng,
            )
            toks.append(t); oms.append(om); rms.append(rm)
            sizes_used.append(sz)
        return (torch.stack(toks), torch.stack(oms), torch.stack(rms), sizes_used)
