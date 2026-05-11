"""Multi-environment GridWorld for cognitive-map generalization.

Holds a pool of K training environments (each with its own obs_map +
landmark positions) and L held-out test environments. Each training
trajectory is sampled from a random training env. The model never sees
test envs during training.

The cognitive-map claim: a model with the right inductive bias should
LEARN A META-STRATEGY ("explore a random env's structure, then use it
for revisit prediction"), not memorise per-env layouts. Standard
transformers (RoPE / LSTM) have no spatial inductive bias and should
fail at this generalization. MapFormer's path-integration is content-
agnostic, so the same machinery applies to any env.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .environment import GridWorld


class MultiEnvGridWorld:
    """Wraps a pool of GridWorld instances, samples per trajectory."""

    N_ACTIONS = GridWorld.N_ACTIONS

    def __init__(
        self,
        size: int = 64,
        n_obs_types: int = 16,
        p_empty: float = 0.5,
        n_landmarks: int = 200,
        n_train_envs: int = 50,
        n_test_envs: int = 50,
        seed: int = 0,
    ):
        self.size = size
        self.n_obs_types = n_obs_types
        self.p_empty = p_empty
        self.n_landmarks = n_landmarks
        self.n_train_envs = n_train_envs
        self.n_test_envs = n_test_envs
        # All envs share the same hyperparams but have different seeds
        # → different obs_maps + landmark cells.
        self.train_envs = [
            GridWorld(size=size, n_obs_types=n_obs_types, p_empty=p_empty,
                      n_landmarks=n_landmarks, seed=seed + 1 + i)
            for i in range(n_train_envs)
        ]
        self.test_envs = [
            GridWorld(size=size, n_obs_types=n_obs_types, p_empty=p_empty,
                      n_landmarks=n_landmarks, seed=seed + 100000 + i)
            for i in range(n_test_envs)
        ]
        # Forward unified vocab from any env (they all share the layout)
        self.unified_vocab_size = self.train_envs[0].unified_vocab_size

    def generate_trajectory(
        self, n_steps: int = 128, train: bool = True,
        p_transition_noise: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        """Sample a random env from the pool, generate one trajectory."""
        pool = self.train_envs if train else self.test_envs
        idx = int((rng or np.random).randint(0, len(pool)))
        env = pool[idx]
        tokens, obs_mask, rev_mask = env.generate_trajectory(
            n_steps, p_transition_noise=p_transition_noise,
        )
        return tokens, obs_mask, rev_mask, idx  # idx returned for diagnostic

    def generate_batch(
        self, batch_size: int, n_steps: int = 128, train: bool = True,
        p_transition_noise: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        toks, oms, rms = [], [], []
        for _ in range(batch_size):
            t, om, rm, _ = self.generate_trajectory(
                n_steps, train=train, p_transition_noise=p_transition_noise, rng=rng,
            )
            toks.append(t); oms.append(om); rms.append(rm)
        return (torch.stack(toks), torch.stack(oms), torch.stack(rms),
                [None] * batch_size)  # 4th return matches GridWorld.generate_batch sig
