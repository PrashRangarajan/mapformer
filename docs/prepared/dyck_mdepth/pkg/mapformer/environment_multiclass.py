"""Multi-class env infrastructure: pool torus + MiniGrid envs for cross-class training.

The cognitive-map paper's most ambitious test: train on env classes with
fundamentally different action vocabularies and obs spaces, see if the
architecture generalizes.

Disjoint vocab approach:
  Token 0: TORUS_PREFIX
  Token 1: DOORKEY_PREFIX
  Tokens 2-5: torus actions (4 cardinal)
  Tokens 6-12: MiniGrid actions (7)
  Tokens 13-28: torus obs (16 types) + 1 blank
  Tokens 29-94: MiniGrid obj_color obs (66 combinations)
  Tokens 95-294: torus landmarks (200)

  Total: 295 tokens.

Each trajectory starts with a class-prefix token telling the model which
env it's in. Then standard (action, obs) interleaved tokens.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .environment import GridWorld
from .minigrid_env import MiniGridWorld, N_OBJ_TYPES, N_COLORS


# Vocab layout for cross-class training
TORUS_PREFIX_TOKEN = 0
DOORKEY_PREFIX_TOKEN = 1
N_PREFIX_TOKENS = 2

TORUS_ACTION_OFFSET = N_PREFIX_TOKENS                     # 2..5  (4 actions)
TORUS_N_ACTIONS = 4

MINIGRID_ACTION_OFFSET = TORUS_ACTION_OFFSET + TORUS_N_ACTIONS  # 6..12 (7 actions)
MINIGRID_N_ACTIONS = 7

TORUS_OBS_OFFSET = MINIGRID_ACTION_OFFSET + MINIGRID_N_ACTIONS  # 13.. (16 obs + 1 blank)
TORUS_N_OBS = 16
TORUS_BLANK_TOKEN = TORUS_OBS_OFFSET + TORUS_N_OBS

MINIGRID_OBS_OFFSET = TORUS_BLANK_TOKEN + 1                # next free
MINIGRID_N_OBS = N_OBJ_TYPES * N_COLORS                    # 11 * 6 = 66

TORUS_LANDMARK_OFFSET = MINIGRID_OBS_OFFSET + MINIGRID_N_OBS
TORUS_N_LANDMARKS = 200

UNIFIED_VOCAB_SIZE = TORUS_LANDMARK_OFFSET + TORUS_N_LANDMARKS  # ~295


class MultiClassWorld:
    """Pool of envs: torus + DoorKey. Same unified vocab, env-class prefix token."""

    def __init__(
        self,
        n_torus_envs: int = 30,
        n_doorkey_envs: int = 30,
        n_torus_test: int = 30,
        n_doorkey_test: int = 30,
        torus_size: int = 64,
        torus_n_landmarks: int = 200,
        seed: int = 0,
    ):
        self.torus_envs = [
            GridWorld(size=torus_size, n_obs_types=TORUS_N_OBS, p_empty=0.5,
                     n_landmarks=torus_n_landmarks, seed=seed + 1 + i)
            for i in range(n_torus_envs)
        ]
        self.torus_test_envs = [
            GridWorld(size=torus_size, n_obs_types=TORUS_N_OBS, p_empty=0.5,
                     n_landmarks=torus_n_landmarks, seed=seed + 100000 + i)
            for i in range(n_torus_test)
        ]

        # MiniGrid envs are stateful and reset per trajectory; just store wrappers
        self.doorkey_train_seed_base = seed + 10000
        self.doorkey_test_seed_base = seed + 200000
        self.n_doorkey_envs = n_doorkey_envs
        self.n_doorkey_test = n_doorkey_test

        # Pre-construct DoorKey envs (each instance has its own random seed)
        self.doorkey_envs = [
            MiniGridWorld(env_name="MiniGrid-DoorKey-8x8-v0",
                          tokenization="obj_color",
                          seed=self.doorkey_train_seed_base + i)
            for i in range(n_doorkey_envs)
        ]
        self.doorkey_test_envs_objs = [
            MiniGridWorld(env_name="MiniGrid-DoorKey-8x8-v0",
                          tokenization="obj_color",
                          seed=self.doorkey_test_seed_base + i)
            for i in range(n_doorkey_test)
        ]

        self.unified_vocab_size = UNIFIED_VOCAB_SIZE

    def generate_torus_trajectory(self, env, n_steps: int = 128):
        """Wrap torus tokens into unified vocab. Returns (tokens, revisit_mask)."""
        tokens, _, revisit_mask = env.generate_trajectory(n_steps)
        # Original torus tokens: actions 0..3, obs 4..(4+16-1), blank=(4+16), landmarks=(4+16+1)..
        # Remap to unified vocab:
        unified = torch.zeros_like(tokens)
        for i, t in enumerate(tokens):
            ti = t.item()
            if ti < env.N_ACTIONS:  # action
                unified[i] = ti + TORUS_ACTION_OFFSET
            else:
                obs_rel = ti - env.N_ACTIONS  # 0..16 range for obs, blank, landmarks
                if obs_rel < TORUS_N_OBS:
                    unified[i] = obs_rel + TORUS_OBS_OFFSET
                elif obs_rel == TORUS_N_OBS:  # blank
                    unified[i] = TORUS_BLANK_TOKEN
                else:  # landmark
                    lm_idx = obs_rel - TORUS_N_OBS - 1
                    unified[i] = TORUS_LANDMARK_OFFSET + lm_idx
        # Prepend prefix token
        full = torch.cat([torch.tensor([TORUS_PREFIX_TOKEN]), unified])
        # Shift revisit_mask by 1 to account for prefix
        full_revisit = torch.cat([torch.zeros(1, dtype=torch.bool), revisit_mask])
        return full, full_revisit

    def generate_doorkey_trajectory(self, env, n_steps: int = 128):
        """Wrap DoorKey trajectory tokens into unified vocab."""
        tokens, _, revisit_mask = env.generate_trajectory(n_steps)
        # MiniGrid tokens: actions 0..(N_ACTIONS-1=6), obs (N_ACTIONS)..((N_ACTIONS+n_obs_types-1))
        unified = torch.zeros_like(tokens)
        for i, t in enumerate(tokens):
            ti = t.item()
            if ti < env.N_ACTIONS:
                unified[i] = ti + MINIGRID_ACTION_OFFSET
            else:
                obs_rel = ti - env.N_ACTIONS  # 0..(n_obs_types)
                if obs_rel < env.n_obs_types:
                    unified[i] = obs_rel + MINIGRID_OBS_OFFSET
                else:  # blank (n_obs_types)
                    # Reuse TORUS_BLANK_TOKEN for DoorKey blank too (both mean "no obs token")
                    unified[i] = TORUS_BLANK_TOKEN
        full = torch.cat([torch.tensor([DOORKEY_PREFIX_TOKEN]), unified])
        full_revisit = torch.cat([torch.zeros(1, dtype=torch.bool), revisit_mask])
        return full, full_revisit

    def generate_trajectory(
        self, n_steps: int = 128, train: bool = True, env_class: Optional[str] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        if rng is None: rng = np.random
        if env_class is None:
            env_class = "torus" if rng.random() < 0.5 else "doorkey"
        if env_class == "torus":
            pool = self.torus_envs if train else self.torus_test_envs
            env = pool[int(rng.randint(0, len(pool)))]
            tokens, rev = self.generate_torus_trajectory(env, n_steps)
        else:
            pool = self.doorkey_envs if train else self.doorkey_test_envs_objs
            env = pool[int(rng.randint(0, len(pool)))]
            tokens, rev = self.generate_doorkey_trajectory(env, n_steps)
        return tokens, rev, env_class

    def generate_batch(
        self, batch_size: int, n_steps: int = 128, train: bool = True,
        env_class: Optional[str] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        toks, revs, classes = [], [], []
        for _ in range(batch_size):
            t, r, c = self.generate_trajectory(n_steps, train=train,
                                                 env_class=env_class, rng=rng)
            toks.append(t); revs.append(r); classes.append(c)
        # Pad to same length (all should be same length given fixed n_steps)
        return torch.stack(toks), torch.stack(revs), classes
