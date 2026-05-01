"""MiniGrid environment adapter for MapFormer.

Exposes the same `generate_trajectory()` interface as `environment.GridWorld`
so our train.py / train_variant.py / eval scripts work without modification.

Vocabulary layout (mirrors environment.py):
    actions:           0..N_ACTIONS-1                    (N_ACTIONS=7)
    obs_offset:        7
    aliased obs:       7 + 0..n_obs_types-1             (vocab IDs 7..7+K-1)
    blank token:       7 + n_obs_types                  (the "no-object" cell)
    landmark IDs:      7 + n_obs_types+1 onward         (unique-cell tokens)

We tokenize ONLY the cell *directly in front of the agent* (image[3, 2] in
MiniGrid's coords). This collapses the 7×7×3 image to a single discrete
token while preserving the most informative cell — what's about to be
interacted with.

Two tokenization granularities:
  - "obj_only" (default, 11 types): just obj_idx (empty/wall/floor/door/key/
    ball/box/goal/lava/agent/unseen). Simple, matches our 16-obs-types scale.
  - "obj_color" (~36 types): obj_idx * 6 + color_idx. Richer.
  - "full" (~200 types): obj_idx * 6 * 4 + color_idx * 4 + state_idx.
    Closest to a cell-identifying observation but high-entropy.

Trajectories are generated with a *random policy* by default (matches the
original task's random walks). For more realistic data, swap in MiniGrid's
BotAgent or a learned policy.

Compatible with:
  - p_action_noise: per-step probability of replacing the action with a
    random one (matches train.py's noise injection)
  - revisit_mask: based on (x, y, direction) tuple; same tuple = revisit

Limitations:
  - Empty-8x8 / DoorKey-8x8 / KeyCorridor / ObstructedMaze have different
    structures; this wrapper handles all of them via the same MiniGrid API
  - Agent direction matters for what's "in front" — same (x, y) different
    direction sees different obs. Tracking by (x, y, direction) for revisit.
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
import minigrid

# MiniGrid object/color/state index ranges (from minigrid.core.constants)
N_OBJ_TYPES   = 11   # empty, wall, floor, door, key, ball, box, goal, lava, agent, unseen
N_COLORS      = 6    # red, green, blue, purple, yellow, grey
N_STATES      = 4    # door states: open/closed/locked + open object state padding


class MiniGridWorld:
    """Adapter exposing MapFormer-compatible trajectory generation on MiniGrid."""

    N_ACTIONS = 7

    def __init__(
        self,
        env_name: str = "MiniGrid-Empty-8x8-v0",
        tokenization: str = "obj_only",
        seed: int = 0,
        max_episode_steps: int = 1000,
    ):
        self.env_name = env_name
        self.tokenization = tokenization
        self.seed = seed

        # Gymnasium env construction
        self.env = gym.make(env_name, max_episode_steps=max_episode_steps,
                            render_mode=None)
        self.env.reset(seed=seed)

        # Vocab sizing per tokenization
        if tokenization == "obj_only":
            self.n_obs_types = N_OBJ_TYPES
        elif tokenization == "obj_color":
            self.n_obs_types = N_OBJ_TYPES * N_COLORS
        elif tokenization == "full":
            self.n_obs_types = N_OBJ_TYPES * N_COLORS * N_STATES
        else:
            raise ValueError(f"Unknown tokenization {tokenization!r}")

        # Conventions matching environment.py
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS                     # = 7
        self.blank_token = self.n_obs_types                  # the "no info" obs
        self.unified_blank = self.obs_offset + self.blank_token

        # Unified vocab: actions + obs (n_obs_types + blank). No landmarks
        # in MiniGrid's basic envs; if needed we'd add unique-cell tokens
        # downstream.
        self.unified_vocab_size = self.N_ACTIONS + self.n_obs_types + 1

        # No "true" grid_size (MiniGrid envs vary), but we record the env's
        # grid for revisit tracking. After .reset(), env.unwrapped.grid is
        # available.
        self.size = self.env.unwrapped.grid.width
        # Some attributes for compatibility with our existing eval scripts
        # that read these from GridWorld
        self.n_landmarks = 0
        self.first_landmark_rel = self.n_obs_types + 1
        self.p_empty = 0.5  # not really meaningful here, but kept for API

    def _encode_cell(self, cell: np.ndarray) -> int:
        """cell is the (3,) array (obj_idx, color_idx, state_idx)."""
        obj, color, state = int(cell[0]), int(cell[1]), int(cell[2])
        if self.tokenization == "obj_only":
            return obj
        if self.tokenization == "obj_color":
            return obj * N_COLORS + color
        # full
        return obj * N_COLORS * N_STATES + color * N_STATES + state

    def _front_cell_token(self, obs: dict) -> int:
        """Extract the obs token from the cell directly in front of the agent.

        MiniGrid observation layout: image[7, 7, 3] is the agent's egocentric
        view, oriented so image[3, 2] is the cell directly in front. (image
        is rendered from agent's perspective; row 6 = agent's row, row 5 = one
        ahead, etc. The center column is 3.)
        """
        # The cell IN FRONT of the agent in egocentric view is image[3, 5]
        # in MiniGrid's standard 7x7 layout (agent at [3, 6], facing up).
        # Reference: minigrid.core.world_object.WorldObj.encode and the env's
        # gen_obs_grid function — image is "ahead of agent" oriented.
        cell = obs["image"][3, 5]
        return self._encode_cell(cell)

    # Forward-biased action distribution: mimics torus "directed walk".
    # Most steps go forward; turn occasionally; pickup/drop/toggle/done rare.
    # MiniGrid actions: 0=left turn, 1=right turn, 2=forward, 3=pickup,
    # 4=drop, 5=toggle, 6=done.
    FORWARD_BIASED_PROBS = np.array(
        [0.15, 0.15, 0.65, 0.02, 0.01, 0.01, 0.01]
    )
    UNIFORM_PROBS = np.ones(7) / 7

    def _sample_action(self, policy: str) -> int:
        if policy == "uniform":
            return int(np.random.choice(7))
        if policy == "forward_biased":
            return int(np.random.choice(7, p=self.FORWARD_BIASED_PROBS))
        raise ValueError(f"Unknown policy {policy!r}")

    def generate_trajectory(
        self,
        n_steps: int = 128,
        p_action_noise: float = 0.0,
        policy: str = "forward_biased",
    ):
        """Generate an interleaved trajectory of (action, obs) tokens.

        Args:
            n_steps: number of (action, obs) pairs.
            p_action_noise: probability of replacing the chosen action with
                a UNIFORM-random one. Meaningful only when ``policy`` has
                structure (i.e., not "uniform").
            policy: base policy. "forward_biased" gives structured directed
                walks (most steps forward, occasional turns); "uniform"
                samples uniformly over all 7 actions (action noise becomes
                a no-op at p_action_noise > 0).

        Returns:
            tokens:        (2*n_steps,) long tensor in unified vocab
            obs_mask:      (2*n_steps,) bool, True at obs positions
            revisit_mask:  (2*n_steps,) bool, True at obs positions whose
                           (x, y, direction) tuple was seen before.
        """
        obs, info = self.env.reset(seed=self.seed + np.random.randint(1_000_000))
        tokens = []
        is_revisit = []
        seen = set()

        for t in range(n_steps):
            action = self._sample_action(policy)
            if p_action_noise > 0 and np.random.random() < p_action_noise:
                # Action noise replaces with a UNIFORM-random action (so
                # the perturbation is independent of the base policy).
                action = int(np.random.choice(7))
            tokens.append(action + self.action_offset)

            obs, reward, terminated, truncated, info = self.env.step(action)

            agent_pos = tuple(self.env.unwrapped.agent_pos)
            agent_dir = int(self.env.unwrapped.agent_dir)
            state_key = agent_pos + (agent_dir,)
            front_token = self._front_cell_token(obs)
            tokens.append(front_token + self.obs_offset)
            is_revisit.append(state_key in seen)
            seen.add(state_key)

            if terminated or truncated:
                obs, info = self.env.reset(
                    seed=self.seed + np.random.randint(1_000_000)
                )

        tokens = torch.tensor(tokens, dtype=torch.long)
        obs_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        obs_mask[1::2] = True
        revisit_mask = torch.zeros(2 * n_steps, dtype=torch.bool)
        for step_idx, rev in enumerate(is_revisit):
            if rev:
                revisit_mask[2 * step_idx + 1] = True
        return tokens, obs_mask, revisit_mask
