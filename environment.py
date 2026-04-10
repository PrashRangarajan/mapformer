"""
2D Grid environment for MapFormer training and evaluation.

Generates (action, observation) trajectories on a grid world.
Non-unique observations are critical: n_obs_types < grid_size^2
ensures the model must do genuine path integration rather than
simply memorising observation->location mappings.
"""

import torch
import numpy as np
from typing import Optional


class GridWorld:
    """Simple 2D grid world with non-unique observations.

    Actions: 0=North(-y), 1=South(+y), 2=West(-x), 3=East(+x)
    Observations are integer types assigned randomly to cells.
    """

    N_ACTIONS = 4
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, size: int = 10, n_obs_types: int = 4, seed: Optional[int] = None):
        """
        Args:
            size: Grid is size x size
            n_obs_types: Number of distinct observation types.
                Must be < size*size for non-unique observations.
            seed: Random seed for reproducible observation maps
        """
        self.size = size
        self.n_obs_types = n_obs_types

        rng = np.random.RandomState(seed)
        self.obs_map = torch.from_numpy(
            rng.randint(0, n_obs_types, (size, size))
        ).long()

        # Track trajectory for interpretability
        self.visited_locations: list[tuple[int, int]] = []
        self.last_x = size // 2
        self.last_y = size // 2

    def generate_trajectory(
        self, T: int = 64, start: Optional[tuple[int, int]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a random walk trajectory.

        Args:
            T: Trajectory length
            start: Starting position (row, col). Defaults to center.

        Returns:
            actions: (T,) long tensor of action indices
            observations: (T,) long tensor of observation types
        """
        if start is not None:
            x, y = start
        else:
            x, y = self.size // 2, self.size // 2

        actions = []
        obs = []
        self.visited_locations = []

        for _ in range(T):
            a = torch.randint(0, self.N_ACTIONS, ()).item()
            dx, dy = self.ACTION_DELTAS[a]
            x = max(0, min(self.size - 1, x + dx))
            y = max(0, min(self.size - 1, y + dy))
            actions.append(a)
            obs.append(self.obs_map[x, y].item())
            self.visited_locations.append((x, y))

        self.last_x = x
        self.last_y = y
        return torch.tensor(actions, dtype=torch.long), torch.tensor(obs, dtype=torch.long)

    def generate_batch(
        self, batch_size: int, T: int = 64
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
        """Generate a batch of trajectories.

        Returns:
            actions: (batch_size, T)
            observations: (batch_size, T)
            all_locations: list of location lists for each trajectory
        """
        all_actions = []
        all_obs = []
        all_locations = []

        for _ in range(batch_size):
            a, o = self.generate_trajectory(T)
            all_actions.append(a)
            all_obs.append(o)
            all_locations.append(list(self.visited_locations))

        return (
            torch.stack(all_actions),
            torch.stack(all_obs),
            all_locations,
        )
