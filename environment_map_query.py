"""Map-Query task: goal-directed readout that cannot be solved by sequence continuation.

Why this exists
---------------
The hier-goal task was invalidated twice by the SAME structural flaw: it scored
next-action prediction on planner demonstrations, and a planner's output is a
structured sequence that is predictable from itself. Measured with n-gram models
on the action stream alone, no model involved:

    raw BFS path        order-1 0.969
    interleaved path    order-1 0.320  but  order-3 0.971

Fixing order 1 created order 3. Closed-loop success for every trained variant was
0.013-0.037 against a random floor of 0.010 -- nothing navigational was learned.

The fix is not another scrambling trick. It is to stop scoring a SEQUENCE.

Design
------
Each episode is:

    [ explore: (a, o) x T_explore ]  then  K query blocks

and each query block is three tokens:

    room_tok_i , local_tok_i , MASK

The model is scored on ONE token per query: its prediction at the position of
`local_tok_i`, which must be the first action of a shortest path from the agent's
current (end-of-explore) position to the queried goal.

Two properties do the work:

1. ONE scored token per query, and goals are drawn i.i.d., so an n-gram over the
   answer stream is at chance BY CONSTRUCTION -- not by luck, and not by a
   scrambling rule that a higher-order model can invert.
2. The answer is a function of (end-of-explore position, goal). Position is only
   available by integrating the explore actions; the goal is given. So
   randomising the goal, or shuffling the explore actions, MUST destroy accuracy.
   Those are no longer post-hoc ablations, they are what the task is.

The answer slot is fed back as MASK, never as the true action. Without this a
model could triangulate its position from a few (goal, answer) pairs and skip
path integration entirely -- a real leak, so the scored answers never enter the
context.

Scoring note: on a torus, several first actions can be simultaneously optimal
(when both axes need travel). A prediction counts as correct if it is ANY optimal
first action; scoring against one canonical choice would distort both chance and
ceiling. Chance is therefore NOT 0.25 and must be measured empirically -- see
`validate_map_query.py`, which reports the random-policy rate.

The start cell is FIXED so that absolute position is inferable at all. That same
choice concentrates the position distribution for small T_explore, which would
let the goal token alone predict the answer -- the next shortcut in this family.
T_explore must therefore be chosen from the measured positional entropy, not
picked. `validate_map_query.py` reports it.
"""
from typing import Optional

import numpy as np
import torch

from .environment import GridWorld


def optimal_first_actions(pos, goal, size):
    """Set of actions that strictly reduce torus L1 distance to the goal."""
    def d(p, g):
        return min((p[0] - g[0]) % size, (g[0] - p[0]) % size) + \
               min((p[1] - g[1]) % size, (g[1] - p[1]) % size)
    cur = d(pos, goal)
    out = set()
    for a, (dx, dy) in GridWorld.ACTION_DELTAS.items():
        nxt = ((pos[0] + dx) % size, (pos[1] + dy) % size)
        if d(nxt, goal) < cur:
            out.add(a)
    return out


class MapQueryGridWorld(GridWorld):
    """Explore then answer K independent one-token goal-direction queries."""

    def __init__(self, size: int = 64, room_size: int = 8, n_obs_types: int = 16,
                 p_empty: float = 0.5, seed: int = 0, start=(0, 0)):
        super().__init__(size=size, n_obs_types=n_obs_types, p_empty=p_empty,
                         n_landmarks=0, seed=seed)
        assert size % room_size == 0
        self.room_size = room_size
        self.rooms_per_side = size // room_size
        self.n_rooms = self.rooms_per_side ** 2
        self.n_local = room_size ** 2
        self.start = start
        base = self.N_ACTIONS + self.obs_vocab_size
        self.room_tok0 = base
        self.local_tok0 = base + self.n_rooms
        self.mask_tok = base + self.n_rooms + self.n_local
        self.unified_vocab_size = self.mask_tok + 1

    def room_local_to_cell(self, room: int, local: int):
        rri, rrj = divmod(room, self.rooms_per_side)
        lx, ly = divmod(local, self.room_size)
        return rri * self.room_size + lx, rrj * self.room_size + ly

    def _draw_obs(self, rng):
        obs = np.full((self.size, self.size), self.blank_token, dtype=np.int64)
        occ = rng.random((self.size, self.size)) >= self.p_empty
        obs[occ] = rng.randint(0, self.n_obs_types, occ.sum())
        return obs

    def generate_query_episode(self, T_explore: int = 256, n_queries: int = 8,
                               rng=None):
        if rng is None:
            rng = np.random
        obs_map = self._draw_obs(rng)
        x, y = self.start
        tokens: list[int] = []

        # ---- explore: directed random walk (paper's run-length 1..10) ----
        t = 0
        while t < T_explore:
            a = int(rng.randint(0, self.N_ACTIONS))
            k = int(rng.randint(1, 11))
            dx, dy = self.ACTION_DELTAS[a]
            for _ in range(k):
                if t >= T_explore:
                    break
                x = (x + dx) % self.size
                y = (y + dy) % self.size
                tokens.append(a + self.action_offset)
                tokens.append(int(obs_map[x, y]) + self.obs_offset)
                t += 1

        # ---- queries: (room, local, MASK); answer never enters the context ----
        score_pos: list[int] = []
        answers: list[set] = []
        for _ in range(n_queries):
            room = int(rng.randint(0, self.n_rooms))
            local = int(rng.randint(0, self.n_local))
            gx, gy = self.room_local_to_cell(room, local)
            tokens.append(self.room_tok0 + room)
            # prediction made AT the local token scores the answer
            score_pos.append(len(tokens))
            tokens.append(self.local_tok0 + local)
            tokens.append(self.mask_tok)
            answers.append(optimal_first_actions((x, y), (gx, gy), self.size))

        full = torch.tensor(tokens, dtype=torch.long)
        info = {"end_pos": (x, y), "T_explore": T_explore,
                "score_pos": score_pos, "answers": answers}
        return full, score_pos, answers, info

    def generate_query_batch(self, batch_size: int, T_explore: int = 256,
                             n_queries: int = 8, rng=None):
        toks, sps, ans, infos = [], [], [], []
        for _ in range(batch_size):
            t, sp, a, info = self.generate_query_episode(T_explore, n_queries, rng)
            toks.append(t); sps.append(sp); ans.append(a); infos.append(info)
        return torch.stack(toks), sps, ans, infos
