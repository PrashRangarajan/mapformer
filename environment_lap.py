"""Lap task: same place, same observation, different lap. Ported from CSCG.

Source (George et al., bioRxiv 10.1101/864421v4, "Representation of paths and
temporal order" / Fig. 5), verbatim:

    "a rat runs four laps in a looping rectangular track before receiving a
     reward. A CSCG exposed to the same sequence learned to distinguish the laps
     and to predict the reward at the end of the 4th lap."

    "the agent is able to identify which lap it is in based on identical local
     observations."

Why this is the sharp test for MapFormer
----------------------------------------
MapFormer encodes position as theta = omega * cumsum(Delta(actions)). On a closed
loop, completing a circuit returns theta to (approximately) the same value -- by
design, since revisiting a place SHOULD produce the same code so attention can
match it. Our whole torus task depends on that.

The lap task inverts the requirement: lap-1-cell-c and lap-4-cell-c are the same
place, the same observation, and the same theta, and must be treated
DIFFERENTLY. CSCG gets this free by allocating a different clone per lap; a
continuous point-estimate position code has no capacity for it.

Two-sided prediction, worth pre-registering:
  - MapFormer FAILS  -> we have located a concrete boundary of the position-code
                        approach: it cannot represent history beyond position.
  - MapFormer WINS   -> it is counting laps through CONTENT attention over the
                        token history, and the SO(2) machinery is not doing the
                        work. Also worth knowing.

Deviation from CSCG, stated not buried
--------------------------------------
CSCG trains on ONE fixed sequence (they model a specific rat experiment). Copied
directly that is pure memorisation -- the episode is deterministic, so a
sufficiently high-order n-gram solves it. To make it an IN-CONTEXT task the
observation assignment is redrawn every episode, so the model must count laps
from the current context rather than memorise one track.

THE DESIGN TENSION, and how it is resolved
------------------------------------------
A 1-D loop with a single forward action fails BOTH ways:
  fixed loop_len    -> reward always lands at token index K*loop_len, so a model
                       predicts it from POSITION ALONE. Gate measured this at
                       exactly **1.000**. Void.
  variable loop_len -> kills that shortcut, but omega*loop_len is then not a
                       multiple of 2*pi, so theta NEVER wraps -- it accumulates
                       monotonically and is a distance-travelled counter, from
                       which lap number falls out free. Measured: Vanilla scored
                       exact **0.828** (random floor 0.250) on that version. The
                       task was not testing the claim.

RESOLUTION: walk a CLOSED RECTANGULAR CIRCUIT on the torus -- right w, down h,
left w, up h. Net displacement per lap is EXACTLY ZERO, so

    theta_{t + loop_len} = theta_t   for every t, independent of omega and of
                                     grid_size,

because theta is omega * cumsum(Delta) and the true displacement sums to zero
around the circuit. Meanwhile w,h are redrawn per episode, so the loop length
2(w+h) varies and the positional shortcut stays dead.

IMPORTANT CONDITION, measured not assumed: theta returns per lap only if the
model has learned FAITHFUL path integration -- Delta(action token) = the true
action displacement, and Delta(observation token) = 0. `action_to_lie` maps token
EMBEDDINGS to Delta, so at random init observation tokens also displace and theta
drifts (measured 3.33 rad between lap 1 and lap 4 at init). That is not a flaw in
the task; it is the bind the task creates:

    to count laps, the model must make the per-lap Delta sum NON-ZERO,
    i.e. it must BREAK faithful path integration.

So the interesting measurement is not just accuracy but the per-lap theta drift
of the TRAINED model. High accuracy WITH near-zero drift would mean lap counting
happened in content attention; high accuracy WITH large drift means the model
abandoned path integration to get it. `probe_lap_theta.py` measures this.
"""
import numpy as np
import torch


class LapWorld:
    """Closed rectangular circuit on a torus, traversed `n_laps` times."""

    N_ACTIONS = 4
    # must match environment.GridWorld so the learned action semantics transfer
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, n_obs_types: int = 40, n_laps: int = 4, size: int = 64,
                 wh_range=(3, 8), fixed_loop: bool = False, seed: int = 0):
        self.n_obs_types = n_obs_types
        self.n_laps = n_laps
        self.size = size
        self.wh_range = wh_range
        self.fixed_loop = fixed_loop
        self.seed = seed
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.reward_tok = self.N_ACTIONS + n_obs_types
        self.unified_vocab_size = self.reward_tok + 1

    def _circuit(self, rng):
        """Action sequence for one closed lap; net displacement is exactly zero."""
        lo, hi = self.wh_range
        if self.fixed_loop:
            w = h = lo
        else:
            w = int(rng.randint(lo, hi + 1)); h = int(rng.randint(lo, hi + 1))
        # right x w, down x h, left x w, up x h  ->  sums to (0, 0)
        return [3] * w + [1] * h + [2] * w + [0] * h

    def generate_lap_episode(self, rng=None):
        """One episode: K laps of a fresh track, reward token at the very end.

        Returns tokens, the K decision positions (one per lap boundary), the
        binary label at each (1 = REWARD is correct here), and info.
        """
        if rng is None:
            rng = np.random
        circuit = self._circuit(rng)
        L = len(circuit)
        K = self.n_laps
        # unique observation per cell WITHIN a lap, so the only aliasing that
        # matters is ACROSS laps -- detecting a boundary is easy, counting is not
        assert self.n_obs_types >= L, "need n_obs_types >= loop_len for distinct cells"
        cell_obs = rng.choice(self.n_obs_types, size=L, replace=False)

        tokens: list[int] = []
        dec_pos: list[int] = []      # index whose NEXT token is the decision
        dec_label: list[int] = []    # 1 if that next token is REWARD
        for lap in range(1, K + 1):
            for c in range(L):
                tokens.append(circuit[c] + self.action_offset)
                last_of_lap = (c == L - 1)
                if last_of_lap:
                    # the model predicts the lap-boundary token FROM here
                    dec_pos.append(len(tokens) - 1)
                    dec_label.append(1 if lap == K else 0)
                if last_of_lap and lap == K:
                    tokens.append(self.reward_tok)
                else:
                    tokens.append(int(cell_obs[c]) + self.obs_offset)

        info = {"loop_len": L, "n_laps": K, "cell_obs": cell_obs,
                "circuit": circuit, "reward_index": len(tokens) - 1}
        return (torch.tensor(tokens, dtype=torch.long),
                dec_pos, dec_label, info)

    def generate_lap_batch(self, batch_size: int, rng=None):
        """Batch of episodes. Variable loop_len => variable length, so pad."""
        eps = [self.generate_lap_episode(rng) for _ in range(batch_size)]
        maxlen = max(e[0].shape[0] for e in eps)
        toks = torch.full((batch_size, maxlen), self.obs_offset, dtype=torch.long)
        valid = torch.zeros(batch_size, maxlen, dtype=torch.bool)
        for i, (t, _dp, _dl, _in) in enumerate(eps):
            toks[i, : t.shape[0]] = t
            valid[i, : t.shape[0]] = True
        return (toks, valid, [e[1] for e in eps], [e[2] for e in eps],
                [e[3] for e in eps])
