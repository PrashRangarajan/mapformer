"""Family-tree task: a NON-COMMUTATIVE relational structure, not a space.

Why this task exists
--------------------
MapFormer's own appendix (B.2.2) motivates non-commutative groups with exactly
this example, verbatim:

    "The ability to learn non commutative rules is fundamental, since rules as
     simple as 'mother' and 'father' should not commute. Indeed, the father of my
     mother is my grand father on my mother's side, while the mother of my father
     is my grand mother on my father's side, hence ending in a different position
     within the family tree/graph. This implies that the learned matrices do not
     commute, i.e.: W_mother W_father != W_father W_mother."

They then validate MapEM-NC on synthetic **4D rotations**, NOT on a family tree:

    "we reproduce our navigation task, where this time actions represent 4D
     rotations and not translations anymore ... we train our models on sequences
     of length 16, and test on length 32."

So the motivating example and the validation are never connected. This is that
family tree. TEM (Whittington et al. 2020) runs the analogous "social hierarchy"
task with relational actions ('child of', 'grandparent of', 'sibling of').

Structure
---------
A person is a path of relations from ego, e.g. "mf" = my mother's father. The
ancestor tree to depth D has 2^(D+1)-1 nodes. Actions:

    0 mother              s -> s+"m"
    1 father              s -> s+"f"
    2 child               s -> s[:-1]
    3 grandmother (mat.)  s -> s+"mm"
    4 grandfather (mat.)  s -> s+"mf"     <- the paper's example
    5 grandmother (pat.)  s -> s+"fm"     <- and its non-commuting partner
    6 grandfather (pat.)  s -> s+"ff"
    7 grandchild          s -> s[:-2]

Non-commutativity is exact and structural: mother-then-father lands on "mf",
father-then-mother lands on "fm", which are different people. No 2-D translation
group can represent this, which is the whole point.

Only VALID actions are sampled at each state (no null moves), so every recorded
action is executed and there is no "stay" signal to exploit.

Task: each node carries an observation drawn per episode from n_obs types
(aliased -- far fewer types than nodes). The agent walks; the model predicts the
observation on arrival, scored ONLY at REVISITED nodes, exactly as in the paper's
navigation task. Getting it right requires knowing you have returned to a
specific person, which on this graph cannot be done by summing translations.
"""
import numpy as np
import torch

RELATIONS = [
    ("mother", "append", "m"),
    ("father", "append", "f"),
    ("child", "drop", 1),
    ("grandmother_maternal", "append", "mm"),
    ("grandfather_maternal", "append", "mf"),
    ("grandmother_paternal", "append", "fm"),
    ("grandfather_paternal", "append", "ff"),
    ("grandchild", "drop", 2),
]


class FamilyTreeWorld:
    """Ancestor tree to depth D, traversed by relational (non-commuting) actions."""

    N_ACTIONS = len(RELATIONS)

    def __init__(self, depth: int = 5, n_obs_types: int = 8, seed: int = 0):
        self.depth = depth
        self.n_obs_types = n_obs_types
        self.seed = seed
        self.action_offset = 0
        self.obs_offset = self.N_ACTIONS
        self.unified_vocab_size = self.N_ACTIONS + n_obs_types
        self.nodes = self._all_nodes()
        self.node_index = {n: i for i, n in enumerate(self.nodes)}

    def _all_nodes(self):
        out = [""]
        frontier = [""]
        for _ in range(self.depth):
            nxt = []
            for s in frontier:
                for c in "mf":
                    nxt.append(s + c)
            out.extend(nxt); frontier = nxt
        return out

    def _apply(self, s, a):
        """Return the node reached by relation `a` from `s`, or None if invalid."""
        _name, kind, arg = RELATIONS[a]
        if kind == "append":
            t = s + arg
            return t if len(t) <= self.depth else None
        return s[:-arg] if len(s) >= arg else None

    def _valid(self, s):
        return [a for a in range(self.N_ACTIONS) if self._apply(s, a) is not None]

    def generate_episode(self, n_steps: int = 128, rng=None):
        """Walk the tree; predict the observation on arrival, scored at revisits."""
        if rng is None:
            rng = np.random
        # observations redrawn per episode -> the model must build the map
        # IN CONTEXT rather than memorise one tree
        obs = rng.randint(0, self.n_obs_types, size=len(self.nodes))

        s = ""
        tokens, revisit = [], []
        seen = set([s])
        scored = set()          # DEDUP: each node scored at most once per episode
        for _ in range(n_steps):
            valid = self._valid(s)
            a = int(valid[rng.randint(len(valid))])
            s = self._apply(s, a)
            tokens.append(a + self.action_offset); revisit.append(False)
            tokens.append(int(obs[self.node_index[s]]) + self.obs_offset)
            # Dedup is load-bearing. Relations invert (mother then child is the
            # identity), so the walk oscillates and the same node's observation
            # repeats back to back. Without dedup the answer n-gram hits 0.333 at
            # order 2 against a chance of 0.125, and last-observation 0.192.
            score = (s in seen) and (s not in scored)
            if score:
                scored.add(s)
            revisit.append(score)
            seen.add(s)
        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(revisit, dtype=torch.bool),
                {"n_nodes": len(self.nodes), "n_distinct_visited": len(seen)})

    def generate_batch(self, batch_size: int, n_steps: int = 128, rng=None):
        eps = [self.generate_episode(n_steps, rng) for _ in range(batch_size)]
        return (torch.stack([e[0] for e in eps]),
                torch.stack([e[1] for e in eps]),
                [e[2] for e in eps])
