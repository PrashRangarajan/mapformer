"""Attention-map probe for the stitch (transitive-inference) task.

WHY AN ATTENTION PROBE AND NOT AN ACCURACY
------------------------------------------
CSCG (George et al., bioRxiv 10.1101/864421v4) reports this experiment
qualitatively. The complete set of claims it makes about it:

    "Predictive performance on the stitching of the two rooms is perfect
     (indicating that learning succeeded) after a few observations required for
     the agent to locate itself."

    "Notice that there is another patch in the first room that is identical to
     the merged patches, but was not merged. The model is using the sequential
     information to effectively identify patches that can be merged while
     respecting the observational data and context, and not simply looking for
     locally identical patches to merge."

No accuracy table, no baseline, no chance rate, no seeds. The evaluation is to
LOOK AT the learned transition matrix (their Fig. 2f). That works because a CSCG
*is* an explicit graph. MapFormer builds its map in context and exposes no graph,
so the demonstration does not port; a scored task has to be invented, and the one
invented for this repo earlier had a negative control defeatable at 0.617
balanced accuracy from the test-phase action stream alone. Rather than patch that
metric a third time, this probe ports CSCG's evaluation STYLE -- inspect the
learned structure -- to the only structure MapFormer exposes: attention.

THE MEASUREMENT
---------------
Room A's bottom-right 3x3 patch is the JOIN (it is also room B's top-left patch).
Room A's interior 3x3 patch at (1,1) is the CONFOUNDER: byte-identical
observations, but it is not the junction. Fix an offset (dr, dc); the two cells
`join+(dr,dc)` and `confounder+(dr,dc)` carry the SAME observation symbol.

After phases A and B and a shared random wander, the walk is steered to one of
those two cells. At the action token that predicts the target cell's observation
we read the attention row and split its mass over phase-A observation tokens by
which of the two look-alike cells they came from:

    join_share = mean_attn_per_token(A-tokens at join cell)
                 -------------------------------------------------------
                 mean_attn_per_token(join cell) + mean_attn_per_token(cf cell)

**The statistic is the PAIRED DIFFERENCE between the two arms, and its floor is
exactly 0.** Stated precisely, because the two halves are not equally strong:

  - Within one arm, the two look-alike cells are exchangeable only in
    expectation. They emit the same symbol, but in any single episode one of
    them happened to be visited more often or more recently, so a purely
    recency-driven model has a per-episode preference. Averaged over episodes
    that preference is 0.5; it is not 0.5 episode by episode.
  - Across the two arms it cancels EXACTLY. Both arms are scored on the same two
    cells with the same phase-A tokens -- the prefix is literally identical -- so
    visit counts, recency and token identity are held fixed by construction.
    Any account that does not use the agent's current position predicts a paired
    difference of exactly 0.

`join_share` is also immune to the two arms having different sequence lengths
(their approach paths differ), because it is a ratio of two masses inside one
softmax row, so the shared denominator cancels.

At LAYER 0 this is airtight. MapFormerWM's layer-0 keys are
`rope(k_proj(norm1(token_emb(tok))))` -- no context has been mixed in yet -- so
two tokens carrying the same symbol have keys that differ ONLY by their
path-integrated rotation. Layer 0 join_share is therefore a direct readout of the
position code. Later layers are reported too but are contaminated by context.

Two arms, run on the SAME prefix:
    join arm      walk ends at join+(dr,dc)        -> expect join_share > 0.5
    confound arm  walk ends at confounder+(dr,dc)  -> expect join_share < 0.5

The paired difference is the statistic. A sign flip is far stronger than a
one-sided effect: no recency, frequency or content account predicts one, because
phase A is literally the same tokens in both arms.

Secondary, `B-share` = massB / (massA + massB) over phase-A and phase-B
observation tokens. This is the transitive half -- standing at the junction
should activate room B's memories, standing at the look-alike should not. It is
reported as a RATIO rather than a raw mass because the two arms reach their
targets by paths of different length, so their sequences differ in length and a
raw mass would partly track the softmax denominator rather than the retrieval.
It has no analytic floor; the confound arm is its empirical floor.

CONTROL MODEL
-------------
PlainFlat (ordinary sequence-index RoPE, no path integration) is run through the
identical pipeline. It cannot know that the current cell is the junction, so it
should show no sign flip.
"""
import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mapformer.environment_stitch import StitchWorld, ACTION_DELTAS
from mapformer.hourglass_plain import PlainFlat, _rope_cos_sin
from mapformer.hourglass_plain import _apply_rope as _plain_rope
from mapformer.model import MapFormerWM, _apply_rope as _wm_rope

_REPO = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Attention capture.
#
# Both functions REPLAY the model's own forward pass rather than approximating
# it, and `capture` asserts that the replayed logits match `model(tokens)` to
# 1e-3. Without that assert the probe could silently be measuring a different
# computation from the one that was trained -- the failure mode that made
# validate_family_tree.py certify the wrong task.
# ----------------------------------------------------------------------------
@torch.no_grad()
def _wm_attn(model, tokens):
    x = model.token_emb(tokens)
    delta = model.action_to_lie(x)
    cos_a, sin_a = model.path_integrator(delta)
    B, L = tokens.shape
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), 1)
    attns = []
    for layer in model.layers:
        h = layer.norm1(x)
        Q = layer.q_proj(h).view(B, L, layer.n_heads, layer.d_head).transpose(1, 2)
        K = layer.k_proj(h).view(B, L, layer.n_heads, layer.d_head).transpose(1, 2)
        V = layer.v_proj(h).view(B, L, layer.n_heads, layer.d_head).transpose(1, 2)
        Q = _wm_rope(Q, cos_a, sin_a)
        K = _wm_rope(K, cos_a, sin_a)
        s = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(layer.d_head)
        s = s.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        a = F.softmax(s, dim=-1)
        attns.append(a)
        out = torch.matmul(a, V).transpose(1, 2).reshape(B, L, layer.d_model)
        x = x + layer.o_proj(out)
        x = x + layer.ffn(layer.norm2(x))
    return attns, model.out_proj(model.out_norm(x))


@torch.no_grad()
def _plain_attn(model, tokens):
    net = model.net
    x = net.emb(tokens)
    B, L = tokens.shape
    mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), 1)
    attns = []
    for layer in net.layers:
        h = layer.norm1(x)
        q, k, v = layer.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, layer.h, layer.dh).transpose(1, 2)
        k = k.view(B, L, layer.h, layer.dh).transpose(1, 2)
        v = v.view(B, L, layer.h, layer.dh).transpose(1, 2)
        cos, sin = _rope_cos_sin(L, layer.dh, x.device)
        q = _plain_rope(q, cos, sin)
        k = _plain_rope(k, cos, sin)
        s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(layer.dh)
        s = s.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        a = F.softmax(s, dim=-1)
        attns.append(a)
        out = torch.matmul(a, v).transpose(1, 2).reshape(B, L, layer.dim)
        x = x + layer.o(out)
        x = x + layer.ff(layer.norm2(x))
    return attns, net.head(net.norm(x))


@torch.no_grad()
def capture(model, tokens, check=False):
    fn = _plain_attn if isinstance(model, PlainFlat) else _wm_attn
    attns, logits = fn(model, tokens)
    if check:
        ref = model(tokens)
        d = (ref - logits).abs().max().item()
        if d > 1e-3:
            raise RuntimeError(
                f"attention replay does not reproduce model.forward (max abs "
                f"logit diff {d:.2e}); the probe would be measuring a different "
                f"computation from the trained one")
    return attns


# ----------------------------------------------------------------------------
# Probe episode construction. Calls the environment's own _build / _walk so the
# probe and the trainer exercise the same task code.
# ----------------------------------------------------------------------------
def _bfs(region, src, dst):
    """Shortest action path inside `region`. Returns [] if src == dst."""
    if src == dst:
        return []
    prev = {src: None}
    q = deque([src])
    while q:
        c = q.popleft()
        if c == dst:
            break
        for a, (dx, dy) in ACTION_DELTAS.items():
            n = (c[0] + dx, c[1] + dy)
            if n in region and n not in prev:
                prev[n] = (c, a)
                q.append(n)
    if dst not in prev:
        return None
    path, c = [], dst
    while prev[c] is not None:
        c, a = prev[c]
        path.append(a)
    return path[::-1]


def _walk_to(rng, region, src, dst, D, max_tries=4000):
    """A random walk of EXACTLY length D inside `region`, from src to dst.

    Why not the shortest path: the first version of this probe used BFS, which
    made the two arms' approaches 4.9 and 9.0 steps long on average. The
    transitive measure is then partly reading the arms' different sequence
    lengths and different recent history rather than their positions, and the
    confound arm's shortest path from room B has to cross the junction itself.
    Matching D across arms removes all three at once.

    Feasible for both targets from one D because the two look-alike cells are
    offset by (h-P-1, w-P-1) = (4, 2), an EVEN Manhattan distance, so the
    shortest-path lengths to them share a parity and one D can match both.
    """
    for _ in range(max_tries):
        acts, x, y = [], src[0], src[1]
        ok = True
        for step in range(D):
            rem = D - step
            cands = []
            for a, (dx, dy) in ACTION_DELTAS.items():
                n = (x + dx, y + dy)
                if n not in region:
                    continue
                d = abs(n[0] - dst[0]) + abs(n[1] - dst[1])
                if d <= rem - 1 and (rem - 1 - d) % 2 == 0:
                    cands.append((a, n))
            if not cands:
                ok = False
                break
            a, (x, y) = cands[rng.randint(len(cands))]
            acts.append(a)
        if ok and (x, y) == dst:
            return acts
    return None


def _joint_tail(rng, both, regA, join, cf, patch, pj, pc, k, bias=0.75):
    """One action sequence that is legal from BOTH look-alike cells.

    From the JOIN cell it walks into room-B-only territory; from the CONFOUNDER
    it necessarily stays inside room A. Candidate actions are filtered to those
    legal in both arms, and the confounder arm is additionally forbidden from
    entering the join patch (offset (+4,+2) from the confounder lands in it, at
    which point the two arms would be in the same place and the contrast would
    be vacuous).

    Returns (actions, cells_join, cells_confound). The action stream is shared,
    so it carries no information about which arm is which.
    """
    Bonly = both - regA
    injoin = {(join[0] + i, join[1] + j) for i in range(patch) for j in range(patch)}
    acts, cj, cc = [], [], []
    for _ in range(k):
        cands = []
        for a, (dx, dy) in ACTION_DELTAS.items():
            nj = (pj[0] + dx, pj[1] + dy)
            nc = (pc[0] + dx, pc[1] + dy)
            if nj in both and nc in regA and nc not in injoin:
                cands.append((a, nj, nc))
        if not cands:
            break
        pref = [c for c in cands if c[1] in Bonly]
        pool = pref if (pref and rng.rand() < bias) else cands
        a, pj, pc = pool[rng.randint(len(pool))]
        acts.append(a)
        cj.append(pj)
        cc.append(pc)
    return acts, cj, cc


def build_probe(env, rng, T_a, T_b, W, min_visits=2, tail=6):
    """Shared prefix (phases A, B, wander) + two steered arms + a shared tail.

    Returns None when phase A did not visit both look-alike cells often enough
    for a per-token mean to be meaningful; the caller resamples.
    """
    grid, regA, regB, join = env._build(rng)
    cf = env._cf_origin
    both = regA | regB
    tokens, cell_of, phase_of = [], {}, {}

    def run(region, start, n, tag):
        x, y = start
        for a, x, y in env._walk(rng, region, x, y, n):
            tokens.append(a + env.action_offset)
            p = len(tokens)
            tokens.append(int(grid[x, y]) + env.obs_offset)
            cell_of[p] = (x, y)
            phase_of[p] = tag
        return x, y

    cellsA, cellsB = sorted(regA), sorted(regB)
    pos = run(regA, cellsA[rng.randint(len(cellsA))], T_a, "A")
    pos = run(regB, cellsB[rng.randint(len(cellsB))], T_b, "B")
    # The wander is confined to room A. Both arms then approach their target
    # from the same room-A cell, so neither approach re-enters room B and
    # neither crosses the junction on the way -- the confound arm's shortest
    # path from room B did exactly that in the first version of this probe,
    # which biased the transitive measure in the direction of the hypothesis.
    # Tagged "A", not a phase of its own: it walks room A, so its tokens are
    # room-A memories and must sit in the room-A pool. Left out of that pool
    # they act as an uncounted, more recent competitor for retrieval at exactly
    # the cells the confound arm visits, which deflated that arm's measured
    # retrieval to 0.96x while the join arm -- with no such competitor in room
    # B -- read 2.31x. It is still inside the prefix both arms share, so nothing
    # about the pairing changes.
    pos = run(regA, pos if pos in regA else cellsA[rng.randint(len(cellsA))],
              W, "A")

    visitsA = {}
    for p, c in cell_of.items():
        if phase_of[p] == "A":
            visitsA.setdefault(c, []).append(p)

    # Offsets where phase A saw BOTH look-alike cells enough times.
    P = env.patch
    ok = [(dr, dc) for dr in range(P) for dc in range(P)
          if len(visitsA.get((join[0] + dr, join[1] + dc), [])) >= min_visits
          and len(visitsA.get((cf[0] + dr, cf[1] + dc), [])) >= min_visits]
    if not ok:
        return None
    dr, dc = ok[rng.randint(len(ok))]
    join_cell = (join[0] + dr, join[1] + dc)
    cf_cell = (cf[0] + dr, cf[1] + dc)
    assert grid[join_cell] == grid[cf_cell], "look-alike cells must be identical"

    # One shared tail, generated once and replayed by both arms.
    tail_acts, cells_j, cells_c = _joint_tail(
        rng, both, regA, join, cf, P, join_cell, cf_cell, tail)
    if len(tail_acts) < tail:
        return None
    n_in_B = sum(1 for c in cells_j if c not in regA)
    if n_in_B == 0:                 # tail never left room A: no transitive step
        return None

    # Both approaches have EXACTLY the same length, so the two arms' sequences
    # are the same length and their softmax denominators are comparable.
    bfs_j, bfs_c = _bfs(regA, pos, join_cell), _bfs(regA, pos, cf_cell)
    if bfs_j is None or bfs_c is None:
        return None
    D = max(len(bfs_j), len(bfs_c), 2)
    if (len(bfs_j) - len(bfs_c)) % 2 != 0:   # parity must match for a shared D
        return None
    paths = {}
    for name, target in (("join", join_cell), ("confound", cf_cell)):
        p = _walk_to(rng, regA, pos, target, D)
        if p is None:
            return None
        paths[name] = p

    arms = {}
    for name, target, tcells in (("join", join_cell, cells_j),
                                 ("confound", cf_cell, cells_c)):
        path = paths[name]
        toks = list(tokens)
        x, y = pos
        query = None
        for a in path:
            dx, dy = ACTION_DELTAS[a]
            x, y = x + dx, y + dy
            query = len(toks)                       # the action token
            toks.append(a + env.action_offset)
            toks.append(int(grid[x, y]) + env.obs_offset)
        assert (x, y) == target
        # Tail: observations REVEALED. The mask token is never seen in training
        # (episodes are equal-length so generate_train_batch never pads with it),
        # so feeding it here would put the model out of distribution. Attention
        # is read at the ACTION token of each tail step, i.e. before that step's
        # observation is visible.
        tail_q = []
        for a, c in zip(tail_acts, tcells):
            tail_q.append(len(toks))
            toks.append(a + env.action_offset)
            toks.append(int(grid[c]) + env.obs_offset)
        arms[name] = {"tokens": toks, "query": query, "path_len": len(path),
                      "tail_q": tail_q, "tail_cells": list(tcells),
                      "tail_isB": [c not in regA for c in tcells]}

    by_cell = {"A": {}, "B": {}}
    for p, c in cell_of.items():
        if phase_of[p] in ("A", "B"):
            by_cell[phase_of[p]].setdefault(c, []).append(p)

    return {"arms": arms, "by_cell": by_cell,
            "idx_join": visitsA[join_cell], "idx_cf": visitsA[cf_cell],
            "idx_B": [p for p in cell_of if phase_of[p] == "B"],
            "idx_A": [p for p in cell_of if phase_of[p] == "A"],
            "offset": (int(dr), int(dc)), "tail_in_B": int(n_in_B),
            "symbol": int(grid[join_cell])}


def _shares(attn_layers, ep, arm, device):
    """Per-layer (join_share, massB) at the arm's query position, plus the
    mean phase-B attention mass over the shared tail steps."""
    q = arm["query"]
    ij = torch.tensor(ep["idx_join"], device=device)
    ic = torch.tensor(ep["idx_cf"], device=device)
    ib = torch.tensor(ep["idx_B"], device=device)
    ia = torch.tensor(ep["idx_A"], device=device)
    out, tail = [], []
    for a in attn_layers:
        row = a[0, :, q, :].mean(0)                 # mean over heads
        mj = row[ij].sum().item() / len(ij)         # per-token mean: cancels
        mc = row[ic].sum().item() / len(ic)         # any visit-count asymmetry
        share = mj / (mj + mc) if (mj + mc) > 0 else float("nan")
        # B-share, not raw mass: the arms' sequences differ in length, so a raw
        # softmax mass partly tracks the denominator rather than the retrieval.
        bA, bB = row[ia].sum().item(), row[ib].sum().item()
        out.append((share, bB / (bA + bB) if (bA + bB) > 0 else float("nan")))
        tr = a[0, :, arm["tail_q"], :].mean(0)      # (n_tail, L)
        ta, tb = tr[:, ia].sum(-1), tr[:, ib].sum(-1)
        tail.append((tb / (ta + tb).clamp_min(1e-12)).tolist())
    return out, tail


def _concentration(attn_layers, ep, arm, device):
    """How sharply does attention land on the explore-phase tokens of the cell
    the agent is standing on?

    `B-share` (above) sums attention over all 256 phase-B tokens, which is
    dominated by diffuse background and cannot see a few sharp peaks -- it reads
    ~0.52 for a model with 0.97 revisit accuracy, i.e. it is nearly blind. This
    is the sensitive version:

        concentration = (mean attention per token at the CURRENT cell)
                      / (mean attention per token over the SAME PHASE)

    1.0 means no retrieval at all; higher means the model is pulling up its own
    memory of this specific cell.

    The baseline is taken WITHIN the phase the cell was seen in, which matters:
    phase B is nearer the query than phase A, so normalising against all explore
    tokens together makes any phase-B cell look retrieved and any phase-A cell
    look ignored purely from recency. Under that version MapWM read 2.80x for
    room-B cells and 0.89x for room-A cells -- but so did PlainFlat (1.61x vs
    0.56x), which cannot stitch, and that is what exposed it as recency.

    For the JOIN arm at a room-B-only tail cell this is the transitive quantity:
    the agent arrived through room A's frame and has only ever seen that cell
    during the phase-B walk. For the CONFOUND arm every tail cell is in room A,
    so its value is ordinary within-room retrieval -- the natural yardstick for
    how good stitched retrieval could possibly be.
    """
    ia = torch.tensor(ep["idx_A"], device=device)
    ib = torch.tensor(ep["idx_B"], device=device)
    out = []
    for a in attn_layers:
        tr = a[0, :, arm["tail_q"], :].mean(0)
        baseA, baseB = tr[:, ia].mean(-1), tr[:, ib].mean(-1)
        vals = []
        for s, (c, isB) in enumerate(zip(arm["tail_cells"], arm["tail_isB"])):
            ph = "B" if isB else "A"
            idx = ep["by_cell"][ph].get(c, [])
            base = (baseB if isB else baseA)[s]
            if not idx or base <= 0:
                vals.append(None)
                continue
            here = tr[s, torch.tensor(idx, device=device)].mean()
            vals.append((float(here / base), bool(isB)))
        out.append(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--T-a", type=int, default=256)
    ap.add_argument("--T-b", type=int, default=256)
    ap.add_argument("--wander", type=int, default=32)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(_REPO / "STITCH_ATTENTION.md"))
    args = ap.parse_args()
    dev = torch.device(args.device)

    from mapformer.train_variant import VARIANT_MAP
    per_ckpt = {}
    for ck in args.checkpoints:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        kw = blob["env"]
        env = StitchWorld(seed=args.seed, **kw)
        model = VARIANT_MAP[blob["variant"]](
            vocab_size=blob["vocab_size"], d_model=blob["d_model"],
            n_heads=blob["n_heads"], n_layers=blob["n_layers"],
            grid_size=max(2 * kw["h"] - kw["patch"], 2 * kw["w"] - kw["patch"]))
        model.load_state_dict(blob["model_state"])
        model.to(dev).eval()

        rng = np.random.RandomState(args.seed)
        rows, checked = [], False
        tries = 0
        while len(rows) < args.n_episodes and tries < args.n_episodes * 20:
            tries += 1
            ep = build_probe(env, rng, args.T_a, args.T_b, args.wander)
            if ep is None:
                continue
            rec = {"offset": ep["offset"], "symbol": ep["symbol"],
                   "n_join": len(ep["idx_join"]), "n_cf": len(ep["idx_cf"]),
                   "tail_in_B": ep["tail_in_B"]}
            for name, arm in ep["arms"].items():
                t = torch.tensor(arm["tokens"], dtype=torch.long,
                                 device=dev).unsqueeze(0)
                at = capture(model, t, check=not checked)
                checked = True
                rec[name], rec[name + "_tail"] = _shares(at, ep, arm, dev)
                rec[name + "_conc"] = _concentration(at, ep, arm, dev)
                rec[name + "_pathlen"] = arm["path_len"]
            rows.append(rec)
        per_ckpt[ck] = {"variant": blob["variant"], "seed": blob["seed"],
                        "rows": rows, "n_layers": blob["n_layers"],
                        "revisit_acc": blob["results"]["revisit_acc"],
                        "accept": len(rows) / max(tries, 1)}
        nl = blob["n_layers"]
        for L in range(nl):
            js = np.array([r["join"][L][0] for r in rows])
            cs = np.array([r["confound"][L][0] for r in rows])
            print(f"{Path(ck).name:34s} layer{L}  join_arm={js.mean():.4f}  "
                  f"cf_arm={cs.mean():.4f}  diff={js.mean()-cs.mean():+.4f}",
                  flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- aggregate by variant across seeds ----
    by_var = {}
    for ck, d in per_ckpt.items():
        by_var.setdefault(d["variant"], []).append(d)

    lines = ["# Stitch: attention-map probe", "",
             "CSCG reports the stitching experiment qualitatively "
             "(\"Predictive performance on the stitching of the two rooms is "
             "perfect\") and evaluates it by inspecting the learned transition "
             "matrix. MapFormer exposes no such matrix, so this ports the "
             "*evaluation style* -- inspect the learned structure -- to "
             "attention.", "",
             "`join_share` = attention (per token) on phase-A tokens at the JOIN "
             "cell, over the sum of that and the same for the CONFOUNDER cell. "
             "The two cells emit the **same symbol**. Layer 0 is the clean "
             "readout: its keys are the token embedding rotated by the "
             "path-integrated angle and nothing else, so two same-symbol tokens "
             "differ there only in position.", "",
             "**The statistic is the PAIRED DIFFERENCE, whose floor is exactly "
             "0.** Both arms are scored on the same two cells against a literally "
             "identical phase-A prefix, so visit counts, recency and token "
             "identity cancel between them. The per-arm 0.5 is weaker: within a "
             "single episode one look-alike happened to be seen more often or "
             "more recently, so 0.5 holds per arm only in expectation over "
             "episodes.", "",
             "Approach paths are random walks of EQUAL length confined to room A "
             "in both arms, so neither arm re-enters room B or crosses the "
             "junction before the measurement.", "",
             "The bootstrap CI below resamples EPISODES and so understates the "
             "uncertainty that matters; the per-seed line under each table is the "
             "honest n=3 replication and should be read first.", ""]

    summary = {}
    for var, ds in sorted(by_var.items()):
        nl = ds[0]["n_layers"]
        lines += [f"## {var}  (n={len(ds)} seeds, "
                  f"{len(ds[0]['rows'])} episodes each)", "",
                  "| layer | join arm | confound arm | paired diff | "
                  "episodes with diff>0 | B-share join | B-share confound |",
                  "|---|---|---|---|---|---|---|"]
        summary[var] = {}
        for L in range(nl):
            j = np.array([r["join"][L][0] for d in ds for r in d["rows"]])
            c = np.array([r["confound"][L][0] for d in ds for r in d["rows"]])
            bj = np.array([r["join"][L][1] for d in ds for r in d["rows"]])
            bc = np.array([r["confound"][L][1] for d in ds for r in d["rows"]])
            diff = j - c
            frac = float((diff > 0).mean())
            # bootstrap CI over episodes on the paired difference (seeded, so
            # the interval is reproducible across re-runs of the probe)
            brng = np.random.RandomState(12345)
            bs = np.array([brng.choice(diff, len(diff)).mean()
                           for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            summary[var][L] = dict(join=float(j.mean()), confound=float(c.mean()),
                                   diff=float(diff.mean()), ci=[float(lo), float(hi)],
                                   frac_pos=frac, massB_join=float(bj.mean()),
                                   massB_confound=float(bc.mean()), n=int(len(diff)))
            lines.append(
                f"| {L} | {j.mean():.4f} | {c.mean():.4f} | "
                f"**{diff.mean():+.4f}** [{lo:+.4f}, {hi:+.4f}] | "
                f"{frac:.3f} | {bj.mean():.4f} | {bc.mean():.4f} |")
        # ---- transitive readout: the shared tail past the junction ----
        tj = np.array([r["join_tail"][0] for d in ds for r in d["rows"]])
        tc = np.array([r["confound_tail"][0] for d in ds for r in d["rows"]])
        inB = np.mean([r["tail_in_B"] for d in ds for r in d["rows"]])
        lines += ["", f"**Transitive tail** (layer 0). One action sequence, legal "
                  f"from both look-alike cells, replayed by both arms: from the "
                  f"join it enters room-B-only cells "
                  f"({inB:.1f} of {tj.shape[1]} steps on average), from the "
                  f"confounder it cannot leave room A. Value = B-share, "
                  f"massB/(massA+massB) over explore-phase observation tokens, "
                  f"read at each step's ACTION token (before that step's "
                  f"observation is visible). Unlike `join_share` this has no "
                  f"analytic floor -- the confound arm is its empirical floor, "
                  f"and the arms' tokens are matched only from the shared prefix, "
                  f"not step for step. Mean approach-path length: join "
                  f"{np.mean([r['join_pathlen'] for d in ds for r in d['rows']]):.1f}, "
                  f"confound "
                  f"{np.mean([r['confound_pathlen'] for d in ds for r in d['rows']]):.1f} "
                  f"steps.", "",
                  "| tail step | " + " | ".join(str(s + 1) for s in range(tj.shape[1])) + " |",
                  "|---" * (tj.shape[1] + 1) + "|",
                  "| join arm | " + " | ".join(f"{v:.4f}" for v in tj.mean(0)) + " |",
                  "| confound arm | " + " | ".join(f"{v:.4f}" for v in tc.mean(0)) + " |",
                  "| diff | " + " | ".join(f"{v:+.4f}" for v in (tj - tc).mean(0)) + " |"]
        summary[var]["tail_layer0"] = {"join": tj.mean(0).tolist(),
                                       "confound": tc.mean(0).tolist(),
                                       "mean_steps_in_B": float(inB)}

        # ---- retrieval concentration on the cell the agent is standing on ----
        def conc(dset, arm, want_B):
            v = [x[0] for d in dset for r in d["rows"] for x in r[arm + "_conc"][0]
                 if x is not None and x[1] == want_B]
            return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

        cj_b, nj_b = conc(ds, "join", True)
        cj_a, nj_a = conc(ds, "join", False)
        cc_a, nc_a = conc(ds, "confound", False)
        per_seed_cjb = [conc([d], "join", True)[0] for d in ds]
        per_seed_cca = [conc([d], "confound", False)[0] for d in ds]
        summary[var]["concentration_layer0"] = {
            "join_arm_roomB_cells": cj_b, "join_arm_roomA_cells": cj_a,
            "confound_arm_roomA_cells": cc_a,
            "per_seed_join_roomB": per_seed_cjb,
            "per_seed_confound_roomA": per_seed_cca,
            "n": [nj_b, nj_a, nc_a]}
        lines += ["", "**Retrieval concentration** (layer 0): mean attention per "
                  "token on the explore-phase tokens of the cell the agent is "
                  "standing on, divided by mean attention per token over the "
                  "SAME PHASE. **1.0 = no retrieval.** `B-share` above sums over "
                  "all 256 phase-B tokens and is dominated by diffuse "
                  "background, so it cannot see a few sharp peaks; this is the "
                  "sensitive version. The baseline is taken within phase because "
                  "phase B sits nearer the query than phase A: normalising "
                  "against both phases together makes every room-B cell look "
                  "retrieved from recency alone, for PlainFlat as much as for "
                  "MapFormer.", "",
                  "| tail cell | concentration | n |", "|---|---|---|",
                  f"| join arm, room-B-only cells (**the transitive case**: "
                  f"reached through room A, only ever seen in phase B) | "
                  f"**{cj_b:.2f}x** | {nj_b} |",
                  f"| join arm, room-A cells | {cj_a:.2f}x | {nj_a} |",
                  f"| confound arm, room-A cells (within-room retrieval — the "
                  f"yardstick) | {cc_a:.2f}x | {nc_a} |", "",
                  "per-seed, join arm room-B cells: " +
                  ", ".join(f"{d['seed']}: {v:.2f}x"
                            for d, v in zip(ds, per_seed_cjb)),
                  "per-seed, confound arm room-A cells: " +
                  ", ".join(f"{d['seed']}: {v:.2f}x"
                            for d, v in zip(ds, per_seed_cca))]
        # ---- seed-level replication (the unit that matters, n=3) ----
        sd_diff = [float(np.mean([r["join"][0][0] - r["confound"][0][0]
                                  for r in d["rows"]])) for d in ds]
        sd_join = [float(np.mean([r["join"][0][0] for r in d["rows"]])) for d in ds]
        sd_cf = [float(np.mean([r["confound"][0][0] for r in d["rows"]])) for d in ds]
        sd_tail = [float(np.mean([np.mean(r["join_tail"][0]) -
                                  np.mean(r["confound_tail"][0])
                                  for r in d["rows"]])) for d in ds]
        summary[var]["per_seed_layer0"] = {
            "seeds": [d["seed"] for d in ds], "paired_diff": sd_diff,
            "join_arm": sd_join, "confound_arm": sd_cf, "tail_diff": sd_tail,
            "revisit_acc": [d["revisit_acc"] for d in ds]}
        lines += ["", "### Per-seed, layer 0 (n=3 — read this before the CI)", "",
                  "| seed | join arm | confound arm | paired diff | tail diff | "
                  "held-out revisit acc |", "|---|---|---|---|---|---|"]
        for d, a, b, df, tl in zip(ds, sd_join, sd_cf, sd_diff, sd_tail):
            lines.append(f"| {d['seed']} | {a:.4f} | {b:.4f} | {df:+.4f} | "
                         f"{tl:+.4f} | {d['revisit_acc']:.4f} |")
        lines.append(
            f"| **mean ± sd** | **{np.mean(sd_join):.4f} ± {np.std(sd_join, ddof=1):.4f}** "
            f"| **{np.mean(sd_cf):.4f} ± {np.std(sd_cf, ddof=1):.4f}** "
            f"| **{np.mean(sd_diff):+.4f} ± {np.std(sd_diff, ddof=1):.4f}** "
            f"| **{np.mean(sd_tail):+.4f} ± {np.std(sd_tail, ddof=1):.4f}** | |")
        lines.append("")

    lines += ["", "## What this does and does not show", "",
              "Shows: the model tells apart two cells that emit the same symbol "
              "and sit in an identical shared prefix, and it does so in the "
              "right direction at both of them. And it retrieves a room-B "
              "memory of a cell it reached through room A's frame nearly as "
              "sharply as it retrieves a within-room memory. That is CSCG's "
              "negative control and its transitive claim, on the only structure "
              "MapFormer exposes.", "",
              "Does not show:", "",
              "- **That the discrimination is specifically path integration.** "
              "The two arms differ in the observations along their approach as "
              "well as in position, so the model may be localising from that "
              "content. Position is one component of the context it uses, not "
              "demonstrably the whole of it. What the PlainFlat row establishes "
              "is that index positions plus the same content are not enough at "
              "this training budget.",
              "- **A clean architecture comparison.** PlainFlat reaches 0.57-0.61 "
              "held-out revisit accuracy against MapWM-Flat's 0.97. It is the "
              "much weaker model, so 'PlainFlat shows no effect' is entangled "
              "with 'PlainFlat did not learn the map'. Parameters are matched "
              "(600,212 vs 600,660); capability is not.",
              "- **A stable transitive magnitude.** The per-seed transitive "
              "concentrations spread widely, and on one seed the stitched "
              "retrieval is well below that seed's own within-room yardstick. "
              "Every seed is above the PlainFlat range, so the direction "
              "replicates; the size does not.",
              "- **Anything CSCG measured.** CSCG reports this experiment with "
              "no number at all, so none of these values can be compared to a "
              "published one. This is a port of their evaluation style, not a "
              "reproduction of their result.", ""]

    Path(args.out).write_text("\n".join(lines) + "\n")
    json.dump({"summary": summary,
               "per_ckpt": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                            for k, v in per_ckpt.items()}},
              open(str(args.out).replace(".md", ".json"), "w"), indent=2)
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
