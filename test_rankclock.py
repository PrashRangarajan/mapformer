"""Unit checks for the G-1 (RANKCLOCK) arms before any GPU time. Every claim in `model_rankclock.py` is measured.

  cd /home/prashr && python3 -m mapformer.test_rankclock
"""
import os
import subprocess
import sys
import tempfile
import time
import traceback

import torch

from mapformer.environment import GridWorld
from mapformer.model_rankclock import TORUS_ACTIONS, TORUS_PAIRS
from mapformer.train_variant import VARIANT_MAP

ENV = GridWorld(size=64, n_obs_types=16, p_empty=0.5, n_landmarks=0, seed=0)
V = ENV.unified_vocab_size
KW = dict(vocab_size=V, d_model=128, n_heads=2, n_layers=1, grid_size=64)
ARMS = ("Vanilla", "Vanilla_r2clock", "Vanilla_r2clockxy", "Vanilla_r2DF", "Vanilla_r2AO", "Vanilla_r4mi")


def mk(name, seed=0):
    torch.manual_seed(seed)
    return VARIANT_MAP[name](**KW).eval()


def tokens(seed=1, B=3, L=40):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (B, L), generator=g)


def test_torus_action_ids():
    """The arms hard-code moves 0-3 with inverse pairs (1, 0) and (3, 2); check against the env itself."""
    assert ENV.action_offset == 0, ENV.action_offset
    d = ENV.ACTION_DELTAS
    assert tuple(sorted(d)) == TORUS_ACTIONS, d
    for p, m in TORUS_PAIRS:
        assert tuple(-x for x in d[p]) == tuple(d[m]), (p, m, d[p], d[m])
    obs_ids = [i for i in range(V) if i not in TORUS_ACTIONS]
    return f"moves {d}; pairs {TORUS_PAIRS} are exact inverses; {len(obs_ids)} non-move tokens (vocab {V})"


def test_shared_weights_equal_vanilla_at_init():
    """Every tensor an arm shares with Vanilla (incl. W_in, W_out) is bitwise Vanilla's at the seed; r2Clock adds only
    `action_to_lie.clock` (zeros); r2DF / r2AO add nothing; r4mi differs only in the bottleneck."""
    notes = []
    for s in (0, 7):
        base = mk("Vanilla", s).state_dict()
        for nm in ARMS[1:]:
            sd = mk(nm, s).state_dict()
            extra, missing = set(sd) - set(base), set(base) - set(sd)
            shared = set(sd) & set(base)
            diff = sorted(k for k in shared if sd[k].shape != base[k].shape or not torch.equal(sd[k], base[k]))
            if nm == "Vanilla_r2clock":
                assert extra == {"action_to_lie.clock"} and not missing, (extra, missing)
                assert torch.count_nonzero(sd["action_to_lie.clock"]) == 0
                assert not diff, diff
            elif nm == "Vanilla_r2clockxy":
                assert extra == {"action_to_lie.clock_x", "action_to_lie.clock_y"} and not missing, (extra, missing)
                assert all(torch.count_nonzero(sd[k]) == 0 for k in extra)
                assert not diff, diff
            elif nm in ("Vanilla_r2DF", "Vanilla_r2AO"):
                assert not extra and not missing and not diff, (extra, missing, diff)
            else:   # r4mi
                assert all(k.startswith("action_to_lie.") for k in diff + sorted(extra | missing)), diff
            if s == 0:
                notes.append(f"{nm}: +{sorted(extra) or 'nothing'}, differs at {diff or 'nothing'}")
    return "; ".join(notes)


def test_param_counts():
    n = {nm: sum(p.numel() for p in mk(nm).parameters()) for nm in ARMS}
    assert n["Vanilla_r2clock"] - n["Vanilla"] == 64, n
    assert n["Vanilla_r2clockxy"] - n["Vanilla"] == 128, n
    assert n["Vanilla_r2DF"] == n["Vanilla"] == n["Vanilla_r2AO"], n
    return ", ".join(f"{k} {v:,}" for k, v in n.items())


def test_slots_equal_vanilla_function_at_init():
    tok = tokens()
    with torch.no_grad():
        a = mk("Vanilla")(tok)
        for nm in ("Vanilla_r2clock", "Vanilla_r2clockxy"):
            b = mk(nm)(tok)
            assert torch.equal(a, b), f"{nm}: max|d| {(a - b).abs().max():.3e}"
    return "both slot arms: forward bitwise-equal to Vanilla at init"


def test_axis_slot_acts_only_on_its_axis():
    """clock_x changes only the x-moves' steps, clock_y only the y-moves'; observation steps never change."""
    m = mk("Vanilla_r2clockxy")
    with torch.no_grad():
        base = m.step_table().clone()
        m.action_to_lie.clock_x.normal_()
        dx = (m.step_table() - base).abs().amax(dim=(1, 2))
        m.action_to_lie.clock_x.zero_(); m.action_to_lie.clock_y.normal_()
        dy = (m.step_table() - base).abs().amax(dim=(1, 2))
    x_ids, y_ids = set(TORUS_ACTIONS[:2]), set(TORUS_ACTIONS[2:])
    for i in range(V):
        assert (dx[i] > 0) == (i in x_ids), (i, float(dx[i]))
        assert (dy[i] > 0) == (i in y_ids), (i, float(dy[i]))
    return f"clock_x moves only tokens {sorted(x_ids)}, clock_y only {sorted(y_ids)}; observation steps untouched"


def test_r2clock_slot_is_a_per_token_constant():
    """The slot adds clock_c to EVERY token's step (moves AND observations): theta_c(t) gains omega_c * t * clock_c."""
    m = mk("Vanilla_r2clock")
    with torch.no_grad():
        m.action_to_lie.clock.normal_()
        e = m.token_emb.weight[None]
        with_c = m.action_to_lie(e)[0]
        m2 = mk("Vanilla")
        without = m2.action_to_lie(m2.token_emb.weight[None])[0]
        d = with_c - without
        same = (d - m.action_to_lie.clock[None]).abs().max().item()
    assert same < 1e-6, same
    return f"step(r2clock) - step(Vanilla) == clock for all {V} tokens (max|d| {same:.1e})"


def test_r2DF_forbids_drift_and_observation_steps():
    for s in range(6):
        m = mk("Vanilla_r2DF", s)
        with torch.no_grad():
            st = m.step_table()
        non = [i for i in range(V) if i not in TORUS_ACTIONS]
        assert st[non].abs().max().item() == 0.0, "non-move step not exactly 0"
        for p, q in TORUS_PAIRS:
            assert torch.equal(st[p], -st[q]), f"pair ({p},{q}) not exact negatives"
            assert st[p].abs().max() > 1e-3, "blind pair"
        clock = st[list(TORUS_ACTIONS)].sum(0).abs().max().item()
        assert clock == 0.0, clock
    return "over 6 seeds: non-move steps exactly 0, each pair exact negatives (non-blind), move-sum clock exactly 0"


def test_r2AO_frees_moves_and_zeroes_observations():
    for s in range(6):
        m = mk("Vanilla_r2AO", s)
        with torch.no_grad():
            st = m.step_table()
            ref = m.action_to_lie(m.token_emb.weight[list(TORUS_ACTIONS)].view(1, 4, -1))[0]
        non = [i for i in range(V) if i not in TORUS_ACTIONS]
        assert st[non].abs().max().item() == 0.0
        assert torch.equal(st[list(TORUS_ACTIONS)], ref), "move steps are not the free W_out W_in e(move)"
    m = mk("Vanilla_r2AO", 0)
    with torch.no_grad():
        clock = m.step_table()[list(TORUS_ACTIONS)].sum(0).abs().max().item()
    assert clock > 1e-3, f"AO's move clock is ~0 at init ({clock:.2e}): then AO and DF are not distinguishable at init"
    return f"over 6 seeds: non-move steps exactly 0, moves = W_out W_in e(move); init move clock {clock:.3f} (free, != 0)"


def test_grads_reach_the_step_map():
    for nm in ARMS[1:]:
        m = mk(nm).train()
        m(tokens(2, 4, 48)).sum().backward()
        a2l = m.action_to_lie
        for k in ("w_in", "w_out"):
            g = getattr(a2l, k).weight.grad
            assert g is not None and g.abs().sum() > 0, f"{nm}: {k} got no gradient"
        if nm == "Vanilla_r2clock":
            assert a2l.clock.grad is not None and a2l.clock.grad.abs().sum() > 0, "clock got no gradient"
        if nm == "Vanilla_r2clockxy":
            for k in ("clock_x", "clock_y"):
                g = getattr(a2l, k).grad
                assert g is not None and g.abs().sum() > 0, f"{k} got no gradient"
    return "W_in, W_out (and clock) receive gradient in every new arm"


def test_train_variant_end_to_end_with_snapshots():
    """train_variant with every new arm, 2 epochs x 2 batches on CPU, --snapshot-every 1: checkpoint written, snapshots at
    e0000 / e0001 / e0002, the last snapshot equal to the checkpoint's step-map tensors."""
    tmp = tempfile.mkdtemp(prefix="rankclock_e2e_")
    out = []
    for nm in ("Vanilla_r2clock", "Vanilla_r2clockxy", "Vanilla_r2DF", "Vanilla_r2AO"):
        d = os.path.join(tmp, nm)
        cmd = [sys.executable, "-m", "mapformer.train_variant", "--variant", nm, "--seed", "5", "--epochs", "2",
               "--n-batches", "2", "--batch-size", "4", "--n-steps", "32", "--n-layers", "1", "--n-heads", "2",
               "--d-model", "128", "--n-landmarks", "0", "--schedule", "cosine", "--lr", "1e-3",
               "--device", "cpu", "--output-dir", d, "--snapshot-every", "1"]
        r = subprocess.run(cmd, cwd="/home/prashr", capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        snaps = sorted(os.listdir(os.path.join(d, "snaps")))
        assert snaps == ["e0000.pt", "e0001.pt", "e0002.pt"], snaps
        ck = torch.load(os.path.join(d, f"{nm}.pt"), map_location="cpu", weights_only=False)["model_state_dict"]
        last = torch.load(os.path.join(d, "snaps", "e0002.pt"))
        assert set(last) and all(torch.equal(last[k], ck[k]) for k in last), "last snapshot != checkpoint"
        assert all(k.startswith(("action_to_lie.", "path_integrator.", "token_emb.")) for k in last)
        out.append(f"{nm}: {len(snaps)} snaps, {sum(v.numel() for v in last.values()):,} floats each")
    return "; ".join(out)


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    fails = 0
    t00 = time.time()
    for t in TESTS:
        t0 = time.time()
        try:
            info = t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
            if info:
                print("      " + str(info).replace("\n", "\n      "))
        except Exception:
            fails += 1
            print(f"FAIL {t.__name__} ({time.time() - t0:.1f}s)")
            traceback.print_exc()
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} passed in {time.time() - t00:.1f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
