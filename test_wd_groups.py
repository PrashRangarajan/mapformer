"""Rule 19 checks for the RANKDECAY weight-decay groups (`train.build_optimizer`, `--wd-select` / `--wd-names`).

  cd /home/prashr && python3 -m mapformer.test_wd_groups [--ref DIR]

1. Default path: the exact pre-edit command (recorded before train.py was changed) reproduces bitwise -- losses and every
   weight. (--ref points at that recording; skipped with a note if absent.)
2. Grouping alone changes nothing: two groups both decayed at wd equal the single-group optimiser bitwise over real steps.
3. Targeting: with all gradients zero, one AdamW step scales EXACTLY the decayed parameters by (1 - lr * wd) and leaves every
   other parameter bit-identical, for "only" and for "exclude".
4. Bad names are refused; train_variant records --wd-select / --wd-names in the checkpoint's args.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import torch

from mapformer.train import build_optimizer
from mapformer.train_variant import VARIANT_MAP

OMEGA = "path_integrator.omega"
REF_CMD = ["--variant", "Vanilla_r4mi", "--seed", "77", "--epochs", "3", "--n-batches", "3", "--batch-size", "4",
           "--n-steps", "64", "--n-layers", "1", "--n-heads", "2", "--d-model", "128", "--n-landmarks", "0",
           "--schedule", "cosine", "--lr", "1e-3", "--weight-decay", "0.05", "--data-workers", "0", "--device", "cpu"]


def mk(seed=3):
    torch.manual_seed(seed)
    return VARIANT_MAP["Vanilla_r4mi"](vocab_size=21, d_model=128, n_heads=2, n_layers=1, grid_size=64)


def run_tv(out, extra=()):
    r = subprocess.run([sys.executable, "-m", "mapformer.train_variant", *REF_CMD, "--output-dir", out, *extra],
                       cwd="/home/prashr", capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    return torch.load(os.path.join(out, "Vanilla_r4mi.pt"), map_location="cpu", weights_only=False)


def test_default_path_bitwise(ref):
    if not ref or not os.path.exists(os.path.join(ref, "Vanilla_r4mi.pt")):
        return "SKIPPED (no pre-edit reference given)"
    a = torch.load(os.path.join(ref, "Vanilla_r4mi.pt"), map_location="cpu", weights_only=False)
    b = run_tv(tempfile.mkdtemp(prefix="wdg_def_"))
    assert a["losses"] == b["losses"], (a["losses"], b["losses"])
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    bad = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert not bad, bad[:5]
    return f"pre-edit vs post-edit: losses {b['losses']} identical; all {len(sa)} tensors bitwise equal"


def _steps(model, opt, n=5, seed=11):
    torch.manual_seed(1234)          # dropout draws from the global RNG: both compared runs must see the same masks
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        tok = torch.randint(0, 21, (4, 48), generator=g)
        loss = model(tok).float().logsumexp(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_grouping_alone_is_neutral():
    out = []
    for sel in ("only", "exclude"):
        m1, m2 = mk(), mk()
        s1 = _steps(m1, build_optimizer(m1, 1e-3, 0.05))
        s2 = _steps(m2, build_optimizer(m2, 1e-3, 0.05, sel, (OMEGA,), wd_other=0.05))
        bad = [k for k in s1 if not torch.equal(s1[k], s2[k])]
        assert not bad, (sel, bad[:5])
        out.append(sel)
    return f"two groups both at wd 0.05 == one group, bitwise over 5 real steps ({', '.join(out)})"


def test_decay_hits_exactly_the_selected_parameters():
    lr, wd = 1e-2, 0.5
    notes = []
    for sel in ("only", "exclude"):
        m = mk()
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        opt = build_optimizer(m, lr, wd, sel, (OMEGA,))
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
        decayed_should = {n for n in before if (n == OMEGA) == (sel == "only")}
        for n, p in m.named_parameters():
            if n in decayed_should:
                want = before[n] * (1 - lr * wd)
                assert torch.allclose(p.detach(), want, rtol=0, atol=1e-7), (sel, n)
                assert not torch.equal(p.detach(), before[n]) or before[n].abs().max() == 0, (sel, n, "unchanged")
            else:
                assert torch.equal(p.detach(), before[n]), (sel, n, "changed but should not decay")
        notes.append(f"{sel}: {len(decayed_should)} decayed, {len(before) - len(decayed_should)} untouched")
    return "; ".join(notes)


def test_bad_names_refused():
    m = mk()
    for sel, names in (("only", ("no.such.param",)), ("exclude", ()), ("bogus", (OMEGA,))):
        try:
            build_optimizer(m, 1e-3, 0.05, sel, names)
        except ValueError:
            continue
        raise AssertionError(f"accepted wd_select={sel!r} wd_names={names!r}")
    return "unknown names, empty names and an unknown mode all raise ValueError"


def test_train_variant_records_the_groups():
    b = run_tv(tempfile.mkdtemp(prefix="wdg_tv_"), ["--wd-select", "only", "--wd-names", OMEGA])
    A = b["config"]["args"]
    assert A["wd_select"] == "only" and A["wd_names"] == [OMEGA], (A.get("wd_select"), A.get("wd_names"))
    d = run_tv(tempfile.mkdtemp(prefix="wdg_tv0_"))
    assert d["config"]["args"]["wd_select"] is None and d["config"]["args"]["wd_names"] == []
    assert b["losses"] != d["losses"], "omega-only decay gave the same run as decaying everything"
    return "args record wd_select / wd_names; omega-only differs from the default run, as it must"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    tests = [("test_default_path_bitwise", lambda: test_default_path_bitwise(a.ref)),
             ("test_grouping_alone_is_neutral", test_grouping_alone_is_neutral),
             ("test_decay_hits_exactly_the_selected_parameters", test_decay_hits_exactly_the_selected_parameters),
             ("test_bad_names_refused", test_bad_names_refused),
             ("test_train_variant_records_the_groups", test_train_variant_records_the_groups)]
    fails = 0
    for name, t in tests:
        try:
            print(f"PASS {name}\n      {t()}")
        except Exception as e:
            fails += 1
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
