"""Construction / equivalence checks for TW_AMBIG_PREREG.md, CPU only (CUDA_VISIBLE_DEVICES=""). Output:
tw_ambig_checks_out.txt. Every check prints PASS / FAIL with its number.

 C1  class identity: MapWM = VARIANT_MAP['Vanilla_r4'], RoPE = VARIANT_MAP['RoPE'] (the text-world arms)
 C2  shared init: at each seed every arm's base weights equal the context-free arm's (1L: MapWM; 2L: CF2 / HSR and
     Vanilla_r4 at 2 layers); extra keys listed
 C3  forward equivalence at init: RoleTag (tagged stream) == MapWM (plain stream); CF2 == HSR; DirOnlyRole's step is
     zero except at movement-role direction words; RoleTag with nm_emb = 0 has zero step at non-movement uses
 C4  causality: perturbing a later token changes no step and no logit at earlier positions (every path arm)
 C5  CF2's alpha is a buffer (0, not trained); HSR's alpha trains (one CPU optimiser step)
 C6  rescore_hook: every arm's layers are KNOWN classes (hooked = n_layers, nothing skipped); the readout's own
     rescale equals rescore_hook's
 C7  readout validation on constructed models: context-free swap ratio 1.000 on every class; RoleTag with
     nm_emb = 0 -> ratio 0 and non-movement drift 0; DirOnlyRole -> non-movement swap 0
 C8  readout positive control: the stored TW_NORMSTEP MapWM s10 checkpoint, run through this module's eval on the
     p_nm = 0 task (= TextWorld): registered accuracy reproduced; reliance and swap on a trained map
 C9  trainer reproduction, CPU vs CPU, bitwise: train_tw_ambig --p-nm 0 vs train_tw_normstep, MapWM, same seed and a
     small config (losses, state dict, eval accuracy)
 C10 trainer path vs the stored GPU run's first epoch (CPU, the stored 900-epoch LR schedule): close, not bitwise
     (GPU vs CPU); the bitwise GPU check is deferred to the GPU pilot
 C11 the full analysis report runs on a synthetic n=8 result set with every readout key (and the n<8 pilot path)
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "/home/prashr")
torch.set_num_threads(4)
from mapformer.environment_tw_ambig import TextWorldAmbig, NM_CLASSES
from mapformer.model_tw_ambig import ARMS, build, steps, RoleTag, CF2
from mapformer.model_rank import MapFormerWM_r4
from mapformer.model_baseline_rope import MapFormerWM_RoPE
import mapformer.tw_ambig_readouts as RO

REPO = "/home/prashr/mapformer"
FAIL = []


def check(name, ok, msg=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)
    if not ok:
        FAIL.append(name)


def mk(arm, seed, p_nm=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    env = TextWorldAmbig(seed=seed, p_nm=p_nm, tag_roles=ARMS[arm][2])
    return build(arm, env).eval(), env


def walks(env, n, seed=1):
    np.random.seed(seed); out = []
    for _ in range(n):
        t, o, r = env.generate_trajectory(256); out.append((t, list(env.ctx)))
    return out


def main():
    from mapformer.train_variant import VARIANT_MAP
    check("C1 MapWM is VARIANT_MAP['Vanilla_r4']", VARIANT_MAP["Vanilla_r4"] is MapFormerWM_r4)
    check("C1 RoPE is VARIANT_MAP['RoPE']", VARIANT_MAP["RoPE"] is MapFormerWM_RoPE)

    for seed in (40, 41):
        ref1, _ = mk("MapWM", seed); sd1 = ref1.state_dict()
        torch.manual_seed(seed); np.random.seed(seed)
        ref2 = MapFormerWM_r4(vocab_size=89, d_model=128, n_heads=2, n_layers=2, grid_size=64); sd2 = ref2.state_dict()
        for arm in ARMS:
            m, _ = mk(arm, seed); sd = m.state_dict()
            ref = sd1 if ARMS[arm][1] == 1 else sd2
            if arm.startswith("RoPE"):
                torch.manual_seed(seed); np.random.seed(seed)
                ref = MapFormerWM_RoPE(vocab_size=89, d_model=128, n_heads=2, n_layers=ARMS[arm][1], grid_size=64).state_dict()
            shared = [k for k in sd if k in ref]
            eq = all(torch.equal(sd[k], ref[k]) for k in shared)
            check(f"C2 s{seed} {arm}: {len(shared)} shared tensors equal the reference", eq,
                  f"(extra: {[k for k in sd if k not in ref]})")

    m1, e1 = mk("MapWM", 40); mt, et = mk("RoleTag", 40); md, ed = mk("DirOnlyRole", 40)
    W1, Wt = walks(e1, 4), walks(et, 4)
    with torch.no_grad():
        same = all(torch.equal(et.untag(b[0]), a[0]) for a, b in zip(W1, Wt))
        dl = max(float((m1(a[0][None]) - mt(b[0][None])).abs().max()) for a, b in zip(W1, Wt))
        check("C3 tagged stream == plain after untag (4 walks)", same)
        check("C3 RoleTag(tagged) logits == MapWM(plain) logits at init", dl == 0.0, f"max diff {dl}")
        bad = 0; nmv = 0
        for t, ctx in walks(ed, 4):
            d = steps(md, t[None])[0].abs().sum((-1, -2))
            mv = set(i for i, r, f in ctx if r == "move")
            bad += int(sum(float(d[i]) > 0 for i in range(len(t)) if i not in mv)); nmv += int(sum(float(d[i]) > 0 for i in mv))
        check("C3 DirOnlyRole: non-zero steps only at movement-role direction words", bad == 0 and nmv > 0,
              f"(non-zero elsewhere {bad}; move words with a step {nmv})")
        mz = copy.deepcopy(mt); mz.nm_emb.data.zero_()
        z = max(float(steps(mz, t[None])[0][[i for i, r, f in ctx if r == "nm"]].abs().max()) for t, ctx in Wt)
        check("C3 RoleTag with nm_emb = 0: zero step at every non-movement use", z == 0.0, f"max {z}")
        mh, eh = mk("HSR", 40); mc, ec = mk("CF2", 40)
        dh = max(float((mh(t[None]) - mc(t[None])).abs().max()) for t, _ in walks(eh, 3))
        check("C3 CF2 == HSR at init (alpha 0)", dh == 0.0, f"max diff {dh}")

        for arm in ARMS:
            m, e = mk(arm, 41); t = walks(e, 1)[0][0][None]; t2 = t.clone(); j = 150
            t2[0, j] = (t2[0, j] + 7) % e.unified_vocab_size
            lo = float((m(t)[0, :j] - m(t2)[0, :j]).abs().max())
            s1, s2 = steps(m, t), steps(m, t2)
            ls = 0.0 if s1 is None else float((s1[0, :j] - s2[0, :j]).abs().max())
            check(f"C4 {arm}: causal (perturb token {j}: earlier logits / steps unchanged)", lo == 0.0 and ls == 0.0,
                  f"logit {lo} step {ls}")
        # CF2's step is context-free: the step at t depends on token t only
        t = walks(ec, 1)[0][0][None]; t2 = t.clone(); t2[0, 20] = (t2[0, 20] + 3) % ec.unified_vocab_size
        s1, s2 = steps(mc, t), steps(mc, t2)
        cf = float((s1[0, 21:] - s2[0, 21:]).abs().max())
        check("C4 CF2: step context-free (perturbing an earlier token changes no later step)", cf == 0.0, f"{cf}")

    mh, eh = mk("HSR", 42); mc, ec = mk("CF2", 42)
    check("C5 CF2 ctx_alpha is a buffer, not a parameter",
          "ctx_alpha" not in dict(mc.named_parameters()) and "ctx_alpha" in dict(mc.named_buffers()))
    for m, e, name in ((mh, eh, "HSR"), (mc, ec, "CF2")):
        m.train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        np.random.seed(3); tok, om, rv, _ = e.generate_batch(4, 1024)
        lg = m(tok[:, :-1]); msk = rv[:, 1:]
        loss = torch.nn.functional.cross_entropy(lg[msk], tok[:, 1:][msk]); opt.zero_grad(); loss.backward(); opt.step()
        a = float(m.ctx_alpha.detach())
        check(f"C5 {name} alpha after one AdamW step (lr 1e-2, {int(msk.sum())} targets): {a:+.5f}",
              int(msk.sum()) > 0 and ((a != 0.0) if name == "HSR" else (a == 0.0)))

    import mapformer.rescore_hook as RH
    import torch.nn as nn
    orig = nn.Module.eval
    RH.install("auto")
    for arm in ARMS:
        RH.STATS["hooked"] = 0; RH.STATS["skipped"] = set()
        m, e = mk(arm, 43)
        m.eval()
        check(f"C6 {arm}: rescore_hook hooks {RH.STATS['hooked']} = {ARMS[arm][1]} layers, skipped {sorted(RH.STATS['skipped'])}",
              RH.STATS["hooked"] == ARMS[arm][1] and not RH.STATS["skipped"])
    # the readout's rescale vs rescore_hook, on a TRAINED model (stored TW_NORMSTEP MapWM s12, a below-ceiling run whose
    # eval-mode accuracy the rescale moves; p_nm = 0 task = TextWorld)
    ck12 = f"{REPO}/runs/tw_normstep/p0/MapWM_s12/MapWM.pt"; te0 = TextWorldAmbig(seed=10000, p_nm=0.0)
    W = RO.walk_set(te0, 20, 10**6)
    m = build("MapWM", te0); m.load_state_dict(torch.load(ck12, map_location="cpu", weights_only=False)["model_state_dict"])
    m.eval(); a_hook = RO.acc_split(m, W, "cpu")["acc"]
    nn.Module.eval = orig
    m2 = build("MapWM", te0); m2.load_state_dict(m.state_dict()); m2.eval()
    a_plain = RO.acc_split(m2, W, "cpu")["acc"]; a_ro = RO.rescored(m2, W, "cpu")
    check("C6 readout rescale == rescore_hook (stored MapWM s12, 20 walks)", a_hook == a_ro and a_ro != a_plain,
          f"hook {a_hook:.4f} vs readout {a_ro:.4f} (eval mode without rescale {a_plain:.4f})")

    te = TextWorldAmbig(seed=10000, p_nm=0.3); Ws = RO.walk_set(te, 12, 321)
    m, _ = mk("MapWM", 44); sw = RO.swap(m, Ws, "cpu", te, n_max=20)
    r = [sw[f"{c}@15"] / sw["move@15"] for c in NM_CLASSES]
    check("C7 MapWM (untrained) swap ratio 1.000 on every class", all(abs(x - 1) < 1e-6 for x in r), f"{np.round(r, 6)}")
    m, _ = mk("CF2", 44); sw = RO.swap(m, Ws, "cpu", te, n_max=20)
    r = [sw[f"{c}@15"] / sw["move@15"] for c in NM_CLASSES]
    check("C7 CF2 (untrained) swap ratio 1.000 on every class", all(abs(x - 1) < 1e-6 for x in r), f"{np.round(r, 6)}")
    tet = TextWorldAmbig(seed=10000, p_nm=0.3, tag_roles=True); Wst = RO.walk_set(tet, 12, 321)
    mz, _ = mk("RoleTag", 44); mz.nm_emb.data.zero_(); sw = RO.swap(mz, Wst, "cpu", tet, n_max=20)
    check("C7 RoleTag nm_emb=0: non-movement swap exactly 0 on every class",
          all(sw[f"{c}@15"] == 0.0 for c in NM_CLASSES), f"(move {sw['move@15']:.4f})")
    md_, _ = mk("DirOnlyRole", 44); sw = RO.swap(md_, Wst, "cpu", tet, n_max=20)
    dd = RO.drift_disp(md_, RO.walk_set(tet, 6, 5), "cpu", tet)
    check("C7 DirOnlyRole: non-movement swap exactly 0; non-movement-sentence drift exactly 0 (no word there steps)",
          all(sw[f"{c}@15"] == 0.0 for c in NM_CLASSES) and dd["nm_drift_rad"] == 0.0, f"(nm drift {dd['nm_drift_rad']})")

    # C8 positive control on a stored trained text-world model (p_nm = 0 task = TextWorld, vocabulary 58)
    ck = f"{REPO}/runs/tw_normstep/p0/MapWM_s10/MapWM.pt"; ev = json.load(open(f"{REPO}/runs/tw_normstep/p0/MapWM_s10/eval.json"))
    b = torch.load(ck, map_location="cpu", weights_only=False)
    te0 = TextWorldAmbig(seed=10000, p_nm=0.0); m = build("MapWM", te0); m.load_state_dict(b["model_state_dict"]); m.eval()
    W0 = RO.walk_set(te0, 200, 10**6)
    a0 = RO.acc_split(m, W0, "cpu")
    check("C8 stored TW_NORMSTEP MapWM s10 through this eval (p_nm=0): accuracy reproduced",
          abs(a0["acc"] - ev["eval"]["1024"]["acc"]) <= 0.002, f"{a0['acc']:.4f} vs registered {ev['eval']['1024']['acc']:.4f}"
          f" (GPU-trained, CPU eval); every gap clean: n_contam {a0['n_contam']}")
    sub = RO.reliance(m, W0[:100], "cpu", te0); a100 = RO.acc_split(m, W0[:100], "cpu")["acc"]
    check("C8 theta reliance on a trained map is large", a100 - sub > 0.3, f"acc {a100:.4f} -> {sub:.4f} with type-mean steps")
    torch.manual_seed(45); mu = build("MapWM", te0); mu.eval()
    au = RO.acc_split(mu, W0[:100], "cpu")["acc"]; su = RO.reliance(mu, W0[:100], "cpu", te0)
    check("C8 theta reliance on an untrained model is ~0", abs(au - su) < 0.03, f"{au:.4f} -> {su:.4f}")
    dd = RO.drift_disp(m, RO.walk_set(te0, 20, 10**6 + 1), "cpu", te0)
    check("C8 trained map: few drifting channels, a real per-move displacement", dd["drift"] <= 8 and dd["disp"] > 0.05,
          f"drift {dd['drift']}/64, disp {dd['disp']:.3f} rad, sharp {dd['sharp']:.3f}")

    # C9 trainer reproduction, CPU vs CPU bitwise
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", PYTHONPATH="/home/prashr")
    small = ["--seed", "40", "--epochs", "2", "--n-batches", "3", "--n-steps", "256", "--batch-size", "4",
             "--n-trials", "5", "--device", "cpu", "--data-workers", "3"]
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, "-m", "mapformer.train_tw_ambig", "--arm", "MapWM", "--p-nm", "0",
                        "--output-dir", f"{d}/a"] + small, check=True, env=env, cwd="/home/prashr", capture_output=True)
        subprocess.run([sys.executable, "-m", "mapformer.train_tw_normstep", "--arm", "MapWM", "--output-dir", f"{d}/b"]
                       + small, check=True, env=env, cwd="/home/prashr", capture_output=True)
        A = torch.load(f"{d}/a/MapWM.pt", weights_only=False); B = torch.load(f"{d}/b/MapWM.pt", weights_only=False)
        eqw = all(torch.equal(A["model_state_dict"][k], B["model_state_dict"][k]) for k in B["model_state_dict"])
        ea, eb = json.load(open(f"{d}/a/eval.json")), json.load(open(f"{d}/b/eval.json"))
        check("C9 CPU bitwise: train_tw_ambig --p-nm 0 == train_tw_normstep (MapWM, seed 40, small config)",
              A["losses"] == B["losses"] and eqw and ea["eval"]["256"]["acc"] == eb["eval"]["256"]["acc"],
              f"losses {A['losses']} vs {B['losses']}; weights equal {eqw}; acc {ea['eval']['256']['acc']:.4f} vs "
              f"{eb['eval']['256']['acc']:.4f}")
        # the same, p_nm = 0.3, is run twice: determinism of the new stream on CPU
        for tag in ("c", "d"):
            subprocess.run([sys.executable, "-m", "mapformer.train_tw_ambig", "--arm", "HSR", "--output-dir", f"{d}/{tag}"]
                           + small, check=True, env=env, cwd="/home/prashr", capture_output=True)
        C = torch.load(f"{d}/c/HSR.pt", weights_only=False); D = torch.load(f"{d}/d/HSR.pt", weights_only=False)
        check("C9 CPU determinism: HSR on the p_nm=0.3 task, run twice, identical losses", C["losses"] == D["losses"],
              f"{C['losses']}")

    # C10 the trainer path vs the stored GPU run's first epoch (CPU; the stored schedule: 900 epochs x 98 batches)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tw_ambig_repro import repro
    los, old = repro("cpu", 1)
    check("C10 epoch-1 loss, CPU re-run of the stored TW_NORMSTEP MapWM s10 config vs the stored GPU run",
          abs(los[0] - old[0]) < 0.02, f"{los[0]:.6f} vs {old[0]:.6f} (diff {los[0] - old[0]:+.2e}; "
          "bitwise only on the same device: deferred to the GPU pilot, tw_ambig_repro.py --device cuda:0)")

    # C11 the analysis report on synthetic data with every readout key
    import mapformer.analyze_tw_ambig as AN
    mr, _ = mk("HSR", 46)
    keys = RO.readouts.__code__  # noqa (documentation: keys come from a real readout call below)
    with tempfile.TemporaryDirectory() as d:
        torch.save({"model_state_dict": mr.state_dict(), "losses": [1.0] * 10, "arm": "HSR", "seed": 46,
                    "config": {"args": {"size": 64, "p_nm": 0.3}, "vocab_size": 89}}, f"{d}/HSR.pt")
        tmpl = RO.readouts(f"{d}/HSR.pt", "cpu", n_acc=20)
    rng = np.random.default_rng(1); res = {}
    for a in AN.ARMS:
        for s in AN.SEEDS:
            r = {k: (v if not isinstance(v, float) else float(np.clip(v + 0.01 * rng.standard_normal(), 0, 2)))
                 for k, v in tmpl.items()}
            r.update(acc=float(rng.uniform(0.6, 1)), acc2048=0.8, acc_tm=float(rng.uniform(0.6, 1)), cls="SOLVED",
                     tail=0.01, final_loss=0.01, acc_readout=0.8, arm=a, seed=s)
            if a.startswith("RoPE"):
                r = {k: r[k] for k in ("acc", "acc2048", "acc_tm", "cls", "tail", "final_loss", "acc_readout", "acc_clean",
                                       "acc_contam", "n_clean", "n_contam", "acc_rescored", "arm", "seed")}
            res[f"{a}_s{s}"] = r
    try:
        AN.report(res, AN.SEEDS); AN.report(res, AN.SEEDS[:1]); ok = True
    except Exception as ex:                                   # noqa
        ok = False; print("report raised", repr(ex))
    check("C11 analysis report runs end to end on a synthetic n=8 set and on the n=1 pilot path", ok)
    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")


if __name__ == "__main__":
    main()
