"""Construction checks and gate for RANK_TESTS_PREREG.md (CPU; run from /home/prashr with PYTHONPATH=/home/prashr).
Ring (experiment 1): C1 matched init across R1 / R2 / R1DF and MapWM's base; C2 R2 equals model_rank_perhead's per-head
rank 2; C3 R1DF's step table (observation rows 0, -1 action = -(+1 action)) and gradients reach its parameters; C4 the
T1 drift readout on R1DF is exactly 0; G1 wrap-only revisit share and floors on the ring stream.
Escape (experiment 2): C5 lifted checkpoints equal their rank-2 source in float64 and load through train_variant's
--init-from checks (dry run of one batch on CPU happens in the pilot script, not here)."""
import sys, numpy as np, torch
from mapformer.environment_cancel import GridWorldCancel
from mapformer.model_ring_rank import MapFormerWM_Ring_R1, MapFormerWM_Ring_R2, MapFormerWM_Ring_R1DF, MapFormerWM_Ring_R1AO
from mapformer.train_ring_rank import build as ring_build
from mapformer.model_rank_perhead import MapFormerWM_PerHead
from mapformer.train_variant import VARIANT_MAP
from mapformer.ring_rank_readouts import drift_kappa

ok_all = True
def report(name, ok, extra=""):
    global ok_all
    ok_all &= bool(ok); print(f"{'PASS' if ok else 'FAIL'} {name} {extra}")

env = GridWorldCancel(size=32, seed=0, p_plus=0.5)
kw = dict(vocab_size=env.unified_vocab_size, d_model=128, n_heads=2, n_layers=1, grid_size=32)
for s in (20, 27, 120):
    torch.manual_seed(s); base = VARIANT_MAP["Vanilla"](**kw).state_dict()
    torch.manual_seed(s); r1 = MapFormerWM_Ring_R1(**kw).state_dict()
    torch.manual_seed(s); r2 = MapFormerWM_Ring_R2(**kw).state_dict()
    torch.manual_seed(s); df = MapFormerWM_Ring_R1DF(**kw, action_ids=(0, 1)).state_dict()
    torch.manual_seed(s); ph = MapFormerWM_PerHead(**kw).state_dict()
    nb = [k for k in base if "action_to_lie" not in k]
    same = all(torch.equal(base[k], r1[k]) and torch.equal(base[k], r2[k]) and torch.equal(base[k], df[k]) for k in nb)
    report(f"C1 s{s} non-bottleneck weights equal across MapWM base, R1, R2, R1DF ({len(nb)} tensors)", same)
    report(f"C1 s{s} R1 and R1DF identical (bottleneck included)", r1.keys() == df.keys() and all(torch.equal(r1[k], df[k]) for k in r1))
    report(f"C2 s{s} R2 equals model_rank_perhead.MapFormerWM_PerHead", r2.keys() == ph.keys() and all(torch.equal(r2[k], ph[k]) for k in r2))
    shapes = {k: tuple(v.shape) for k, v in r1.items() if "action_to_lie" in k}
    print(f"     R1 bottleneck shapes {shapes}; R2 {[tuple(v.shape) for k, v in r2.items() if 'action_to_lie' in k]}")

torch.manual_seed(5); m = MapFormerWM_Ring_R1DF(**kw, action_ids=(0, 1))
T = m.step_table()
report("C3 R1DF observation and blank rows of the step table are 0", float(T[2:].abs().max()) == 0.0)
report("C3 R1DF -1 action step = -(+1 action step)", torch.equal(T[1], -T[0]) and float(T[0].abs().max()) > 0)
tok = torch.from_numpy(np.stack([env.generate_trajectory(64)[0][:-1].numpy() for _ in range(4)]))
out = m(tok); out.logsumexp(-1).mean().backward()
g = {n: (p.grad is not None and float(p.grad.abs().max()) > 0) for n, p in m.named_parameters() if "action_to_lie" in n or "token_emb" in n}
report("C3 gradients reach R1DF's bottleneck and embeddings", all(g.values()), str(g))
k = drift_kappa(m, kw["vocab_size"], n_obs_types=16, p_empty=0.5)
report("C4 R1DF drift readout kappa = 0 on both heads", max(k) == 0.0, f"kappa {k}")
torch.manual_seed(5); m1 = MapFormerWM_Ring_R1(**kw)
print(f"     R1 at init: kappa {[round(x, 3) for x in drift_kappa(m1, kw['vocab_size'], 16, 0.5)]} (untrained, for scale)")

# Amendment 1 checks: R1AO (observation steps 0, actions free; same init as R1), R2S (= Vanilla, shared rank 2)
for sd in (20, 27):
    torch.manual_seed(sd); ao = ring_build("R1AO", env, 32); torch.manual_seed(sd); r1m = ring_build("R1", env, 32)
    A = ao.step_table()
    report(f"C6 s{sd} R1AO: same state as R1, observation rows 0, action rows free",
           all(torch.equal(ao.state_dict()[k], r1m.state_dict()[k]) for k in r1m.state_dict()) and float(A[2:].abs().max()) == 0
           and not torch.allclose(A[0], -A[1]))
    torch.manual_seed(sd); r2s = ring_build("R2S", env, 32).state_dict()
    torch.manual_seed(sd); van = VARIANT_MAP["Vanilla"](**kw).state_dict()
    report(f"C6 s{sd} R2S equals Vanilla (shared rank 2)", r2s.keys() == van.keys() and all(torch.equal(r2s[k], van[k]) for k in van))

# G1 (Amendment 1: calls the task code): wrap-only revisits on the eval stream (np seed 10^6, 400 walks, T = 128). The
# unwrapped coordinate is rebuilt from the action tokens (0 -> +1, 1 -> -1) starting at the first visited cell.
genv = GridWorldCancel(size=32, seed=10000, p_plus=0.5); np.random.seed(10 ** 6)
n_rev = n_wrap = n_t = 0
for _ in range(400):
    tok, _om, rev = genv.generate_trajectory(128)
    acts = tok[0::2].numpy(); cells = [p[0] for p in genv.visited_locations]
    u = cells[0]; seen = {cells[0]: {u}}; n_t += 1
    for t in range(1, 128):
        u += 1 if acts[t] == 0 else -1
        assert u % 32 == cells[t]
        c = cells[t]
        if c in seen:
            n_rev += 1; n_wrap += u not in seen[c]
        seen.setdefault(c, set()).add(u); n_t += 1
print(f"G1 ring eval stream via GridWorldCancel (p_plus 0.5, T = 128, 400 walks): revisits {n_rev / n_t:.3f} of steps; "
      f"wrap-only {n_wrap / max(1, n_rev):.3f} of revisits")
print("G1 floor quoted from CANCEL_PREREG.md (better of best constant and n-gram 1-5 = 0.554 at p_plus 0.5; computed there on its own stream, not recomputed here)")

# C5: lifted checkpoints (built by build_rank_lift into the run dir) equal their source in float64
import glob
paths = sorted(glob.glob("/home/prashr/mapformer/runs/rank_tests/escape/lifted/p0/Vanilla_r4mi_s*/Vanilla_r4mi.pt"))
for p in paths:
    b = torch.load(p, map_location="cpu", weights_only=False); c = b["config"]
    report(f"C5 {p.split('/')[-2]}: fp64 logit diff {c['lift_max_logit_diff_fp64']:.1e} < 1e-9; variant {b['variant']}; losses empty",
           c["lift_max_logit_diff_fp64"] < 1e-9 and b["variant"] == "Vanilla_r4mi" and b["losses"] == [])
if not paths:
    print("C5 SKIPPED: run `python3 -m mapformer.build_rank_lift --out runs/rank_tests/escape/lifted` first")
# C7 (Amendment 1): F checkpoints = the trained rank-2 network with the bottleneck Vanilla_r4mi draws at the seed
for s_ in range(8):
    pth = f"/home/prashr/mapformer/runs/rank_tests/escape/fresh/p0/Vanilla_r4mi_s{s_}/Vanilla_r4mi.pt"
    try:
        b = torch.load(pth, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"C7 SKIPPED s{s_}: build with `python3 -m mapformer.build_rank_lift --fresh --out runs/rank_tests/escape/fresh`"); continue
    src = torch.load(f"/home/prashr/mapformer/runs/rank_mi/p0/Vanilla_s{s_}/Vanilla.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    c = b["config"]; kw4 = dict(vocab_size=c["vocab_size"], d_model=c["d_model"], n_heads=c["n_heads"], n_layers=c["n_layers"], grid_size=c["grid_size"])
    torch.manual_seed(s_); init = VARIANT_MAP["Vanilla_r4mi"](**kw4).state_dict(); sd = b["model_state_dict"]
    bott = all(torch.equal(sd[k], init[k]) for k in ("action_to_lie.w_in.weight", "action_to_lie.w_out.weight"))
    rest = all(torch.equal(sd[k], src[k]) for k in src if "action_to_lie" not in k)
    report(f"C7 s{s_} F: bottleneck = Vanilla_r4mi init at the seed, every other weight = the trained rank-2 source", bott and rest)
print("ALL PASS" if ok_all else "SOME CHECK FAILED"); sys.exit(0 if ok_all else 1)
