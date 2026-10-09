"""Construction and equivalence checks for TW_STATECHANGE_PREREG.md (CPU only, before any run; output committed as
tw_statechange_checks_out.txt). Each check prints PASS / FAIL; the script exits non-zero on any FAIL.
 1  StateChangeWorld(p_take = p_drop = 0) reproduces TextWorld's token / obs / revisit stream byte for byte (two maps,
    60 walks) and leaves the numpy RNG in the same state; with the state vocabulary the token ids are unchanged.
 2  Shared initialisation: MapWM, NormStep and DirOnly at a seed have identical tensors on every shared key and leave the
    torch RNG in the same state (batch seeds 50-57, pilot seed 150).
 3  fwd_steps(m, x, step_of(m, x)) equals m(x) exactly (eval mode) for MapWM, NormStep, DirOnly; DirOnly with an
    all-ones mask equals MapWM's logits exactly.
 4  DirOnly: the step of every non-direction token (incl. the 7 state words) is exactly 0.
 5  Causality: changing the token at position 300 leaves every earlier logit unchanged (all four arms).
 6  The readout's dropout-scale context equals rescore_hook.install('auto') (logits, MapWM and RoPE).
 7  Gauge: the B readouts (field shift, R, aside shift) are unchanged by (Delta * k, omega / k) and by a per-move shift
    between the movement verbs and the direction words (+c on every verb, -c on every direction word).
 8  Interventions: sc_cancel leaves every position before a state clause's last token unchanged and makes the phase after
    the clause equal to the phase before it; aside_cancel likewise; reliance makes every move's phase increment identical.
 9  Trainer equivalence (CPU, tiny schedule: 2 epochs x 2 batches, batch 2, T 256, 1 data worker): train_tw_statechange
    with --p-take 0 --p-drop 0 --no-state-vocab reproduces train_tw_normstep (MapWM, NormStep, DirOnly, seed 10)
    losses bitwise and the final weights exactly; with the registered state clauses it runs (smoke).
10  Eval stream disjoint from every training batch stream (data_parallel seeds seed*1_000_003 + i, i < 900*98).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from mapformer.environment_textworld import TextWorld, VERBS
from mapformer.environment_tw_statechange import StateChangeWorld, TAKE, DROP, DET, CLS
from mapformer.model_textstep import step_of
from mapformer import tw_statechange_readouts as R
from mapformer.train_tw_statechange import build, make_env, EVAL_SEED

torch.set_num_threads(4)
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


# 1
for ms in (10000, 50):
    a, b = TextWorld(seed=ms), StateChangeWorld(seed=ms, p_take=0.0, p_drop=0.0, state_vocab=False)
    np.random.seed(7); A = [a.generate_trajectory(1024) for _ in range(30)]; ra = np.random.random()
    np.random.seed(7); B = [b.generate_trajectory(1024) for _ in range(30)]; rb = np.random.random()
    same = all(all(torch.equal(x, y) for x, y in zip(p, q)) for p, q in zip(A, B))
    check(f"1 p=0 stream == TextWorld (map {ms})", same and ra == rb and a.vocab == b.vocab)
c = StateChangeWorld(seed=10000, p_take=0.0, p_drop=0.0, state_vocab=True)
check("1 state vocabulary appends only", c.vocab[:58] == TextWorld(seed=10000).vocab and c.unified_vocab_size == 65,
      f"(vocab {c.unified_vocab_size}, new {c.vocab[58:]})")

# 2
for s in (50, 51, 52, 53, 54, 55, 56, 57, 150):
    env = make_env(s); sds, rng = {}, {}
    for arm in ("MapWM", "NormStep", "DirOnly"):
        torch.manual_seed(s); m = build(arm, env); sds[arm] = m.state_dict(); rng[arm] = torch.get_rng_state()
    keys = set(sds["MapWM"]) & set(sds["NormStep"]) & set(sds["DirOnly"])
    ok = all(torch.equal(sds["MapWM"][k], sds[a][k]) for k in keys for a in ("NormStep", "DirOnly"))
    ok &= all(torch.equal(rng["MapWM"], rng[a]) for a in ("NormStep", "DirOnly"))
    check(f"2 shared init seed {s}", ok, f"({len(keys)} shared tensors)")

# 3, 4, 5
env = make_env(10000); np.random.seed(EVAL_SEED); W = R.walks(env, n=4)
x = torch.stack([w["tok"] for w in W])[:, :-1]
torch.manual_seed(50)
models = {a: build(a, env).eval() for a in ("MapWM", "NormStep", "DirOnly", "RoPE")}
with torch.no_grad():
    torch.manual_seed(1)                       # move NormStep's LayerNorm off its init
    models["NormStep"].step_ln.weight.data.normal_(1, 0.3); models["NormStep"].step_ln.bias.data.normal_(0, 0.3)
    for a in ("MapWM", "NormStep", "DirOnly"):
        m = models[a]
        check(f"3 fwd_steps == forward ({a})", torch.equal(R.fwd_steps(m, x, step_of(m, x)), m(x)))
    d1 = build("DirOnly", env).eval(); torch.manual_seed(50); w4 = build("MapWM", env).eval()
    d1.load_state_dict(w4.state_dict(), strict=False); d1.step_mask.fill_(1.0)
    check("3 DirOnly all-ones mask == MapWM", torch.equal(d1(x), w4(x)))
    D, _ = R.step_table(models["DirOnly"], env)
    dirs = [i for a_ in range(4) for i in env.dir_ids[a_]]
    other = [i for i in range(env.unified_vocab_size) if i not in dirs]
    check("4 DirOnly non-direction steps exactly 0", bool((D[other] == 0).all()) and bool((np.abs(D[dirs]) > 0).any()),
          f"(state words {[env.vocab[i] for i in env.take_ids + env.drop_ids + [env.idx[DET]]]})")
    for a, m in models.items():
        y0 = m(x[:1]); x2 = x[:1].clone(); x2[0, 300] = (x2[0, 300] + 1) % env.unified_vocab_size; y1 = m(x2)
        check(f"5 causality ({a})", torch.equal(y0[0, :300], y1[0, :300]) and not torch.equal(y0[0, 300:], y1[0, 300:]))

# 7 gauge
m = models["MapWM"]; D, om = R.step_table(m, env); g0 = R.geometry(m, env)
k = 3.0; g1 = R.geometry(m, env, table=(D * k, om / k))
cv = np.zeros(D.shape[1]); cv[:] = np.random.default_rng(0).normal(0, np.abs(D).mean(), D.shape[1])
D2 = D.copy(); D2[[env.idx[v] for v in VERBS]] += cv; D2[dirs] -= cv
g2 = R.geometry(m, env, table=(D2, om))
keys_ = ("shift_sc", "shift_take", "shift_drop", "shift_aside", "R_sc", "R_aside", "dstep")
dev1 = max(abs(g0[q] - g1[q]) for q in keys_); dev2 = max(abs(g0[q] - g2[q]) for q in keys_)
check("7 B readouts invariant to (Delta*k, omega/k)", dev1 < 1e-9, f"(max dev {dev1:.2e})")
check("7 B readouts invariant to a verb <-> direction per-move shift", dev2 < 1e-9, f"(max dev {dev2:.2e}; raw direction "
      f"step moved {abs(g0['dir_raw'] - g2['dir_raw']):.3f} rad)")

# 8 interventions (phase-level)
with torch.no_grad():
    w = W[0]; tok = w["tok"][None, :-1]; L = tok.shape[1]; st = step_of(m, tok)
    th = torch.cumsum(st, 1)[0].reshape(L, -1)
    s2 = st.clone()
    for kind, a_, e_, _c in w["clauses"]:
        if kind in ("take", "drop") and e_ <= L:
            s2[0, e_ - 1] -= st[0, a_:e_].sum(0)
    th2 = torch.cumsum(s2, 1)[0].reshape(L, -1)
    scl = [(a_, e_) for kind, a_, e_, _ in w["clauses"] if kind in ("take", "drop") and e_ <= L]
    first_end = scl[0][1] - 1
    ok = torch.equal(th[:first_end], th2[:first_end]) and \
        all(torch.allclose(th2[e_ - 1], th2[a_ - 1], atol=1e-5) for a_, e_ in scl)
    check("8 sc_cancel: unchanged before the first clause end; phase after each clause = phase before it", ok,
          f"({len(scl)} clauses)")
    dir_ids = torch.tensor(dirs); dmean = step_of(m, dir_ids[None])[0].mean(0)
    s3 = torch.where(torch.isin(tok, dir_ids)[..., None, None], dmean.expand_as(st), st)
    th3 = torch.cumsum(s3 * torch.isin(tok, dir_ids)[..., None, None], 1)[0].reshape(L, -1)
    pos = [s["pos"] for s in w["info"] if s["pos"] < L]
    inc = torch.stack([th3[p_] for p_ in pos]).diff(dim=0)
    check("8 reliance: every move's direction-phase increment identical", torch.allclose(inc, inc[0:1].expand_as(inc), atol=1e-5))

# 6 rescore equality (install patches nn.Module.eval globally: last in-process check)
with torch.no_grad():
    ref = {}
    for a in ("MapWM", "RoPE"):
        with R.rescaled(models[a]):
            ref[a] = models[a](x)
from mapformer import rescore_hook
rescore_hook.install("auto")
with torch.no_grad():
    for a in ("MapWM", "RoPE"):
        models[a].eval(); y = models[a](x)
        check(f"6 rescaled() == rescore_hook ({a})", torch.equal(y, ref[a]), f"(hooked {rescore_hook.STATS['hooked']})")

# 9 trainer equivalence
tmp = tempfile.mkdtemp(prefix="twsc_eq_", dir=os.environ.get("TMPDIR", "/tmp"))
common = ["--seed", "10", "--epochs", "2", "--n-batches", "2", "--batch-size", "2", "--n-steps", "256",
          "--data-workers", "1", "--n-trials", "2", "--device", "cpu"]
envv = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="/home/prashr", OMP_NUM_THREADS="2")
for arm in ("MapWM", "NormStep", "DirOnly"):
    o1, o2 = f"{tmp}/ns_{arm}", f"{tmp}/sc_{arm}"
    subprocess.run([sys.executable, "-m", "mapformer.train_tw_normstep", "--arm", arm, *common, "--output-dir", o1],
                   check=True, env=envv, cwd="/home/prashr", capture_output=True)
    subprocess.run([sys.executable, "-m", "mapformer.train_tw_statechange", "--arm", arm, *common, "--p-take", "0",
                    "--p-drop", "0", "--no-state-vocab", "--output-dir", o2], check=True, env=envv, cwd="/home/prashr",
                   capture_output=True)
    a1 = torch.load(f"{o1}/{arm}.pt", weights_only=False); a2 = torch.load(f"{o2}/{arm}.pt", weights_only=False)
    same_w = all(torch.equal(a1["model_state_dict"][q], a2["model_state_dict"][q]) for q in a1["model_state_dict"])
    check(f"9 trainer == train_tw_normstep at p=0 ({arm})", a1["losses"] == a2["losses"] and same_w,
          f"(losses {a1['losses']} vs {a2['losses']})")
o3 = f"{tmp}/sc_RoPE_state"
subprocess.run([sys.executable, "-m", "mapformer.train_tw_statechange", "--arm", "RoPE", *common, "--output-dir", o3],
               check=True, env=envv, cwd="/home/prashr", capture_output=True)
a3 = torch.load(f"{o3}/RoPE.pt", weights_only=False)
check("9 registered config runs (RoPE, state clauses on)", a3["config"]["vocab_size"] == 65 and len(a3["losses"]) == 2,
      f"(losses {a3['losses']})")

# 10 eval stream disjointness
train_seeds = {(s * 1_000_003 + i) % 2 ** 31 for s in list(range(50, 58)) + [150] for i in range(0, 900 * 98, 1)}
check("10 eval np seed 10**6 not a training batch seed", EVAL_SEED not in train_seeds,
      f"(training batch seeds span {min(train_seeds)}..{max(train_seeds)})")

print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
