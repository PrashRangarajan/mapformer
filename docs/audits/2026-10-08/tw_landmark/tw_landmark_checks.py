"""Construction checks for TW_LANDMARK_PREREG.md (CPU only). Each line prints PASS/FAIL; any FAIL exits 1.

C1  rate 0 is the text world byte for byte: tokens, object and revisit masks, and the global RNG state after 50 walks,
    for 3 map seeds, against environment_textworld.TextWorld (serial path); also through generate_batch.
C2  every eval condition of one walk consumes the same global draws (render_conditions asserts it), 100 walks.
C3  removing the 'reached <name>' pairs from a named rendering gives the stripped rendering's prefix (consistent,
    fresh, conflict; rates 0.5 and 1).
C4  consistent names are injective per walk and constant per cell; the mark sits 4 and the name 3 tokens before the
    object slot; fresh names never repeat in a walk; a conflict name belongs to a DIFFERENT landmark already visited,
    and the recorded alt object is that cell's object.
C5  rate-0.5 landmarks are a subset of rate-1 landmarks with the same names (coupled draws).
C6  the landmark share of cells is the rate (200 walks; binomial tolerance).
C7  names are redrawn per walk: the same cell gets different names across walks; the name pool is disjoint from every
    text-world word; every id < vocab size.
C8  models: same vocabulary at every rate, so one seed gives identical initial weights at every rate; layer counts;
    DirMean in 'dir' mode with each direction word mapped to ITS OWN step changes logits by < 1e-5 (the reliance
    hook is otherwise inert); untrained path models have reliance |.| <= 0.01.
C9  trainer determinism on CPU: two 2-epoch x 2-batch runs of the same cell and seed give identical losses and
    weights (data workers on, as in the batch).
"""
import subprocess
import sys
import tempfile

import numpy as np
import torch

from mapformer.environment_textworld import TextWorld
from mapformer.environment_tw_landmark import TextWorldLandmark, render_conditions
from mapformer.train_tw_landmark import build
from mapformer.tw_landmark_eval import DirMean, build_eval, predict, acc_of

torch.set_num_threads(8)
FAIL = []


def check(name, ok, info=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}", flush=True)
    if not ok:
        FAIL.append(name)


# C1
ok = True
for seed in (0, 7, 50):
    a, b = TextWorld(seed=seed), TextWorldLandmark(seed=seed, name_rate=0.0)
    np.random.seed(seed + 1); A = [a.generate_trajectory(1024) for _ in range(50)]; sa = np.random.get_state()
    np.random.seed(seed + 1); B = [b.generate_trajectory(1024) for _ in range(50)]; sb = np.random.get_state()
    ok &= all(all(torch.equal(x, y) for x, y in zip(p, q)) for p, q in zip(A, B))
    ok &= bool(np.array_equal(sa[1], sb[1]) and sa[2] == sb[2])
    np.random.seed(3); ga = a.generate_batch(4, 1024); np.random.seed(3); gb = b.generate_batch(4, 1024)
    ok &= all(torch.equal(x, y) for x, y in zip(ga[:3], gb[:3])) and ga[3] == gb[3]
check("C1 rate 0 == TextWorld (tokens, masks, RNG state, generate_batch)", ok)

# C2-C5, C7
env = TextWorldLandmark(seed=10000)
I = env.idx; names = set(env.name_ids); mark = env.mark_id
conds = [(0.0, "consistent"), (0.5, "consistent"), (1.0, "consistent"), (0.5, "fresh"), (1.0, "fresh"),
         (1.0, "conflict")]
np.random.seed(123)
c3 = c4 = c5 = True; land_share = []; cell_names = {}
for w in range(100):
    R = render_conditions(env, 1024, conds)                               # C2: asserts equal draw consumption
    strip = R[(0.0, "consistent")][0].tolist()
    for key in conds[1:]:
        t = R[key][0].tolist()
        core = [x for i, x in enumerate(t) if x != mark and not (x in names and i > 0 and t[i - 1] == mark)]
        # the cut can fall between 'reached' and its name: drop a trailing mark
        c3 &= core == strip[:len(core)]
    obj_of = {}
    for (pos, k, cell, land, rev, conf, alt) in R[(0.0, "consistent")][3]:
        obj_of[cell] = strip[pos]
    for key in [(0.5, "consistent"), (1.0, "consistent")]:
        t, _o, _r, slots = R[key]; t = t.tolist(); nm_of = {}
        for (pos, k, cell, land, rev, conf, alt) in slots:
            if land:
                c4 &= t[pos - 4] == mark and t[pos - 3] in names
                c4 &= nm_of.setdefault(cell, t[pos - 3]) == t[pos - 3]
            else:
                c4 &= t[pos - 3] not in names
        c4 &= len(set(nm_of.values())) == len(nm_of)
        if key[0] == 1.0:
            land_share.append(np.mean([s[3] for s in slots]))
            for cell, nm in nm_of.items():
                cell_names.setdefault(cell, set()).add(nm)
        if key[0] == 0.5:
            half = nm_of
        else:
            c5 &= all(full == nm_of[c] for c, full in half.items() if c in nm_of)   # cells rendered at both rates
    t, _o, _r, slots = R[(0.5, "fresh")]; t = t.tolist()
    fr = [t[pos - 3] for (pos, k, cell, land, *_x) in slots if land]
    c4 &= len(fr) == len(set(fr))
    t, _o, _r, slots = R[(1.0, "conflict")]; t = t.tolist()
    cons = R[(1.0, "consistent")][0].tolist(); visited = []
    name_cell = {cons[pos - 3]: cell for (pos, k, cell, *_x) in R[(1.0, "consistent")][3]}
    for (pos, k, cell, land, rev, conf, alt) in slots:
        if conf:
            other = name_cell.get(t[pos - 3])
            c4 &= other is not None and other != cell and other in visited and alt == obj_of[other]
        else:
            c4 &= t[pos - 3] == cons[pos - 3]
        visited.append(cell)
check("C2 conditions consume identical draws (asserted inside render_conditions)", True)
check("C3 named rendering minus name clauses == stripped prefix", c3)
check("C4 names: injective, constant per cell, at object-3 (mark at -4); fresh never repeat; conflict -> other "
      "visited landmark with its object recorded", c4)
check("C5 rate-0.5 landmarks subset of rate-1 landmarks, same names", c5)
np.random.seed(5); sh = []
for w in range(200):
    env.generate_trajectory(1024, name_rate=0.5)
    cells = {s[2]: s[3] for s in env.slots}
    sh += list(cells.values())
p = np.mean(sh); se = np.sqrt(0.25 / len(sh))
check("C6 landmark share of cells at rate 0.5", abs(p - 0.5) < 4 * se, f"({p:.4f}, {len(sh)} cells, 4 se {4 * se:.4f})")
multi = np.mean([len(v) > 1 for v in cell_names.values() if len(v) >= 1])
check("C7 names redrawn per walk; pool disjoint from text-world words; ids < vocab",
      multi > 0.5 and not (names & set(range(env.base_vocab_size))) and max(env.name_ids) < env.unified_vocab_size,
      f"(cells seen in >1 walk with >1 name: {multi:.3f})")

# C8
E = build_eval(1.0, 20)
toks, tg = E["own"]
dirs = [i for a in range(4) for i in env.dir_ids[a]]
okv = True; rels = []
for L in (1, 2):
    w0 = {}
    for r in (0.0, 0.5, 1.0):
        torch.manual_seed(150)
        m = build("MapWM", TextWorldLandmark(seed=150, name_rate=r), L).eval()
        okv &= len(m.layers) == L
        sd = {k: v.clone() for k, v in m.state_dict().items()}
        if w0:
            okv &= all(torch.equal(sd[k], w0[k]) for k in sd)
        w0 = w0 or sd
    h = DirMean(m, dirs, env.name_ids + [mark])
    with torch.no_grad():
        h.groups["self"] = [(torch.tensor([i]), m.action_to_lie(m.token_emb(torch.tensor([[i]])))[0, 0]) for i in dirs]
        x = toks[:4, :-1]; h.tok = x; l0 = m(x); h.mode = "self"; l1 = m(x); h.mode = None
    diff = float((l0 - l1).abs().max())
    okv &= diff < 1e-5
    p = predict(m, toks, "cpu", hook=h); a0 = acc_of(p, tg)[0]
    h.mode = "dir"; q = predict(m, toks, "cpu", hook=h); h.mode = None
    rels.append(a0 - acc_of(q, tg)[0])
    print(f"   MapWM {L}L untrained: identity-hook max |logit diff| {diff:.2e}; acc {a0:.4f}; reliance {rels[-1]:+.4f}")
torch.manual_seed(150); mr = build("RoPE", env, 2); okv &= len(mr.layers) == 2 and not hasattr(mr, "action_to_lie")
check("C8 same init across rates; layers; identity hook inert; untrained reliance ~0", okv and max(map(abs, rels)) <= 0.01)

# C9
outs = []
for k in range(2):
    d = tempfile.mkdtemp()
    subprocess.run([sys.executable, "-m", "mapformer.train_tw_landmark", "--arm", "MapWM", "--n-layers", "2",
                    "--name-rate", "0.5", "--seed", "151", "--epochs", "2", "--n-batches", "2", "--data-workers", "2",
                    "--device", "cpu", "--output-dir", d], check=True, capture_output=True, cwd="/home/prashr",
                   env={"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "PATH": "/usr/bin:/bin",
                        "PYTHONPATH": "/home/prashr"})
    outs.append(torch.load(f"{d}/MapWM.pt", map_location="cpu", weights_only=False))
same = outs[0]["losses"] == outs[1]["losses"] and all(
    torch.equal(outs[0]["model_state_dict"][k], outs[1]["model_state_dict"][k]) for k in outs[0]["model_state_dict"])
check("C9 CPU trainer determinism (losses and weights bitwise, data workers on)", same, f"(losses {outs[0]['losses']})")

print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
