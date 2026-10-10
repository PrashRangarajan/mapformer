"""TinyStories pilot readouts, as declared in TINYSTORIES_PILOT.md (post hoc on the best checkpoints; no verdicts).
  PYTHONPATH=/home/prashr python3 docs/audits/2026-10-09/tinystories_probe.py [device]
1 val loss per arm vs floors; 2 step table (MapWM); 3 opposition pairs vs random pairs; 4 causal substitutions;
5 '<eos>' step. The step of token v is omega * W_out W_in emb(v) (context-free), so substitutions replace rows of the
per-token table D[v] = W_out W_in emb(v) and feed D[tokens] to the path integrator in place of the module's output."""
import json, sys, numpy as np, torch
from mapformer.train_tinystories import build, evaluate, D as DATA

R = "/home/prashr/mapformer/runs/tinystories/p0"
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
import os
SEEDS = [int(x) for x in os.environ.get("TS_SEEDS", "0 1 2").split()]   # smoke test only
NB = int(os.environ.get("TS_NB", "200"))
itos = json.load(open(f"{DATA}/vocab.json")); V = len(itos); idx = {w: i for i, w in enumerate(itos)}
tr = np.memmap(f"{DATA}/train.bin", dtype=np.uint16, mode="r"); val = np.memmap(f"{DATA}/val.bin", dtype=np.uint16, mode="r")
freq = np.bincount(np.asarray(tr[:50_000_000]), minlength=V).astype(float); freq /= freq.sum()
FLOORS = "unigram 5.574, bigram 3.603, trigram 2.935"

FUNCTION = """the a an and but or so then because if when while as that this these those it its he she they we you i
me him her them us my your his their our to of in on at for with from by about into onto over under up down out off
was were is are am be been being had has have do did does not no can could will would should must may might just very
too also there here what who where why how all some any one every each other more most than like now again only""".split()
MOTION = """go goes went gone going come came comes coming walk walked walks walking run ran runs running jump jumped
jumps climb climbed climbs fly flew flies swim swam swims move moved moves return returned returns leave left leaves
arrive arrived arrives enter entered enters""".split()
SPATIAL = "up down in out into outside inside back away home there here over under".split()
PAIRS = [("up", "down"), ("in", "out"), ("inside", "outside"), ("came", "went"), ("come", "go"), ("open", "close"),
         ("opened", "closed"), ("start", "stop"), ("forward", "back"), ("left", "right"), ("push", "pull"),
         ("give", "take"), ("on", "off")]
PUNCT = {i for i, w in enumerate(itos) if len(w) == 1 and not w.isalnum() and w not in ("\n",)}
NL = {idx["\n"], idx["<eos>"]}
FUNC = {idx[w] for w in FUNCTION if w in idx}
MOT = {idx[w] for w in MOTION if w in idx}
SPA = {idx[w] for w in SPATIAL if w in idx} - FUNC
QUOTE = {idx['"']} if '"' in idx else set()


def load(name, s):
    ck = torch.load(f"{R}/{name}_s{s}.best.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = build(name, c["vocab"], c["dim"], c["heads"], c["n_layers"], c["seq_len"], c["rank"])
    m.load_state_dict(ck["state_dict"]); return m.to(dev).eval(), c, ck["iter"]


def table(m):
    with torch.no_grad():
        return m.action_to_lie(m.token_emb.weight[None])[0]          # (V, H, nb) delta per token, before omega


class Swap(torch.nn.Module):                                           # feeds a fixed per-token table to the integrator
    def __init__(self, Dt): super().__init__(); self.Dt = Dt; self.tok = None
    def forward(self, x): return self.Dt[self.tok]


def eval_with(m, c, Dt, n=None):
    n = n or NB
    orig = m.action_to_lie
    sw = Swap(Dt.to(dev)); m.action_to_lie = sw
    h = m.register_forward_pre_hook(lambda mod, args: setattr(sw, "tok", args[0]))
    try:
        return evaluate(m, val, c["batch_size"], c["seq_len"], dev, n)
    finally:
        h.remove(); m.action_to_lie = orig


rng = np.random.default_rng(0)
out = {}
print(f"floors (val, nats/token): {FLOORS}")
for name in ("Vanilla", "RoPE"):
    for s in SEEDS:
        m, c, it = load(name, s)
        v = evaluate(m, val, c["batch_size"], c["seq_len"], dev, NB)
        out[f"{name}_s{s}"] = {"val": v, "iter": it}
        print(f"{name:8s} s{s}: val {v:.4f} nats/token (best checkpoint, iter {it})")
        if name != "Vanilla":
            continue
        Dt = table(m); om = m.path_integrator.omega.detach().to(Dt.device)
        st = (Dt * om).reshape(V, -1).cpu().numpy()                  # radians per token
        n_ = np.linalg.norm(st, axis=1); w = freq
        mean = (w[:, None] * st).sum(0); clock = float(np.linalg.norm(mean) / (w * n_).sum())
        tot = (w * n_).sum()
        share = {k: float((w * n_)[list(S)].sum() / tot) for k, S in
                 (("punct", PUNCT - QUOTE), ("quote", QUOTE), ("newline/eos", NL), ("function", FUNC),
                  ("motion", MOT), ("spatial", SPA))}
        share["rest"] = 1 - sum(share.values())
        top = [i for i in np.argsort(-n_) if w[i] > 1e-5][:30]
        rel = n_ / (w * n_).sum()                                      # step norm relative to the mean step norm
        d = st - mean; nd = np.linalg.norm(d, axis=1)
        opp = lambda a, b: float(np.linalg.norm(d[a] + d[b]) / ((nd[a] + nd[b]) / 2))
        pairs = {f"{a}/{b}": opp(idx[a], idx[b]) for a, b in PAIRS if a in idx and b in idx}
        common = np.nonzero(w > 1e-4)[0]
        base = np.array([opp(*rng.choice(common, 2, replace=False)) for _ in range(3000)])
        # causal substitutions
        Dm = (torch.tensor(w, dtype=Dt.dtype, device=Dt.device)[:, None, None] * Dt).sum(0)
        clockD = Dm.expand_as(Dt).clone()
        content = [i for i in range(V) if i not in PUNCT and i not in NL and i not in FUNC]
        contD = Dt.clone(); contD[content] = Dm
        L_clock = eval_with(m, c, clockD) - v
        L_zero = eval_with(m, c, torch.zeros_like(Dt)) - v
        L_cont = eval_with(m, c, contD) - v
        L_id = eval_with(m, c, Dt) - v                                 # must be 0: the swap itself changes nothing
        eos = float(rel[idx["<eos>"]])
        out[f"{name}_s{s}"].update(dict(clock=clock, share=share, pairs=pairs, base_med=float(np.median(base)),
                                       base_p5=float(np.percentile(base, 5)), L_clock=L_clock, L_zero=L_zero,
                                       L_content=L_cont, L_identity=L_id, eos_rel=eos,
                                       top=[(itos[i], float(rel[i])) for i in top]))
        print(f"   clock share {clock:.3f} | share of step: " + ", ".join(f"{k} {v_:.2f}" for k, v_ in share.items()))
        print("   largest steps (x mean step): " + " ".join(f"{itos[i]!r}:{rel[i]:.1f}" for i in top).replace("'\\n'", "NL"))
        print("   opposition (common removed; 0 cancel, 2 same): " + "  ".join(f"{k} {x:.2f}" for k, x in pairs.items())
              + f"  | random pairs median {np.median(base):.2f}, 5th pct {np.percentile(base, 5):.2f}")
        print(f"   causal (val loss change): mean-step clock {L_clock:+.4f} | content words -> mean {L_cont:+.4f} | "
              f"zero steps {L_zero:+.4f} | identity swap {L_id:+.1e} | <eos> step {eos:.2f}x mean")
        del m; torch.cuda.empty_cache() if dev.startswith("cuda") else None
va_ = [out[f"Vanilla_s{s}"]["val"] for s in SEEDS]; ro = [out[f"RoPE_s{s}"]["val"] for s in SEEDS]
print(f"MapWM - RoPE val (per seed): " + " ".join(f"{a - b:+.4f}" for a, b in zip(va_, ro))
      + f"; mean {np.mean(va_) - np.mean(ro):+.4f} (n = 3, no test claimed)")
if SEEDS == [0, 1, 2] and NB == 200:
    json.dump(out, open("/home/prashr/mapformer/runs/tinystories/PROBE.json", "w"), indent=1)
