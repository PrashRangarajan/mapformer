"""What does the clock of the clock seeds count: moves, sentences or words? (POST HOC, CPU.) Per clause (one move), its
net omega-scaled phase is regressed on [direction one-hot, aside sentence present, number of fillers, adverb present];
in the drifting channels (tw_step_account.json) the per-move tick (mean of the 4 direction intercepts) is compared with
the extra phase an aside SENTENCE, a filler WORD and an adverb WORD add. A sentence counter would give aside ~ tick; a
word counter filler ~ adverb ~ tick / (words per clause); a move counter (= path length) all three ~ 0.
Held-out map, np seed 424242, 40 walks x 1024 words (the eval stream of tw_step_account.py)."""
import sys, json, numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, '/home/prashr/mapformer/docs/paper/expansion/scripts/language')
from tw_lib import TextWorld, ROLES, gen, load, ck_tw, ck_ns, omega, step_table
D_ = '/home/prashr/mapformer/docs/paper/expansion/scripts/language'
J = json.load(open(f'{D_}/tw_step_account.json'))
te = TextWorld(size=64, seed=10000); np.random.seed(424242); W = [gen(te, 1024) for _ in range(40)]; V = len(te.vocab)
rows = []
for g in W:
    cl, ro = g['clause'], g['role']; ncl = cl.max()
    for c in range(ncl):
        r = ro[cl == c]
        rows.append((g, c, (r == ROLES.index('anoun')).any(), (r == ROLES.index('fill')).sum(), (r == ROLES.index('adv')).any()))
for name, ck in [(f'tw_s{s}', ck_tw(s)) for s in (0, 2, 3, 4)] + [(f'MapWM_s{s}', ck_ns('MapWM', s)) for s in (11, 13, 14, 15)] + \
        [(f'NormStep_s{s}', ck_ns('NormStep', s)) for s in (11, 16)]:
    m, _ = load(ck); om = omega(m).numpy().reshape(-1); st = step_table(m, V).reshape(V, -1).numpy() * om
    drift = np.array(J[name]['drift_obs']) > 1.0
    X, Y = [], []
    for g, c, a, f, ad in rows:
        X.append(np.r_[np.eye(4)[g['dirs'][c]], float(a), float(f), float(ad)]); Y.append(st[g['tok'][g['clause'] == c]].sum(0))
    X = np.array(X); Y = np.array(Y)[:, drift]
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    tick = B[:4].mean(0); nt = np.linalg.norm(tick)
    print(f'{name:12s} drift ch {int(drift.sum())}: |tick| per move {nt:.3f} rad | aside sentence {np.linalg.norm(B[4]) / nt:.3f}'
          f' filler word {np.linalg.norm(B[5]) / nt:.3f} adverb word {np.linalg.norm(B[6]) / nt:.3f} (x tick)'
          f' | cos(aside, tick) {B[4] @ tick / (np.linalg.norm(B[4]) * nt):+.2f}')
