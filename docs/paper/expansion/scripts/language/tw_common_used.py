"""Why do the clock seeds depend more on non-direction words? (POST HOC, CPU.) Per text-world path seed: the direction
words' common component c (omega-scaled) in the channels retrieval uses (clause tick 0 mod pi; usage from
tw_step_account.json), relative to a move, against the registered secondary S3 cost of zeroing every non-direction step
(TEXTWORLD_SECONDARY.json). Zeroing the other words leaves c as a per-clause tick in exactly those channels."""
import sys, json, numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, '/home/prashr/mapformer/docs/paper/expansion/scripts/language')
from tw_lib import TextWorld, load, ck_tw, omega, step_table, wrap
D_ = '/home/prashr/mapformer/docs/paper/expansion/scripts/language'
J = json.load(open(f'{D_}/tw_step_account.json')); S = json.load(open('/home/prashr/mapformer/TEXTWORLD_SECONDARY.json'))
te = TextWorld(size=64, seed=10000); V = len(te.vocab); xs, ys = [], []
for s in range(8):
    name = f'tw_s{s}'; m, _ = load(ck_tw(s)); om = omega(m).numpy().reshape(-1)
    st = step_table(m, V).reshape(V, -1).numpy() * om
    A = np.stack([st[te.dir_ids[a]].mean(0) for a in range(4)]); c = A.mean(0)
    u = np.array(J[name]['usage']); tau = np.array(J[name]['tau']); used = np.abs(wrap(2 * tau)) / 2 < 0.05
    w = u * used; rel = float((np.abs(c) * w).sum() / (np.abs(A[0] - c) * w).sum())
    cost = S[f's{s}']['S3']['full'] - S[f's{s}']['S3']['non_direction_zeroed']
    xs.append(rel); ys.append(cost)
    print(f'{name}: |c| / |move| in used channels {rel:.3f} | cost of zeroing non-direction steps {cost:.3f}')
print(f'Spearman over 8 seeds: {float(np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1]):+.3f}')
