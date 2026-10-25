"""G-3 A2 post hoc (analysis): per-channel cancellation vs the initial past-pi set, from the A2 checkpoints.
Usage (from /home/prashr): python3 mapformer/docs/audits/2026-10-25/g3_a2_exp_mech.py A2 exp,softplus,abs,signed
The past-pi-start column uses exp init (Delta = 1; 2 omega_0 > pi) for every arm except softplus (2 ln2 omega_0 > pi);
it is meaningful for exp and softplus only."""
import glob, math, sys, torch, numpy as np
sys.path.insert(0, '/home/prashr')
from mapformer.construct_ssm_sign import load_source
from mapformer.model import PathIntegrator
TP = 2 * math.pi
om0 = PathIntegrator(2, 32, 32).omega.detach().double()          # init omega (H, N)
root = '/home/prashr/mapformer/runs/ssm_sign/p0/' + sys.argv[1]
for arm in sys.argv[2].split(','):
    print("==", sys.argv[1], arm)
    agg = []
    for d in sorted(glob.glob(f'{root}/{arm}_moves_m32_*s21*[0-9]')):
        pt = glob.glob(d + '/*.pt')[0]
        m, b = load_source(pt)
        with torch.no_grad():
            st = m.step_table()[m.act].double()                    # (2, H, N)
            om = m.path_integrator.omega.detach().double()
            A = st * om
            S = A.sum(0)
            clock = torch.angle(torch.exp(1j * S))
            big = torch.angle(torch.exp(1j * A[0])).abs() > 0.05
            canc = (clock.abs() < 0.1) & big
            wrapc = canc & (S.abs() > 1.0)
            past0 = (2 * (math.log(2) if arm == "softplus" else 1.0) * om0.abs() > math.pi)
            pastF = (S.abs() > math.pi) & big
            k = torch.round(S / TP)[canc]
            nb = int(big.sum())
            r = dict(seed=d[-4:], nonblind=nb, cancel=int(canc.sum()), wrapc=int(wrapc.sum()),
                     pastpi_end=int(pastF.sum()), canc_from_past0=int((canc & past0).sum()),
                     canc_not_past0=int((canc & ~past0).sum()), turns=sorted(set(k.abs().int().tolist())),
                     omega_ratio_med=float((om.abs() / om0).median()),
                     step_ratio_med=float((st[0] / st[1]).abs()[big].median()) if big.any() else float('nan'),
                     sum_med=float(S.abs()[big].median()))
            agg.append(r)
            print(r)
    print(' totals: cancel %d of nonblind %d; by wrap %d; cancelling channels that started past pi %d, others %d' % (
        sum(a['cancel'] for a in agg), sum(a['nonblind'] for a in agg), sum(a['wrapc'] for a in agg),
        sum(a['canc_from_past0'] for a in agg), sum(a['canc_not_past0'] for a in agg)))
print('init: channels per head with 2*omega0 > pi:', int((2 * om0[0] > math.pi).sum()), 'of 32')
