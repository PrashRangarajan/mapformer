import json, glob, re, numpy as np
groups = {}
for f in sorted(glob.glob('toy_runs/*.json')):
    k = re.sub(r'_s\d+\.json$', '', f.split('/')[-1]); r = json.load(open(f)); h = r['hist']
    groups.setdefault(k, []).append((r['final_acc'], h[-1]['wrap_share'], h[-1]['opp_sign'], h[-1]['sum_err_med'], h[0]['loss'], h[1]['wrap_share'] if len(h) > 1 else np.nan))
for k, v in groups.items():
    v = np.array(v)
    print(f"{k:24s} n={len(v)} SOLVED(acc>=0.99) {int((v[:,0]>=0.99).sum())}/{len(v)} acc median {np.median(v[:,0]):.3f} [{v[:,0].min():.2f},{v[:,0].max():.2f}] "
          f"wrap share end {np.median(v[:,1]):.2f} (step500 {np.nanmedian(v[:,5]):.2f}) opp-sign end {np.median(v[:,2]):.2f} median |sum err| {np.median(v[:,3]):.3f} init loss {np.median(v[:,4]):.2f}")
