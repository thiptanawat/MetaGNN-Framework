#!/usr/bin/env python3
"""Per-score Monte Carlo dropout error in each substitution arm, from the saved uncertainties.

Each saved score is a mean over T = 30 dropout passes; its standard error is the saved standard
deviation over sqrt(T). The retrained substitution compares arms patient by patient, so what matters
is whether that error is of the same size in the arms being compared. Writes out/arm_noise.json.
"""
import glob, json, os, re, numpy as np
OUTD = os.path.expanduser("~/metagnn/out"); T = 30
PAT = re.compile(r"^gnn_B_mb0\.0_p(?P<pf>\d)_r(?P<rf>\d)(?:_(?P<mode>mean|zero|indicator|rewire\d+))?_preds\.npz$")
R = {}
for d in ("dh", "ctrl", "rewire"):
    for f in sorted(glob.glob(f"{OUTD}/{d}/*_preds.npz")):
        m = PAT.match(os.path.basename(f))
        if not m: continue
        z = np.load(f, allow_pickle=True); U = z["uncertainties"][:, z["heldout_mask"]]
        mode = m.group("mode") or "real"; pf, rf = m.group("pf"), m.group("rf"); cell = f"p{pf}r{rf}"
        R.setdefault(cell, {})[mode] = dict(mean_se=float((U / np.sqrt(T)).mean()), n_patients=int(U.shape[0]))
        print(cell, mode, "%.5f" % R[cell][mode]["mean_se"], flush=True)
S = {}
for mode in ("real", "mean", "zero", "indicator", "rewire7"):
    v = [c[mode]["mean_se"] for c in R.values() if mode in c]
    if v: S[mode] = dict(n_cells=len(v), mean_se=float(np.mean(v)))
json.dump(dict(cells=R, summary=S, T=T), open(f"{OUTD}/arm_noise.json", "w"), indent=1)
print(json.dumps(S, indent=1))
