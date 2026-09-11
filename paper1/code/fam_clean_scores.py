#!/usr/bin/env python3
"""Rescore every family-disjoint cell on the held-out reactions that survive the audit.

The family map's gene-rule relation reads the reference network's own rule string, while the
near-duplicate audit reads the project's gene table, which resolves a rule for a small set of
reactions whose record leaves that string empty. Those reactions could not join a family, so the
audit still finds a training-side partner for a few percent of held-out reactions on the
family-disjoint split. This script drops exactly those reactions from each cell's held-out set and
scores the saved predictions again, so the paper can say whether the residue moves any contrast.

Inputs : out/fam/*_preds.npz (every arm's saved scores and held-out mask) and a matched-index file
         written by paper1/code/split_audit.py --family_map ... --matched_out ...
Output : out/fam_clean_scores.json, one record per cell with the full and the cleaned AUROC.
Run on the GPU host from ~/metagnn; CPU only, no GPU.
"""
import os, re, glob, json, argparse
import numpy as np, torch
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.expanduser('~/metagnn/work/crc_624'))
ap.add_argument('--out',  default=os.path.expanduser('~/metagnn/out'))
ap.add_argument('--dir',  default='fam')
ap.add_argument('--matched', default=os.path.expanduser('~/metagnn/work/fam_matched.json'))
A = ap.parse_args()
N_RXN = 10600

y_all = np.asarray(torch.load(os.path.join(A.data, 'activity_pseudolabels.pt'),
                              map_location='cpu', weights_only=False)).astype(int)
M = json.load(open(A.matched))
DROP = {int(k[1:]): np.asarray(v['matched_any'], dtype=int) for k, v in M['folds'].items()}
HELD = {int(k[1:]): np.asarray(v['heldout'], dtype=int) for k, v in M['folds'].items()}

PAT = re.compile(r'^(?P<model>gnn_B|gnn_A|mlp_emb|mlp)_mb(?P<lam>[0-9.]+)_p(?P<pf>\d)_r(?P<rf>\d)(?P<rest>.*)_preds\.npz$')
def mean_auc(P, Y, m):
    return float(np.mean([roc_auc_score(Y[i, m], P[i, m]) for i in range(len(P))]))

R = dict(note='family-disjoint cells scored on the full held-out set and on the subset that survives '
              'the near-duplicate audit; AUROC is the per-patient mean over the cell\'s test patients',
         matched_source=os.path.basename(A.matched), cells={})
for f in sorted(glob.glob(os.path.join(A.out, A.dir, '*_preds.npz'))):
    m = PAT.match(os.path.basename(f))
    if not m:
        continue
    pf, rf = int(m.group('pf')), int(m.group('rf'))
    z = np.load(f, allow_pickle=True); P = z['scores'].astype(np.float64)
    mte = z['heldout_mask'].astype(bool)
    ref = np.zeros(N_RXN, dtype=bool); ref[HELD[rf]] = True
    if not np.array_equal(mte, ref):
        print('held-out mask disagrees with the audit\'s fold, skipped:', os.path.basename(f), flush=True)
        continue
    clean = mte.copy(); clean[DROP[rf]] = False
    Y = np.tile(y_all, (len(P), 1))
    key = os.path.basename(f).replace('_preds.npz', '')
    rec = dict(model=m.group('model'), lam=float(m.group('lam')), pfold=pf, rfold=rf,
               variant=(m.group('rest').replace('_ivr_fam', '').lstrip('_') or 'real'),
               n_patients=int(len(P)), n_heldout=int(mte.sum()), n_dropped=int(len(DROP[rf])),
               n_heldout_clean=int(clean.sum()),
               auroc_heldout=round(mean_auc(P, Y, mte), 6),
               auroc_heldout_clean=round(mean_auc(P, Y, clean), 6),
               prevalence_heldout=round(float(y_all[mte].mean()), 4),
               prevalence_heldout_clean=round(float(y_all[clean].mean()), 4))
    rec['delta_clean_minus_full'] = round(rec['auroc_heldout_clean'] - rec['auroc_heldout'], 6)
    R['cells'][key] = rec
    print(f"{key:48s} full {rec['auroc_heldout']:.4f}  clean {rec['auroc_heldout_clean']:.4f}  "
          f"delta {rec['delta_clean_minus_full']:+.4f}  (dropped {rec['n_dropped']})", flush=True)
json.dump(R, open(os.path.join(A.out, 'fam_clean_scores.json'), 'w'), indent=1)
print('wrote', os.path.join(A.out, 'fam_clean_scores.json'), len(R['cells']), 'cells')
