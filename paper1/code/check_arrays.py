#!/usr/bin/env python3
"""Check prediction arrays against the result files written beside them.

    python3 check_arrays.py ARRAY_DIR CELL_DIR DATA_DIR [--out FILE]

A training cell writes <cell>.json and <cell>_preds.npz. For every array in ARRAY_DIR the mean per-patient AUROC on
the cell's held-out reactions is recomputed from the stored scores and the labels (the label vector of DATA_DIR,
activity_pseudolabels.pt, or the per-patient labels a synthetic cell stores) and compared with
heldout_AUROC_perpatient in CELL_DIR/<cell>.json, to the four decimals the result file carries. The numbers of test
patients and of held-out reactions are compared as well. With --out the record of every array is written as JSON.
Exits nonzero if an array has no result file, if a value differs or if no array was found.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

ap = argparse.ArgumentParser()
ap.add_argument('array_dir'); ap.add_argument('cell_dir'); ap.add_argument('data_dir'); ap.add_argument('--out')
A = ap.parse_args()
y = np.asarray(torch.load(os.path.join(A.data_dir, 'activity_pseudolabels.pt'), map_location='cpu', weights_only=False)).astype(int)

rows, bad = [], 0
for f in sorted(glob.glob(os.path.join(A.array_dir, '*_preds.npz'))):
    name = os.path.basename(f); rfile = os.path.join(A.cell_dir, name[:-len('_preds.npz')] + '.json')
    if not os.path.exists(rfile):
        print('no result file for', name); bad += 1; continue
    r = json.load(open(rfile)); z = np.load(f, allow_pickle=True); P, mt = z['scores'], z['heldout_mask']
    Y = z['labels'] if 'labels' in z.files else np.tile(y, (len(P), 1))
    got = round(float(np.mean([roc_auc_score(Y[i, mt], P[i, mt]) for i in range(len(P))])), 4)
    ok = got == r['heldout_AUROC_perpatient'] and len(P) == r['n_test'] and int(mt.sum()) == r['n_heldout_rxn']
    bad += not ok
    rows.append(dict(array=name, bytes=os.path.getsize(f), sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest(),
                     n_test=int(len(P)), n_heldout_rxn=int(mt.sum()), recomputed=got, result_file=r['heldout_AUROC_perpatient'],
                     result_file_sha256=hashlib.sha256(open(rfile, 'rb').read()).hexdigest(), agree=bool(ok)))
n = len(rows); agree = sum(x['agree'] for x in rows)
print('%d arrays, %d agree with their result file to four decimals, %d do not or have no result file' % (n, agree, bad))
if A.out:
    json.dump(dict(data=os.path.basename(os.path.normpath(A.data_dir)), n_arrays=n, n_agree=agree, arrays_bytes=sum(x['bytes'] for x in rows), arrays=rows),
              open(A.out, 'w'), indent=1)
sys.exit(1 if bad or not n else 0)
