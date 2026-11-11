#!/usr/bin/env python3
"""Write, or check, the archive of the cohort order used at collection (cohort_order.npz).

The percentile a prompt prints is a rank from NumPy's default argsort, which is not stable, and the
panels of the patient-level probe are top-k selections by the same sort. Where values tie, the
result depends on the processor's vector instruction set. This script archives what the collection
host computed, so that cohort_order.py can return it on any machine.

    python3 code/export_cohort_order.py            write cohort_order.npz (needs a machine whose
                                                   default argsort orders ties as the collection
                                                   host did; the script refuses otherwise)
    python3 code/export_cohort_order.py --check    check the committed archive; run by reproduce.sh

The evidence that an order is the collection host's is in the release itself: the positive-control
records stored the percentile each prompt printed (results/pc*/raw.json, fields pct and pct2).
Both modes compare every one of them with the archived order.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

import _paths as PATHS
import cohort_order as CO

D = np.load(PATHS.data('rxn_context.npz'), allow_pickle=True)
X, HAS = D['X'], D['has'].astype(bool)
N = X.shape[0]
ALIGNED = np.array([r['aligned'] for r in PATHS.reactions()])
EXPR = np.where(HAS & ALIGNED)[0]      # reactions that carry expression and can enter a prompt
ARCHIVE = os.path.join(PATHS.ROOT, 'cohort_order.npz')


def stored_percentiles():
    """(patient, reaction, percentile) for every percentile a positive-control record stored."""
    out = []
    for f in sorted(glob.glob(os.path.join(PATHS.ROOT, 'results', 'pc*', 'raw.json'))):
        for key, rec in json.load(open(f)).items():
            if not isinstance(rec, dict) or rec.get('pct') is None:
                continue
            parts = key.split('|')
            out.append((int(parts[1]), int(parts[2]), float(rec['pct'])))
            if rec.get('pct2') is not None and parts[3] not in ('None', ''):
                out.append((int(parts[1]), int(parts[3]), float(rec['pct2'])))
    return out


def compare_with_records(pct):
    recs = stored_percentiles()
    bad = [(p, r) for p, r, v in recs if abs(pct[p, r] - v) > 1e-9]
    tied = sum(int((X[:, r] == X[p, r]).sum()) > 1 for p, r, _ in recs)
    return len(recs), tied, len(bad)


def msi_design():
    des = json.load(open(PATHS.run('msi', 'design.json')))
    return np.array(des['keep']), np.array(des['y']), np.array(des['panel']), int(des['k'])


def fold_scores():
    """For each fold of the inductive reference: the across-patient SD of every expression-bearing
    reaction over the training patients, the score its panel is selected by."""
    keep, y, panel, k = msi_design()
    Xk = X[keep]
    folds = StratifiedKFold(5, shuffle=True, random_state=2024).split(Xk[:, panel], y)
    return [Xk[np.ix_(tr, EXPR)].std(axis=0) for tr, _ in folds], k


def write():
    order = np.argsort(X, axis=0)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(N)[:, None], axis=0)
    n, tied, bad = compare_with_records(100.0 * ranks / (N - 1))
    if bad:
        sys.exit('this machine does not order tied values as the collection host did: %d of the %d percentiles '
                 'stored at collection differ from its ranks; the archive was not written' % (bad, n))
    stable = np.argsort(X, axis=0, kind='stable')
    cols = np.where((order != stable).any(axis=0))[0]
    keep, y, panel, k = msi_design()
    chosen = np.array(sorted(EXPR[np.argsort(-X[:, EXPR].std(axis=0))[:k]].tolist()))
    if chosen.tolist() != sorted(panel.tolist()):
        sys.exit('the panel selected on this machine is not the panel of the archived collection; the archive was not written')
    scores, k = fold_scores()
    out = dict(shape=np.array(X.shape), cols=cols.astype(np.int32), order=order[:, cols].astype(np.int16), panel=chosen.astype(np.int32))
    for i, s in enumerate(scores):
        out['inductive_fold_%d' % i] = np.array(sorted(EXPR[np.argsort(-s)[:k]].tolist()), dtype=np.int32)
    np.savez_compressed(ARCHIVE, **out)
    print('wrote %s: the order of %d reactions with ties, the probe panel and %d fold panels; %d stored percentiles agree (%d on tied values)'
          % (os.path.relpath(ARCHIVE, PATHS.ROOT), len(cols), len(scores), n, tied))


def check():
    pct = CO.percentile_as_collected(X)                     # raises if the archive does not order X
    n, tied, bad = compare_with_records(pct)
    if bad:
        sys.exit('cohort_order.npz: %d of the %d percentiles stored at collection differ from the archived order' % (bad, n))
    keep, y, panel, k = msi_design()
    sel = CO.top_k_as_collected('panel', X[:, EXPR].std(axis=0), k, EXPR)
    if sorted(sel.tolist()) != sorted(panel.tolist()):
        sys.exit('cohort_order.npz: the archived panel is not the panel of the archived collection')
    scores, k = fold_scores()
    for i, s in enumerate(scores):
        CO.top_k_as_collected('inductive_fold_%d' % i, s, k, EXPR)
    here = np.argsort(X, axis=0)
    same = bool((here == CO.argsort_as_collected(X)).all())
    print('cohort_order.npz: the archived order sorts the expression matrix; %d percentiles stored at collection agree with it, '
          '%d of them on tied values; the probe panel and %d fold panels are valid selections' % (n, tied, len(scores)))
    print('this machine\'s own argsort %s the archived order' % ('gives' if same else 'does not give'))


if __name__ == '__main__':
    check() if '--check' in sys.argv[1:] else write()
