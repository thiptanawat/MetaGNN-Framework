#!/usr/bin/env python3
"""Loss-fit, inner-validation and outer-test scores from the saved predictions of every cell.

Every cell of the grid saves each test patient's score on all 10,600 reactions together with the
outer held-out mask. Under the matched rule (--inner_val_rxn 0.2 in double_holdout_ctrl.py) the
training reactions are further split into a loss-fit part and an inner-validation part that the
model never fits and that early stopping reads; that split is a seeded, stratified function of the
reaction fold and is rebuilt here bit for bit. The script then scores each cell on the three masks
separately, so that a "train-reaction" figure never mixes reactions the model fitted with
reactions it was only selected on, and adds per-patient AUPRC on the outer held-out reactions
for every arm, which the cell files did not record.

Under the grid's rule there is no inner partition, so the same three masks describe: reactions
the model fitted (the whole training set), the reactions that the matched rule would have withheld
(fitted here) and the outer held-out reactions. Scoring both protocols on identical masks is what
makes the two selection rules comparable on the same reduced set.

Writes out/mask_scores.json and out/masks/masks_p{pf}_r{rf}.npz (fit, inner_val, outer_test).
Run on the GPU host from ~/metagnn (the data record and the saved predictions live there); CPU only.
"""
import os, sys, re, glob, json, argparse
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.expanduser('~/metagnn/work/crc_624'))
ap.add_argument('--out', default=os.path.expanduser('~/metagnn/out'))
ap.add_argument('--dirs', nargs='+', default=['ivr', 'dh', 'ctrl', 'permute'])
ap.add_argument('--seed', type=int, default=2024); ap.add_argument('--inner_val_rxn', type=float, default=0.2)
ap.add_argument('--pfolds', type=int, default=5); ap.add_argument('--rfolds', type=int, default=3)
A = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer as T
N_RXN = 10600; FEAT = os.path.join(A.data, 'reaction_features')

# ---- the folds and the expression-availability vector, exactly as the training scripts build them
meta = pd.read_csv(os.path.join(A.data, 'clinical_metadata.tsv'), sep='\t'); ids = meta['tcga_barcode'].tolist(); proj = meta['project'].fillna('U').tolist()
pk = StratifiedKFold(A.pfolds, shuffle=True, random_state=A.seed); pfolds = []
for tr, te in pk.split(ids, proj):
    tr_ids = [ids[i] for i in tr]; trn, val = train_test_split(tr_ids, test_size=0.15, random_state=A.seed); pfolds.append(dict(train=trn, val=val, test=[ids[i] for i in te]))
y_all = np.asarray(torch.load(os.path.join(A.data, 'activity_pseudolabels.pt'), map_location='cpu', weights_only=False)).astype(int)
torch.manual_seed(A.seed)
probe = T.UnifiedDataset(A.data, pfolds[0]['train'][:20], 'cohort_220', bipartite_only=False, feature_override_dir=FEAT)
has = np.stack([probe[i]['reaction'].x[:, 0].numpy() for i in range(len(probe))]).max(0) > 0
rk = StratifiedKFold(A.rfolds, shuffle=True, random_state=A.seed); rfolds = list(rk.split(np.zeros(N_RXN), 2 * y_all + has.astype(int)))

def masks(pf, rf):
    """(fit, inner_val, outer_test) boolean masks of one cell under the matched rule; the inner
    partition is the seeded stratified draw of double_holdout_ctrl.py."""
    rtr, rte = rfolds[rf]
    mtr = np.zeros(N_RXN, dtype=bool); mtr[rtr] = True; mte = ~mtr
    mfit, mval = mtr.copy(), np.zeros(N_RXN, dtype=bool)
    rng = np.random.default_rng(A.seed + 500 + pf * 10 + rf)
    strat = 2 * y_all[rtr] + has[rtr].astype(int); val = []
    for s in np.unique(strat):
        idx = rtr[strat == s]; k = int(round(A.inner_val_rxn * len(idx)))
        val.extend(rng.choice(idx, size=k, replace=False).tolist())
    mval[val] = True; mfit[val] = False
    return mfit, mval, mte

os.makedirs(os.path.join(A.out, 'masks'), exist_ok=True)
for pf in range(A.pfolds):
    for rf in range(A.rfolds):
        mfit, mval, mte = masks(pf, rf)
        np.savez_compressed(os.path.join(A.out, 'masks', f'masks_p{pf}_r{rf}.npz'), fit=mfit, inner_val=mval, outer_test=mte,
                            n_fit=int(mfit.sum()), n_inner_val=int(mval.sum()), n_outer_test=int(mte.sum()))

PAT = re.compile(r'^(?P<model>gnn_B|gnn_A|mlp_emb|mlp)_mb(?P<lam>[0-9.]+)_p(?P<pf>\d)_r(?P<rf>\d)(?P<rest>.*)_preds\.npz$')
def mean_auc(P, Y, m):
    return float(np.mean([roc_auc_score(Y[i, m], P[i, m]) for i in range(len(P))]))
def mean_ap(P, Y, m):
    return float(np.mean([average_precision_score(Y[i, m], P[i, m]) for i in range(len(P))]))

R = dict(note='per-cell scores on the loss-fit, inner-validation and outer-test reaction masks rebuilt from the run seed; '
              'AUROC and AUPRC are per-patient means over the test patients of the cell; under the grid\'s rule the '
              'inner-validation reactions were fitted, so their score there is a fit score on a reduced set',
         seed=A.seed, inner_val_rxn=A.inner_val_rxn, cells={})
for d in A.dirs:
    for f in sorted(glob.glob(os.path.join(A.out, d, '*_preds.npz'))):
        m = PAT.match(os.path.basename(f))
        if not m: continue
        rest = m.group('rest')
        z = np.load(f, allow_pickle=True); P = z['scores'].astype(np.float64)
        if 'labels' in z.files: continue                     # synthetic-target runs carry their own label matrix; not this analysis
        pf, rf = int(m.group('pf')), int(m.group('rf'))
        mfit, mval, mte = masks(pf, rf)
        if not np.array_equal(mte, z['heldout_mask']):
            print('held-out mask mismatch, skipped:', f); continue
        Y = np.tile(y_all, (len(P), 1))
        ind = mte & has
        rec = dict(dir=d, model=m.group('model'), lam=float(m.group('lam')), pfold=pf, rfold=rf, variant=rest.lstrip('_') or 'real',
                   matched_rule=rest.endswith('_ivr'), n_patients=int(len(P)),
                   n_fit=int(mfit.sum()), n_inner_val=int(mval.sum()), n_outer_test=int(mte.sum()),
                   auroc_fit=round(mean_auc(P, Y, mfit), 4), auroc_inner_val=round(mean_auc(P, Y, mval), 4),
                   auroc_outer_test=round(mean_auc(P, Y, mte), 4), auroc_train_all=round(mean_auc(P, Y, ~mte), 4),
                   auroc_outer_exprbearing=round(mean_auc(P, Y, ind), 4),
                   auprc_fit=round(mean_ap(P, Y, mfit), 4), auprc_inner_val=round(mean_ap(P, Y, mval), 4),
                   auprc_outer_test=round(mean_ap(P, Y, mte), 4), auprc_outer_exprbearing=round(mean_ap(P, Y, ind), 4),
                   prevalence_outer_test=round(float(y_all[mte].mean()), 4), prevalence_outer_exprbearing=round(float(y_all[ind].mean()), 4))
        key = os.path.basename(f).replace('_preds.npz', '')
        R['cells'][d + '/' + key] = rec
        print(f"{d:8s} {key:44s} fit {rec['auroc_fit']:.4f}  inner {rec['auroc_inner_val']:.4f}  outer {rec['auroc_outer_test']:.4f}  "
              f"AUPRC outer {rec['auprc_outer_test']:.4f}", flush=True)
json.dump(R, open(os.path.join(A.out, 'mask_scores.json'), 'w'), indent=1)
print('wrote', os.path.join(A.out, 'mask_scores.json'), len(R['cells']), 'cells')
