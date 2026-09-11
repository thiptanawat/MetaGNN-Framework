#!/usr/bin/env python3
"""Full-scale double hold-out evaluation (patients x reactions) on the rebuilt TCGA-CRC 624 features.
Designed for one GPU (RTX 5090 / H100). Restart-safe: every (model, lambda, pfold, rfold) cell writes its own
result file and is skipped when present.

Usage (from the data record v2 root, i.e. the directory containing crc_624/):
  python double_holdout.py --data crc_624 --out results_double_holdout --device cuda \
      --models gnn_B:0.0 gnn_B:0.2 gnn_A:0.0 mlp:0.0 mlp_emb:0.0 --pfolds 5 --rfolds 3
Each cell: train on train-patients x train-reactions (BCE on train reactions only + lambda * mass-balance),
select on val-patients x train-reactions AUROC (patience 15, max 80 epochs), evaluate on test-patients x
held-out reactions with MC-Dropout T=30, save per-patient predictions (scores, sigma) and the diagnostics.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trainer as T
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', default='results_double_holdout'); ap.add_argument('--device', default='cuda')
ap.add_argument('--models', nargs='+', default=['gnn_B:0.0', 'gnn_B:0.2', 'gnn_A:0.0', 'mlp:0.0', 'mlp_emb:0.0'])
ap.add_argument('--pfolds', type=int, default=5); ap.add_argument('--rfolds', type=int, default=3); ap.add_argument('--epochs', type=int, default=80)
ap.add_argument('--patience', type=int, default=15); ap.add_argument('--T', type=int, default=30); ap.add_argument('--seed', type=int, default=2024)
ap.add_argument('--n_patients', type=int, default=0, help='0 = all patients; smaller for smoke tests')
ap.add_argument('--preset', choices=['core', 'full'], default=None, help='core = gnn_B:0.0 mlp:0.0 mlp_emb:0.0 (answers the main question); full = adds gnn_B:0.2 gnn_A:0.0')
ap.add_argument('--only_pfold', type=int, default=-1,
                help='restrict to a single patient fold, so a control can be compared against its '
                     'counterpart in the main grid without sweeping folds that add nothing to the answer')
ap.add_argument('--only_rfold', type=int, default=-1,
                help='restrict to a single reaction fold so independent cells can run as parallel '
                     'processes without racing to claim the same cell')
ap.add_argument('--feature_mode', choices=['real','mean','zero','indicator','permute'], default='real',
                help='real = unchanged. mean = every patient is shown the TRAIN-patient cohort mean '
                     'expression, which removes patient specificity but keeps reaction-level expression. '
                     'zero = no expression at all, leaving topology and metabolite features. '
                     'indicator = the presence pattern of the train-cohort mean (1 where nonzero), '
                     'the same binary column for every patient. permute = every patient is shown another '
                     'patient\'s real column (a seeded derangement within each split), the wrong-patient arm.')
ap.add_argument('--inner_val_rxn', type=float, default=0.0,
                help='hold out this fraction of the TRAINING reactions (stratified on label and presence, seeded) '
                     'as an inner validation set: the model is fitted on the remaining training reactions and '
                     'early stopping reads the validation patients on the inner held-out reactions, so the '
                     'stopping criterion matches the test question (unseen reactions) instead of rewarding '
                     'memorization of training reactions. Output tag suffix _ivr. 0 = the main protocol.')
ap.add_argument('--synth_target', choices=['none', 'cohortmedian', 'ownrank'], default='none',
                help='positive control for the audit: replace the shared label of a seeded fraction '
                     '(--synth_frac) of the expression-bearing reactions by a PATIENT-VARYING label built from '
                     'the column the model receives. cohortmedian = 1 where the patient\'s within-patient '
                     'percentile at the reaction lies above the cohort median of that percentile at the same '
                     'reaction, so every varying reaction is positive for about half the cohort and a column '
                     'that is the same for every patient carries no information about it. ownrank = 1 where '
                     'the reaction lies above the median of the patient\'s own expression-bearing reactions '
                     '(within-patient percentile > 0.5), a label that is largely predictable from the cohort '
                     'mean because percentiles are similar across patients. The other reactions keep the shared '
                     'label. Each patient then has its own label vector; it enters the loss for training '
                     'patients, the stopping criterion for validation patients and the evaluation for test '
                     'patients. The reaction folds stay those of the main grid. Output tag suffix '
                     '_synth<frac> (cohortmedian) or _synthown<frac> (ownrank). none = the main protocol.')
ap.add_argument('--synth_frac', type=float, default=1.0,
                help='fraction of the expression-bearing reactions whose label is made patient-varying '
                     'under --synth_target (drawn once, seeded from --seed); 1.0 = all of them')
ap.add_argument('--train_seed', type=int, default=-1,
                help='seed for parameter initialization and dropout only; the folds, the train/val split and '
                     'the reaction folds stay on --seed, so a cell trained under a second --train_seed is the '
                     'same cell with a different training realization. -1 = use --seed. The output tag gets '
                     'the suffix _ts<train_seed>.')
ap.add_argument('--expr_transform', choices=['none', 'rank'], default='none',
                help='rank = replace every patient\'s expression column by its within-patient percentile among the '
                     'expression-bearing reactions (average ranks for ties; zeros stay zero), the second normalization '
                     'of the robustness check; the cohort-mean arm then averages the transformed columns. The output '
                     'tag gets the suffix _rank.')
ap.add_argument('--family_map', default=None,
                help='path to a families_*.json from build_families.py: the reaction folds and the inner-validation '
                     'partition are then drawn family by family (family_folds.py), so that no held-out reaction has a '
                     'stoichiometric or gene-rule relative among the fitted or selected-on reactions; the cell tag gains _<family_tag>')
ap.add_argument('--family_tag', default='fam', help='tag suffix that names the family scheme (fam = the union map; famS2, famS3 for the alternatives)')
ap.add_argument('--preflight', action='store_true', help='check environment + data, time a few steps, print an ETA, then exit')
A = ap.parse_args()
if A.preset == 'core': A.models = ['gnn_B:0.0', 'mlp:0.0', 'mlp_emb:0.0']
elif A.preset == 'full': A.models = ['gnn_B:0.0', 'gnn_B:0.2', 'gnn_A:0.0', 'mlp:0.0', 'mlp_emb:0.0']
os.makedirs(A.out, exist_ok=True); dev = torch.device(A.device if torch.cuda.is_available() or A.device == 'cpu' else 'cpu')
N_RXN = 10600; FEAT = os.path.join(A.data, 'reaction_features')
ARCH = {'A': dict(hidden=128, layers=2, heads=4), 'B': dict(hidden=256, layers=3, heads=8)}

class ReactionMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2, emb=0):
        super().__init__(); self.emb = nn.Embedding(N_RXN, emb) if emb else None
        d = in_dim + emb; self.net = nn.Sequential(nn.Linear(d, hidden), nn.ELU(), T.MCDropout(dropout), nn.Linear(hidden, hidden // 2), nn.ELU(), T.MCDropout(dropout), nn.Linear(hidden // 2, 1), nn.Sigmoid())
    def forward(self, x_dict, edge_index_dict=None):
        x = x_dict['reaction']
        if self.emb is not None: x = torch.cat([x, self.emb.weight], 1)
        return self.net(x).squeeze(-1)
FEAT_OVERRIDE = None   # a 1-D tensor over reactions, or None to leave the input untouched
# permute arm: every patient is shown another patient's real expression column. The donor map is a
# derangement drawn once per split (train, val, test separately, so no held-out patient's column enters
# training), from the run seed; PER_PATIENT holds the donor column for each dataset position.
PER_PATIENT = None     # list of 1-D tensors aligned with the dataset currently being iterated, or None
def derangement(n, rng):
    perm = np.arange(n)
    if n < 2: return perm
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)): return perm
def donor_columns(ids, rng):
    import h5py as _h5
    cols = []
    for _p in ids:
        with _h5.File(os.path.join(FEAT, _p + '.h5')) as _h: cols.append(np.asarray(_h['X_R'][:, 0], dtype=np.float32))
    perm = derangement(len(ids), rng)
    return [torch.tensor(cols[perm[i]]) for i in range(len(ids))], [ids[perm[i]] for i in range(len(ids))]
def rank_col(v):
    """Within-patient percentile of the nonzero entries (average ranks for ties), zeros unchanged."""
    nz = v > 0; n = int(nz.sum())
    if n == 0: return v
    vals = v[nz]
    uniq, inv, counts = torch.unique(vals, return_inverse=True, return_counts=True)
    last = torch.cumsum(counts, 0).to(torch.float32); first = last - counts.to(torch.float32) + 1
    out = v.clone(); out[nz] = ((first + last) / 2)[inv] / n
    return out
def rank_col_np(v):
    from scipy.stats import rankdata
    v = v.copy(); nz = v > 0
    if nz.any(): v[nz] = rankdata(v[nz]) / nz.sum()
    return v
def fwd(model, b, ov=None):
    xr = b['reaction'].x
    if ov is not None:
        xr = xr.clone(); xr[:, 0] = ov.to(xr.device)
    if A.expr_transform == 'rank':
        xr = xr.clone(); xr[:, 0] = rank_col(xr[:, 0])
    if FEAT_OVERRIDE is not None:
        xr = xr.clone(); xr[:, 0] = FEAT_OVERRIDE.to(xr.device)
    return model(x_dict={'reaction': xr, 'metabolite': b['metabolite'].x}, edge_index_dict={r: b[r].edge_index for r in b.edge_types})
def ds(ids): torch.manual_seed(A.seed); return T.UnifiedDataset(A.data, ids, 'cohort_220', bipartite_only=False, feature_override_dir=FEAT)
def build(name, sample):
    kind = name.split('_')[0]
    if kind == 'gnn':
        a = ARCH[name.split('_')[1]]
        return T.MetaGNN(rxn_in_dim=sample['reaction'].x.shape[1], met_in_dim=sample['metabolite'].x.shape[1], hidden_dim=a['hidden'], n_layers=a['layers'], heads=a['heads'], dropout=0.2, edge_types=list(sample.edge_types))
    return ReactionMLP(sample['reaction'].x.shape[1], emb=(64 if name == 'mlp_emb' else 0))
def auc_masked(model, dset, mask, ovs=None, ys_own=None):
    """AUROC over the patients of dset on the masked reactions; ys_own gives each patient its own
    label vector (the synthetic patient-varying target), otherwise the shared label in the data."""
    model.eval(); ps = []; ys = []
    with torch.no_grad():
        for i in range(len(dset)): b = dset[i].to(dev); ps.append(fwd(model, b, None if ovs is None else ovs[i]).cpu().numpy()[mask]); ys.append((b['reaction'].y.cpu().numpy() if ys_own is None else ys_own[i])[mask])
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))

meta = pd.read_csv(os.path.join(A.data, 'clinical_metadata.tsv'), sep='\t'); ids = meta['tcga_barcode'].tolist(); proj = meta['project'].fillna('U').tolist()
if A.n_patients: ids, _, proj, _ = train_test_split(ids, proj, train_size=A.n_patients, random_state=A.seed, stratify=proj)
pk = StratifiedKFold(A.pfolds, shuffle=True, random_state=A.seed); pfolds = []
for tr, te in pk.split(ids, proj):
    tr_ids = [ids[i] for i in tr]; trn, val = train_test_split(tr_ids, test_size=0.15, random_state=A.seed); pfolds.append(dict(train=trn, val=val, test=[ids[i] for i in te]))
T_START = time.time()
import h5py
with h5py.File(os.path.join(A.data, 'recon3d_stoich.h5')) as h: S = torch.tensor(h['S'][:], dtype=torch.float32).to(dev)
y_all = np.asarray(torch.load(os.path.join(A.data, 'activity_pseudolabels.pt'), map_location='cpu', weights_only=False)).astype(int)
probe = ds(pfolds[0]['train'][:20]); has = np.stack([probe[i]['reaction'].x[:, 0].numpy() for i in range(len(probe))]).max(0) > 0
rk = StratifiedKFold(A.rfolds, shuffle=True, random_state=A.seed); rfolds = list(rk.split(np.zeros(N_RXN), 2 * y_all + has.astype(int)))
FAMILY = None
if A.family_map:
    import family_folds as FF
    FAMILY, _scheme = FF.load_family_map(A.family_map)
    rfolds = FF.family_rfolds(FAMILY, y_all, has, n_folds=A.rfolds, seed=A.seed)
    for _r in FF.fold_report(FAMILY, rfolds, y_all, has):
        print(f"family fold {_r['fold']} ({_scheme}): {_r['n_test']} held-out reactions in {_r['n_families_test']} families, prevalence {_r['prevalence_test']:.4f}, "
              f"expression-bearing {_r['exprbearing_test']:.4f}, held-out reactions with a family relative in training: {_r['heldout_with_family_relative_in_train']}", flush=True)

# Synthetic patient-varying target (positive control for the audit). SYNTH_Y maps a patient id to
# its own label vector; SYNTH_VARY marks the reactions whose label varies across patients. The
# shared label y_all is kept for the reaction folds, so the held-out reactions are the main grid's.
SYNTH_Y, SYNTH_VARY = None, None
if A.synth_target != 'none':
    _rng = np.random.default_rng(A.seed + 900)
    _cand = np.flatnonzero(has); _k = int(round(A.synth_frac * len(_cand)))
    SYNTH_VARY = np.zeros(N_RXN, dtype=bool); SYNTH_VARY[np.sort(_rng.choice(_cand, size=_k, replace=False))] = True
    import h5py as _h5
    _R = np.zeros((len(ids), N_RXN), dtype=np.float32)     # within-patient percentiles, every patient
    for _k, _p in enumerate(ids):
        with _h5.File(os.path.join(FEAT, _p + '.h5')) as _h: _R[_k] = rank_col_np(np.asarray(_h['X_R'][:, 0], dtype=np.float32))
    if A.synth_target == 'cohortmedian':
        _thr = np.median(_R, axis=0)                          # the cohort median percentile at each reaction
    else:
        _thr = np.full(N_RXN, 0.5, dtype=np.float32)          # the patient's own median
    SYNTH_Y = {}
    for _k, _p in enumerate(ids):
        _y = y_all.copy(); _y[SYNTH_VARY] = (_R[_k, SYNTH_VARY] > _thr[SYNTH_VARY]).astype(int); SYNTH_Y[_p] = _y
    _Yall = np.stack([SYNTH_Y[_p] for _p in ids])
    print(f'synth_target={A.synth_target}: {int(SYNTH_VARY.sum())} of {int(has.sum())} expression-bearing reactions carry a '
          f'patient-varying label (frac {A.synth_frac}); mean positive rate on them {_Yall[:, SYNTH_VARY].mean():.4f}; '
          f'the other {N_RXN - int(SYNTH_VARY.sum())} reactions keep the shared label', flush=True)
    del _R
def own_labels(id_list):
    """Per-patient label vectors for a split, or None when the shared label applies."""
    return None if SYNTH_Y is None else [SYNTH_Y[_p] for _p in id_list]

if A.preflight:
    print(f'device            : {dev} ({torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"})')
    print(f'patients          : {len(ids)} -> per fold ~{len(pfolds[0]["train"])} train / {len(pfolds[0]["val"])} val / {len(pfolds[0]["test"])} test')
    print(f'reactions         : {N_RXN}, {A.rfolds} folds -> ~{len(rfolds[0][1])} held out per fold')
    print(f'labels active     : {y_all.mean():.4f}; expression-bearing reactions: {int(has.sum())}')
    bench = ds(pfolds[0]['train'][:6]); b0 = bench[0].to(dev); bce0 = nn.BCELoss()
    for spec in A.models:
        name, lam = spec.split(':'); lam = float(lam)
        m = build(name, bench[0]).to(dev); o = torch.optim.AdamW(m.parameters(), lr=1e-3); ts = []
        for k in range(6):
            if dev.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time(); o.zero_grad(); s_ = fwd(m, b0); l = bce0(s_, b0['reaction'].y.float())
            if lam > 0: l = l + lam * (S @ s_).pow(2).mean()
            l.backward(); o.step()
            if dev.type == 'cuda': torch.cuda.synchronize()
            ts.append(time.time() - t0)
        step = float(np.median(ts[2:])); ntr = len(pfolds[0]['train']); nva = len(pfolds[0]['val']); nte = len(pfolds[0]['test'])
        ep_assumed = min(A.epochs, 40)
        per_cell = ep_assumed * (ntr * step + nva * step / 3) + nte * A.T * step / 3
        cells = A.pfolds * A.rfolds
        print(f'  {name:8s} lam={lam}: {sum(p.numel() for p in m.parameters()):>9,} params | {step*1000:6.1f} ms/patient-step | ~{per_cell/60:5.1f} min/cell x {cells} cells = {per_cell*cells/3600:5.2f} h')
        del m, o
    print('\nTotal above assumes ~40 epochs before early stopping; restart-safe, so you can stop and resume anytime.')
    sys.exit(0)

for spec in A.models:
    name, lam = spec.split(':'); lam = float(lam)
    for pf, sp in enumerate(pfolds):
        if A.only_pfold >= 0 and pf != A.only_pfold: continue
        tr_ds = ds(sp['train']); va_ds = ds(sp['val']); te_ds = ds(sp['test'])
        if A.feature_mode == 'zero':
            FEAT_OVERRIDE = torch.zeros(N_RXN)
        elif A.feature_mode in ('mean', 'indicator'):
            import h5py as _h5
            acc = np.zeros(N_RXN)
            for _p in sp['train']:
                with _h5.File(os.path.join(FEAT, _p + '.h5')) as _h:
                    _col = _h['X_R'][:, 0]
                    acc += rank_col_np(_col) if A.expr_transform == 'rank' else _col
            FEAT_OVERRIDE = torch.tensor(acc / len(sp['train']), dtype=torch.float32)
            if A.feature_mode == 'indicator':
                # presence only: 1 where the train-cohort mean is nonzero, 0 elsewhere. Every patient
                # sees the same binary column, so this arm asks how much of the cohort-mean arm's gain
                # is the presence pattern and how much is the magnitudes.
                FEAT_OVERRIDE = (FEAT_OVERRIDE > 0).float()
        else:
            FEAT_OVERRIDE = None
        if FEAT_OVERRIDE is not None:
            print(f'feature_mode={A.feature_mode}: expression column replaced, '
                  f'{int((FEAT_OVERRIDE != 0).sum())} nonzero of {N_RXN}', flush=True)
        OV_TR = OV_VA = OV_TE = None; donors = None
        if A.feature_mode == 'permute':
            _rng = np.random.default_rng(A.seed + 7919 * (pf + 1))
            OV_TR, d_tr = donor_columns(sp['train'], _rng); OV_VA, d_va = donor_columns(sp['val'], _rng); OV_TE, d_te = donor_columns(sp['test'], _rng)
            donors = dict(train=d_tr, val=d_va, test=d_te)
            print(f'feature_mode=permute: every patient shown another patient\'s column (derangement within each split, seed {A.seed + 7919 * (pf + 1)})', flush=True)
        for rf, (rtr, rte) in enumerate(rfolds):
            if A.only_rfold >= 0 and rf != A.only_rfold: continue
            tag = f'{name}_mb{lam}_p{pf}_r{rf}' + ('' if A.feature_mode=='real' else f'_{A.feature_mode}') + ('' if A.expr_transform == 'none' else f'_{A.expr_transform}') + ('' if A.train_seed < 0 else f'_ts{A.train_seed}') + ('' if A.synth_target == 'none' else (f'_synth{A.synth_frac:g}' if A.synth_target == 'cohortmedian' else f'_synthown{A.synth_frac:g}')) + ('' if A.inner_val_rxn <= 0 else '_ivr') + ('' if FAMILY is None else f'_{A.family_tag}'); rfile = os.path.join(A.out, tag + '.json')
            if os.path.exists(rfile): continue
            mtr = torch.zeros(N_RXN, dtype=torch.bool); mtr[rtr] = True; mte = ~mtr
            # inner validation reactions: a stratified, seeded subset of the training reactions that the
            # model never fits and that early stopping is read on; the outer held-out reactions stay untouched
            mfit, mval = mtr.clone(), torch.zeros(N_RXN, dtype=torch.bool)
            if A.inner_val_rxn > 0 and FAMILY is not None:
                _m = FF.family_inner_val(FAMILY, rtr, y_all, has, frac=A.inner_val_rxn, seed=A.seed + 500 + pf * 10 + rf)
                mval = torch.tensor(_m); mfit = mtr & ~mval
                print(f'inner validation reactions (whole families): {int(mval.sum())} of {int(mtr.sum())} training reactions held out for early stopping', flush=True)
            elif A.inner_val_rxn > 0:
                _rng = np.random.default_rng(A.seed + 500 + pf * 10 + rf)
                _strat = 2 * y_all[rtr] + has[rtr].astype(int)
                _val = []
                for _s in np.unique(_strat):
                    _idx = rtr[_strat == _s]; _k = int(round(A.inner_val_rxn * len(_idx)))
                    _val.extend(_rng.choice(_idx, size=_k, replace=False).tolist())
                mval[_val] = True; mfit[_val] = False
                print(f'inner validation reactions: {int(mval.sum())} of {int(mtr.sum())} training reactions held out for early stopping', flush=True)
            mtr_d = mfit.to(dev)
            Y_TR, Y_VA, Y_TE = own_labels(sp['train']), own_labels(sp['val']), own_labels(sp['test'])
            Y_TR_d = None if Y_TR is None else [torch.tensor(_y, dtype=torch.float32, device=dev) for _y in Y_TR]
            torch.manual_seed((A.seed if A.train_seed < 0 else A.train_seed) + pf * 100 + rf); model = build(name, tr_ds[0]).to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5); bce = nn.BCELoss()
            best, best_state, pat, t0 = -1, None, 0, time.time()
            for ep in range(1, A.epochs + 1):
                model.train(); g = torch.Generator().manual_seed(A.seed + pf * 1000 + rf * 100 + ep)
                for i in torch.randperm(len(tr_ds), generator=g).tolist():
                    b = tr_ds[i].to(dev); opt.zero_grad(); s = fwd(model, b, None if OV_TR is None else OV_TR[i]); loss = bce(s[mtr_d], (b['reaction'].y.float() if Y_TR_d is None else Y_TR_d[i])[mtr_d])
                    if lam > 0: loss = loss + lam * (S @ s).pow(2).mean()
                    loss.backward(); opt.step()
                v = auc_masked(model, va_ds, (mval if A.inner_val_rxn > 0 else mtr).numpy(), OV_VA, Y_VA)
                if v > best + 1e-5: best, pat, best_state = v, 0, {k: x.detach().clone() for k, x in model.state_dict().items()}
                else: pat += 1
                if pat >= A.patience: break
            model.load_state_dict(best_state); model.train()  # dropout active for MC
            P, U = [], []
            with torch.no_grad():
                for i in range(len(te_ds)):
                    b = te_ds[i].to(dev); o = torch.stack([fwd(model, b, None if OV_TE is None else OV_TE[i]) for _ in range(A.T)]); P.append(o.mean(0).cpu().numpy()); U.append(o.std(0).cpu().numpy())
            P = np.stack(P); U = np.stack(U); mt = mte.numpy()
            # Y is the label matrix the test patients are scored against: one shared row under the
            # main protocol, each patient's own row under the synthetic patient-varying target.
            Y = np.tile(y_all, (len(P), 1)) if Y_TE is None else np.stack(Y_TE)
            Xte = np.stack([te_ds[i]['reaction'].x[:, 0].numpy() for i in range(len(te_ds))]); ind = (Xte.max(0) > 0).astype(float)
            def _mean_auc(mask, S_=None):
                S_ = P if S_ is None else S_
                return round(float(np.mean([roc_auc_score(Y[i, mask], S_[i, mask]) for i in range(len(P))])), 4)
            res = dict(model=name, lambda_mb=lam, pfold=pf, rfold=rf, n_train=len(sp['train']), n_test=len(sp['test']), n_heldout_rxn=int(mt.sum()), epochs=ep, best_val=round(best, 4), minutes=round((time.time() - t0) / 60, 1),
                       heldout_AUROC_perpatient=_mean_auc(mt),
                       heldout_AUROC_pooled=round(roc_auc_score(Y[:, mt].reshape(-1), P[:, mt].reshape(-1)), 4),
                       heldout_exprbearing_AUROC=_mean_auc(mt & (ind > 0)),
                       trainrxn_AUROC_perpatient=_mean_auc(~mt),
                       baseline_rawexpr=_mean_auc(mt, Xte),
                       baseline_indicator=round(float(np.mean([roc_auc_score(Y[i, mt], ind[mt]) for i in range(len(P))])), 4),
                       rho=round(float(P.std(0).mean() / (U.mean() / np.sqrt(A.T))), 4), interpatient_r_median=round(float(np.median(np.corrcoef(P)[np.triu_indices(len(P), 1)])), 4),
                       feature_mode=A.feature_mode, expr_transform=A.expr_transform, train_seed=(A.seed if A.train_seed < 0 else A.train_seed), data=os.path.basename(os.path.normpath(A.data)), n_active_labels=int(y_all.sum()),
                       inner_val_rxn=A.inner_val_rxn, n_inner_val_rxn=int(mval.sum()), n_fit_rxn=int(mfit.sum()))
            if SYNTH_VARY is not None:
                # the positive control's own diagnostics: the held-out reactions whose label varies
                # across patients, the held-out reactions that keep the shared label, and the two
                # fit-free rankings (each patient's own column; the train-cohort mean column)
                import h5py as _h5
                _acc = np.zeros(N_RXN)
                for _p in sp['train']:
                    with _h5.File(os.path.join(FEAT, _p + '.h5')) as _h: _acc += _h['X_R'][:, 0]
                _meancol = np.tile(_acc / len(sp['train']), (len(P), 1))
                mv, ms = mt & SYNTH_VARY, mt & ~SYNTH_VARY
                res.update(synth_target=A.synth_target, synth_frac=A.synth_frac, n_vary_rxn=int(SYNTH_VARY.sum()), n_heldout_vary_rxn=int(mv.sum()),
                           heldout_vary_AUROC=_mean_auc(mv), heldout_shared_AUROC=_mean_auc(ms) if ms.sum() > 0 else None,
                           baseline_rawexpr_vary=_mean_auc(mv, Xte), baseline_cohortmean=_mean_auc(mt, _meancol), baseline_cohortmean_vary=_mean_auc(mv, _meancol),
                           heldout_positive_rate=round(float(Y[:, mt].mean()), 4), heldout_vary_positive_rate=round(float(Y[:, mv].mean()), 4))
            if donors is not None: res['donors_test'] = donors['test']
            if FAMILY is not None:
                res.update(family_map=os.path.basename(A.family_map), family_scheme=_scheme, family_tag=A.family_tag,
                           n_heldout_families=int(len(set(FAMILY[mt].tolist()))))
            _extra = {} if SYNTH_VARY is None else dict(labels=Y.astype(np.int8), vary_mask=SYNTH_VARY)
            _extra.update(fit_mask=mfit.numpy(), inner_val_mask=mval.numpy())
            np.savez_compressed(os.path.join(A.out, tag + '_preds.npz'), patient_ids=np.array(sp['test']), scores=P.astype(np.float32), uncertainties=U.astype(np.float32), heldout_mask=mt, **_extra)
            json.dump(res, open(rfile, 'w'), indent=1); print(json.dumps(res), flush=True)
            done = len([f for f in os.listdir(A.out) if f.endswith('.json')]); total = len(A.models) * A.pfolds * A.rfolds
            el = (time.time() - T_START) / 3600
            print(f'  [{done}/{total} cells | {el:.2f} h elapsed | ETA {el/done*(total-done):.2f} h remaining]', flush=True)
print('ALL CELLS DONE')
