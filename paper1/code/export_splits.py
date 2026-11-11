#!/usr/bin/env python3
"""Write the fixed splits of the study to one file, and check them against the prediction arrays.

The patient folds, the random reaction folds and the family-disjoint reaction folds are
deterministic functions of the seed, the cohort's patient list and the label and availability
vectors. Every driver rebuilds them with the same few lines (double_holdout.py,
naive_baselines_allfolds.py, split_audit.py). This script runs those lines once more and writes
the result as data, so that a reader who only wants the folds does not have to run a driver:

  patient_folds            five folds, each with its train, validation and test barcodes
  reaction_folds_random    three folds, the held-out reaction indices of each (0-based, in the
                           reaction order of the training arrays)
  reaction_folds_family    the same for the family-disjoint split of data/families/families_union.json
  reaction_folds_random_tissue_matched_label
                           the random folds of the runs scored against the HT29-only label: those
                           runs stratify the folds on that label, so their folds differ

Inputs, all in the repository: data/clinical_metadata_msi.tsv (the cohort's barcodes and projects,
in the order of the deposited clinical_metadata.tsv), paper2/rxn_context.npz (the label vector and
the expression-availability mask, the source split_audit.py falls back on) and the family map.
With P1_DATA pointing at the deposited cohort directory, the barcodes, the labels and the
availability mask are read from it instead, exactly as naive_baselines_allfolds.py reads them, and
the file records which source was used.

With --arrays DIR (the directory the release tarballs were unpacked into, one subdirectory per
result family), every array is compared with the folds written here: its patient_ids against the
test patients of its patient fold and its heldout_mask against the held-out reactions of its
reaction fold, family-disjoint when the name says so, the tissue-matched-label folds for the
arrays of the ht29 family, and the random folds otherwise. The counts, by family, are written to
array_check.json beside the split file.

With --check nothing is written: the folds are rebuilt and compared with the committed file, and
the script stops with an error if they differ. The committed file is the reference, because it
was compared with the folds stored in every released prediction array; an environment whose
scikit-learn shuffles differently would otherwise replace it without notice. reproduce.sh runs
the check.

Output: data/splits/splits.json (or P1_OUT). The same file is deposited with the cohort in the data
record, under crc_624/splits/. Needs NumPy, pandas and scikit-learn only.
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = os.path.dirname(HERE)
ROOT = os.path.dirname(P1)
sys.path.insert(0, HERE)
from family_folds import load_family_map, family_rfolds

ap = argparse.ArgumentParser()
ap.add_argument('--arrays', help='directory of unpacked *_preds.npz arrays to check the folds against (searched recursively)')
ap.add_argument('--family_map', default=os.path.join(P1, 'data', 'families', 'families_union.json'))
ap.add_argument('--out', default=os.environ.get('P1_OUT', os.path.join(P1, 'data', 'splits', 'splits.json')))
ap.add_argument('--check', action='store_true', help='compare the rebuilt folds with the file at --out and write nothing')
A = ap.parse_args()
N = 10600; SEED = 2024

# ---- patients, labels and availability -------------------------------------------------------
D = os.environ.get('P1_DATA')
if D:
    D = os.path.expanduser(D)
    import h5py, torch
    meta = pd.read_csv(os.path.join(D, 'clinical_metadata.tsv'), sep='\t')
    y = np.asarray(torch.load(os.path.join(D, 'activity_pseudolabels.pt'), map_location='cpu', weights_only=False)).astype(int)
    source = dict(patients='P1_DATA/clinical_metadata.tsv', labels='P1_DATA/activity_pseudolabels.pt',
                  availability='P1_DATA/reaction_features, first 20 training patients of patient fold 0')
else:
    meta = pd.read_csv(os.path.join(P1, 'data', 'clinical_metadata_msi.tsv'), sep='\t')
    z = np.load(os.path.join(ROOT, 'paper2', 'rxn_context.npz'), allow_pickle=True)
    y = z['labels'].astype(int); has = z['has'].astype(bool)
    source = dict(patients='data/clinical_metadata_msi.tsv', labels='paper2/rxn_context.npz [labels]',
                  availability='paper2/rxn_context.npz [has]')
ids = meta['tcga_barcode'].tolist(); proj = meta['project'].fillna('U').tolist()

# ---- folds, identical to double_holdout.py ---------------------------------------------------
pk = StratifiedKFold(5, shuffle=True, random_state=SEED); pfolds = []
for tr, te in pk.split(ids, proj):
    tr_ids = [ids[i] for i in tr]
    trn, val = train_test_split(tr_ids, test_size=0.15, random_state=SEED)
    pfolds.append(dict(train=trn, val=val, test=[ids[i] for i in te]))
if D:
    has = np.stack([h5py.File(os.path.join(D, 'reaction_features', p + '.h5'), 'r')['X_R'][:, 0]
                    for p in pfolds[0]['train'][:20]]).max(0) > 0
assert len(y) == N and len(has) == N
rk = StratifiedKFold(3, shuffle=True, random_state=SEED)
r_random = [np.asarray(te) for _, te in rk.split(np.zeros(N), 2 * y + has.astype(int))]
comp, scheme = load_family_map(A.family_map)
r_family = [np.asarray(te) for _, te in family_rfolds(comp, y, has, n_folds=3, seed=SEED)]

def _tensor_1d(path):
    """A one-dimensional float tensor saved by PyTorch, read without PyTorch: the file is a zip
    archive whose one storage holds the values as little-endian float32."""
    import zipfile
    z = zipfile.ZipFile(path)
    return np.frombuffer(z.read([n for n in z.namelist() if n.endswith('/data/0')][0]), dtype='<f4')
# the tissue-matched (HT29-only) label of the robustness runs; their folds are stratified on it
y_ht = _tensor_1d(os.path.join(P1, 'data', 'activity_labels_ht29.pt'))
_n_ht = json.load(open(os.path.join(P1, 'data', 'labels_ht29.json')))['models']['HT29']['n_active_matched']
assert len(y_ht) == N and set(np.unique(y_ht).tolist()) <= {0.0, 1.0} and int(y_ht.sum()) == _n_ht, 'unexpected tissue-matched label file'
y_ht = y_ht.astype(int)
r_random_ht = [np.asarray(te) for _, te in rk.split(np.zeros(N), 2 * y_ht + has.astype(int))]

def _partition(folds, n):
    seen = np.zeros(n, dtype=int)
    for f in folds: seen[f] += 1
    return bool((seen == 1).all())
_test = [p for f in pfolds for p in f['test']]
out = dict(
    _generated_by='code/export_splits.py', seed=SEED, n_patients=len(ids), n_reactions=N, source=source,
    family_map=os.path.relpath(A.family_map, P1), family_scheme=scheme,
    checks=dict(n_active=int(y.sum()), n_expression_bearing=int(has.sum()),
                patient_test_folds_partition_the_cohort=bool(sorted(_test) == sorted(ids)),
                patient_fold_sizes=[dict(train=len(f['train']), val=len(f['val']), test=len(f['test'])) for f in pfolds],
                random_folds_partition_the_reactions=_partition(r_random, N),
                family_folds_partition_the_reactions=_partition(r_family, N),
                random_heldout_sizes=[int(len(f)) for f in r_random],
                family_heldout_sizes=[int(len(f)) for f in r_family],
                tissue_matched_label_n_active=int(y_ht.sum()),
                tissue_matched_label_folds_partition_the_reactions=_partition(r_random_ht, N),
                tissue_matched_label_heldout_sizes=[int(len(f)) for f in r_random_ht],
                families_split_across_family_folds=int(sum(len({k for k, f in enumerate(r_family) if c in set(comp[f].tolist())}) > 1
                                                           for c in np.unique(comp)))),
    patient_folds=pfolds,
    reaction_folds_random=[f.tolist() for f in r_random],
    reaction_folds_family=[f.tolist() for f in r_family],
    reaction_folds_random_tissue_matched_label=[f.tolist() for f in r_random_ht])

if A.check:
    if not os.path.exists(A.out):
        sys.exit(f'--check: {A.out} does not exist')
    ref = json.load(open(A.out))
    fields = ('seed', 'n_patients', 'n_reactions', 'patient_folds', 'reaction_folds_random', 'reaction_folds_family',
              'reaction_folds_random_tissue_matched_label')
    differ = [k for k in fields if ref.get(k) != out[k]]
    if differ:
        sys.exit(f'the folds rebuilt in this environment differ from {A.out} in: {", ".join(differ)}. '
                 'The committed file is the reference (it agrees with the folds stored in the released '
                 'prediction arrays); check the scikit-learn and NumPy versions against requirements.txt')
    print(f'the folds rebuilt in this environment equal those of {A.out}: {len(pfolds)} patient folds, '
          f'{len(r_random)} random and {len(r_family)} family reaction folds')
    sys.exit(0)

os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)

# ---- the same folds as the prediction arrays recorded them ------------------------------------
if A.arrays:
    base = os.path.expanduser(A.arrays)
    files = sorted(glob.glob(os.path.join(base, '**', '*_preds.npz'), recursive=True))
    by_family = {}; mismatched = []; skipped = []
    for f in files:
        name = os.path.basename(f); fam = os.path.basename(os.path.dirname(f))
        m = re.search(r'_p(\d)_r(\d)_', name)
        z = np.load(f, allow_pickle=True)
        if not m or 'heldout_mask' not in z.files or 'patient_ids' not in z.files:
            skipped.append(os.path.join(fam, name)); continue
        pf, rf = int(m.group(1)), int(m.group(2))
        folds = r_family if '_fam_' in name else (r_random_ht if fam == 'ht29' else r_random)
        want = np.zeros(N, dtype=bool); want[folds[rf]] = True
        pat_ok = [str(p) for p in z['patient_ids']] == pfolds[pf]['test']
        rxn_ok = bool((np.asarray(z['heldout_mask']).astype(bool) == want).all())
        c = by_family.setdefault(fam, dict(arrays=0, test_patients_match=0, heldout_reactions_match=0))
        c['arrays'] += 1; c['test_patients_match'] += int(pat_ok); c['heldout_reactions_match'] += int(rxn_ok)
        if not (pat_ok and rxn_ok): mismatched.append(os.path.join(fam, name))
    tot = {k: sum(c[k] for c in by_family.values()) for k in ('arrays', 'test_patients_match', 'heldout_reactions_match')}
    check = dict(_generated_by='code/export_splits.py --arrays', arrays_found=len(files), arrays_checked=tot['arrays'],
                 test_patients_match=tot['test_patients_match'], heldout_reactions_match=tot['heldout_reactions_match'],
                 by_family=dict(sorted(by_family.items())), mismatched=mismatched, not_checked=skipped,
                 note='family-disjoint folds for names carrying _fam_, tissue-matched-label folds for the ht29 family, random folds otherwise')
    json.dump(check, open(os.path.join(os.path.dirname(os.path.abspath(A.out)), 'array_check.json'), 'w'), indent=1)
    print(f"arrays: {len(files)} found, {tot['arrays']} checked, test patients match in {tot['test_patients_match']}, "
          f"held-out reactions match in {tot['heldout_reactions_match']}, {len(skipped)} not checked")
    if mismatched:
        print('MISMATCHED:', ', '.join(mismatched[:20]))

json.dump(out, open(A.out, 'w'), indent=1)
print(f'wrote {A.out}: {len(pfolds)} patient folds, {len(r_random)} random and {len(r_family)} family reaction folds '
      f'(held-out sizes {out["checks"]["random_heldout_sizes"]} and {out["checks"]["family_heldout_sizes"]})')
