"""
Dataset Validation Script
Checks integrity of all MetaGNN-CRC dataset files.

Author: Thiptanawat Phongwattana
Affiliation: School of Information Technology, KMUTT
"""
import os, sys
import h5py
import torch
import numpy as np
import pandas as pd

CHECKS_PASSED = 0
CHECKS_FAILED = 0

def check(name, condition, detail=''):
    global CHECKS_PASSED, CHECKS_FAILED
    status = '✓ PASS' if condition else '✗ FAIL'
    if not condition:
        CHECKS_FAILED += 1
        print(f"  {status}: {name}" + (f" — {detail}" if detail else ''))
    else:
        CHECKS_PASSED += 1
        print(f"  {status}: {name}")

def validate(data_root: str):
    print(f"\nValidating MetaGNN-CRC dataset at: {data_root}\n{'='*60}")

    # 1. Clinical metadata
    meta_path = os.path.join(data_root, 'clinical_metadata.tsv')
    check("clinical_metadata.tsv exists", os.path.exists(meta_path))
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path, sep='\t')
        check("n_patients = 219",  len(df) == 219,  f"got {len(df)}")
        check("required columns present",
              {'tcga_barcode','msi_status','ajcc_stage','tumour_site'}.issubset(df.columns))
        check("no duplicate barcodes", df['tcga_barcode'].nunique() == len(df))

    # 2. Edge indices
    for rel in ['substrate_of', 'produces', 'shared_metabolite']:
        ei_path = os.path.join(data_root, 'edge_indices', f'{rel}.pt')
        check(f"edge_indices/{rel}.pt exists", os.path.exists(ei_path))
        if os.path.exists(ei_path):
            ei = torch.load(ei_path)
            check(f"  {rel}: shape[0] == 2", ei.shape[0] == 2, f"got {ei.shape}")

    EXPECTED_EDGES = {'substrate_of': 29847, 'produces': 17471, 'shared_metabolite': 41980}
    for rel, n_exp in EXPECTED_EDGES.items():
        ei_path = os.path.join(data_root, 'edge_indices', f'{rel}.pt')
        if os.path.exists(ei_path):
            ei = torch.load(ei_path)
            check(f"  {rel}: n_edges = {n_exp:,}", ei.shape[1] == n_exp,
                  f"got {ei.shape[1]:,}")

    # 3. Metabolite features
    met_h5 = os.path.join(data_root, 'metabolite_features.h5')
    check("metabolite_features.h5 exists", os.path.exists(met_h5))
    if os.path.exists(met_h5):
        with h5py.File(met_h5, 'r') as f:
            X_M = f['X_M'][:]
        check("X_M shape = (4140, 519)", X_M.shape == (4140, 519), f"got {X_M.shape}")
        check("X_M finite values", np.all(np.isfinite(X_M)))

    # 4. Reaction features (sample 5 patients)
    rxn_dir = os.path.join(data_root, 'reaction_features')
    check("reaction_features/ dir exists", os.path.isdir(rxn_dir))
    if os.path.isdir(rxn_dir):
        files = [f for f in os.listdir(rxn_dir) if f.endswith('.h5')]
        check("n_patient_files = 219", len(files) == 219, f"got {len(files)}")
        for fname in sorted(files)[:5]:
            with h5py.File(os.path.join(rxn_dir, fname), 'r') as f:
                X_R = f['X_R'][:]
            check(f"  {fname}: shape (13543, 2)",
                  X_R.shape == (13543, 2), f"got {X_R.shape}")
            check(f"  {fname}: finite values", np.all(np.isfinite(X_R)))

    # 5. Activity pseudo-labels
    label_path = os.path.join(data_root, 'activity_pseudolabels.pt')
    check("activity_pseudolabels.pt exists", os.path.exists(label_path))
    if os.path.exists(label_path):
        y = torch.load(label_path)
        check("labels shape = (13543,)", y.shape == (13543,), f"got {y.shape}")
        check("binary labels {0,1}", set(y.unique().tolist()).issubset({0.0, 1.0}))
        n_active = int((y == 1).sum())
        check("active reactions in range [6000, 9000]",
              6000 <= n_active <= 9000, f"got {n_active}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed")
    if CHECKS_FAILED == 0:
        print("All checks PASSED ✓")
    else:
        print("Some checks FAILED — see details above")
    return CHECKS_FAILED == 0

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('data_root', nargs='?', default='./data',
                   help='Path to the MetaGNN-CRC dataset root directory')
    args = p.parse_args()
    ok = validate(args.data_root)
    sys.exit(0 if ok else 1)
