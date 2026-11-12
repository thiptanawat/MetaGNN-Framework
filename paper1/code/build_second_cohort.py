#!/usr/bin/env python3
"""Reaction-level expression of a second cohort, built as the first cohort's was built.

The first cohort's gene matrix is log2(1 + counts per million) of the unstranded STAR counts, the
library size being the sum over every gene of the count file
(pipelines/crc/code/00_download_and_generate_690.py, compute_tpm; the dataset is named
vst_expression for historical reasons). It is restricted to the genes that passed that cohort's
expression filter and projected to reactions through the resolved gene-protein-reaction groups: the
minimum within a group, the maximum across groups (gpr_map.py). A reaction with no resolved group
has no value and is stored as zero.

This script applies the same steps to the STAR count files of another cohort, on the same genes and
the same resolved groups, and writes a cohort directory in the layout the analysis scripts read:

    reaction_features/<barcode>.h5    X_R, 10600 x 2 (column 0 the expression, column 1 zero)
    clinical_metadata.tsv             tcga_barcode, project
    expression_qc.json                what was read, kept and dropped

The label vector, the stoichiometric matrix, the metabolite features and the edges do not depend on
any patient; they are linked from the first cohort's directory. No gene is selected on the second
cohort and no label is built from it.

    python3 build_second_cohort.py --verify --first ~/metagnn/work/crc_624 --resolved gpr_resolved_cohort1.json
    python3 build_second_cohort.py --counts DIR --first ~/metagnn/work/crc_624 \
        --resolved gpr_resolved_cohort1.json --project TCGA-BRCA --out ~/metagnn/work/brca

--verify rebuilds the first cohort's reaction features from its own gene matrix and compares them
with the stored files, which checks that the projection used here is the one that built them.
gpr_resolved_cohort1.json maps a reaction index to its resolved groups, each a list of column
indices into the first cohort's gene matrix.
"""
import argparse, glob, json, os, sys
import numpy as np, h5py

N_RXN = 10600
ap = argparse.ArgumentParser()
ap.add_argument('--first', required=True, help='directory of the first cohort (tcga_crc_rnaseq.h5, reaction_features/, shared files)')
ap.add_argument('--resolved', required=True)
ap.add_argument('--counts', help='directory of <barcode>.star_gene_counts.tsv files of the second cohort')
ap.add_argument('--project', default='TCGA-BRCA'); ap.add_argument('--out')
ap.add_argument('--verify', action='store_true')
ap.add_argument('--min_genes_detected', type=int, default=5000, help='patient filter of the first cohort build')
ap.add_argument('--min_total_reads', type=int, default=1_000_000, help='patient filter of the first cohort build')
A = ap.parse_args()
FIRST = os.path.expanduser(A.first)
RES = {int(k): v for k, v in json.load(open(os.path.expanduser(A.resolved))).items()}


def project(G):
    """G: genes x patients. Returns patients x reactions (float32), zero where no group is resolved."""
    X = np.zeros((G.shape[1], N_RXN), dtype=np.float32)
    for r, groups in RES.items():
        best = None
        for grp in groups:
            m = G[grp].min(axis=0)
            best = m if best is None else np.maximum(best, m)
        if best is not None: X[:, r] = best
    return X


with h5py.File(os.path.join(FIRST, 'tcga_crc_rnaseq.h5'), 'r') as f:
    GENES = [g.decode() for g in f['gene_ids'][:]]
    if A.verify:
        V = f['vst_expression'][:]; PID = [p.decode() for p in f['patient_ids'][:]]

if A.verify:
    X = project(V); worst = 0.0; n = 0; nz = []
    for a, pid in enumerate(PID):
        fn = os.path.join(FIRST, 'reaction_features', pid + '.h5')
        if not os.path.exists(fn): continue
        with h5py.File(fn, 'r') as g:
            XR = g['X_R'][:]
        worst = max(worst, float(np.abs(XR[:, 0] - X[a]).max())); n += 1; nz.append(int((XR[:, 0] != 0).sum()))
        assert not XR[:, 1].any()
    print(f'verify: {n} patients of the first cohort, {len(RES)} reactions with a resolved group, '
          f'largest absolute difference from the stored features {worst:.3g}, '
          f'nonzero reactions per patient {min(nz)} to {max(nz)}')
    sys.exit(0 if worst < 1e-5 else 1)

OUT = os.path.expanduser(A.out); os.makedirs(os.path.join(OUT, 'reaction_features'), exist_ok=True)
files = sorted(glob.glob(os.path.join(os.path.expanduser(A.counts), '*.star_gene_counts.tsv')))
gidx = {g: k for k, g in enumerate(GENES)}
kept, dropped, cols = [], [], []
missing_genes = None
for fn in files:
    bc = os.path.basename(fn).split('.')[0]
    ids, cnt = [], []
    with open(fn) as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            if not p[0] or p[0].startswith('#') or p[0] == 'gene_id' or p[0].startswith('N_'): continue
            ids.append(p[0]); cnt.append(int(p[3]))            # column 4 of the GDC file: unstranded counts
    cnt = np.asarray(cnt, dtype=np.float64)
    lib = cnt.sum(); detected = int((cnt > 0).sum())
    if detected < A.min_genes_detected or lib < A.min_total_reads:
        dropped.append(dict(barcode=bc, genes_detected=detected, total_reads=int(lib))); continue
    pos = {g: k for k, g in enumerate(ids)}
    miss = [g for g in GENES if g not in pos]
    if missing_genes is None: missing_genes = miss
    assert miss == missing_genes, 'the count files do not share one gene list'
    v = np.zeros(len(GENES), dtype=np.float64)
    take = np.array([pos[g] for g in GENES if g in pos]); put = np.array([gidx[g] for g in GENES if g in pos])
    v[put] = np.log2(cnt[take] / lib * 1e6 + 1.0)
    cols.append(v.astype(np.float32)); kept.append(bc)
G = np.stack(cols, axis=1)                                      # genes x patients, float32 as in the first cohort
X = project(G)
for a, bc in enumerate(kept):
    XR = np.zeros((N_RXN, 2), dtype=np.float32); XR[:, 0] = X[a]
    with h5py.File(os.path.join(OUT, 'reaction_features', bc + '.h5'), 'w') as g:
        g.create_dataset('X_R', data=XR)
with open(os.path.join(OUT, 'clinical_metadata.tsv'), 'w') as fh:
    fh.write('tcga_barcode\tproject\n')
    for bc in kept: fh.write(f'{bc}\t{A.project}\n')
for shared in ('activity_pseudolabels.pt', 'recon3d_stoich.h5', 'metabolite_features.h5', 'edge_indices'):
    dst = os.path.join(OUT, shared)
    if not os.path.lexists(dst): os.symlink(os.path.join(FIRST, shared), dst)
used = sorted({c for groups in RES.values() for grp in groups for c in grp})
qc = dict(project=A.project, count_files=len(files), patients_kept=len(kept), patients_dropped=dropped,
          genes_first_cohort=len(GENES), genes_missing_in_second=len(missing_genes or []),
          genes_used_by_resolved_groups=len(used),
          used_genes_missing_in_second=int(sum(1 for c in used if GENES[c] in set(missing_genes or []))),
          reactions_with_resolved_group=len(RES),
          nonzero_reactions_per_patient=[int((X != 0).sum(1).min()), int((X != 0).sum(1).max())],
          value_range=[float(X.min()), float(X.max())], transform='log2(1 + counts per million), unstranded STAR counts',
          patient_filter=dict(min_genes_detected=A.min_genes_detected, min_total_reads=A.min_total_reads))
json.dump(qc, open(os.path.join(OUT, 'expression_qc.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in qc.items() if k != 'patients_dropped'}, indent=1)); print('dropped', len(dropped))
