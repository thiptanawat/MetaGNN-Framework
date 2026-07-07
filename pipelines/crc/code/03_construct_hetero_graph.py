"""
Heterogeneous Bipartite Graph Construction
Builds the PyTorch Geometric HeteroData graph tensors from
Recon3D stoichiometric matrix and patient omics data.

Output files:
  edge_indices/substrate_of.pt       - M→R edges  (29,847)
  edge_indices/produces.pt           - R→M edges  (17,471)
  edge_indices/shared_metabolite.pt  - R↔R edges  (41,980)
  metabolite_features.h5             - X_M (4140 × 519)
  reaction_features/<barcode>.h5     - X_R per patient (13543 × 2)
  activity_pseudolabels.pt           - y_r binary labels

Author: Thiptanawat Phongwattana
Affiliation: School of Information Technology, KMUTT
"""

import os
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import torch
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Load Recon3D MATLAB model
# ─────────────────────────────────────────────────────────────────────────────
def load_recon3d(mat_path: str) -> dict:
    """
    Load Recon3D v3.0 from the MATLAB .mat file (COBRA format).

    Download from: https://www.vmh.life/#downloadview
    or via BiGG Database: http://bigg.ucsd.edu/models/Recon3D

    Returns dict with keys:
      S:       stoichiometric matrix (n_met × n_rxn)
      rxns:    list of reaction BiGG IDs
      mets:    list of metabolite BiGG IDs
      genes:   list of gene HGNC symbols
      grRules: list of GPR rule strings (one per reaction)
      lb/ub:   lower/upper flux bounds
    """
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        model = mat['Recon3DModel']   # or 'model' depending on MATLAB version saved
        return {
            'S':       np.array(model.S.todense()),
            'rxns':    list(model.rxns),
            'mets':    list(model.mets),
            'genes':   list(model.genes),
            'grRules': list(model.grRules),
            'lb':      np.array(model.lb),
            'ub':      np.array(model.ub),
        }
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Recon3D from {mat_path}: {e}\n"
            "Ensure the .mat file is Recon3D v3.0 in COBRA MATLAB format."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Build edge indices from stoichiometric matrix S
# ─────────────────────────────────────────────────────────────────────────────
def build_edge_indices(S: np.ndarray, output_dir: str) -> dict:
    """
    Derive three bipartite edge sets from S (n_met × n_rxn):

      substrate_of : metabolite m → reaction r  [S[m,r] < 0]
      produces     : reaction r → metabolite m  [S[m,r] > 0]
      shared_met   : reaction r1 ↔ reaction r2  [share ≥1 metabolite]

    Args:
        S:          stoichiometric matrix (scipy csr_matrix or np.ndarray)
        output_dir: where to save .pt files

    Returns:
        dict of edge_type → edge_index Tensor
    """
    os.makedirs(os.path.join(output_dir, 'edge_indices'), exist_ok=True)

    S_dense = np.array(S) if not isinstance(S, np.ndarray) else S

    # substrate_of: M → R (negative S entries)
    met_sub, rxn_sub = np.where(S_dense < 0)
    ei_sub = torch.tensor(np.stack([met_sub, rxn_sub]), dtype=torch.long)
    torch.save(ei_sub, os.path.join(output_dir, 'edge_indices', 'substrate_of.pt'))
    logger.info(f"substrate_of edges: {ei_sub.shape[1]:,}")

    # produces: R → M (positive S entries)
    met_prod, rxn_prod = np.where(S_dense > 0)
    ei_prod = torch.tensor(np.stack([rxn_prod, met_prod]), dtype=torch.long)
    torch.save(ei_prod, os.path.join(output_dir, 'edge_indices', 'produces.pt'))
    logger.info(f"produces edges: {ei_prod.shape[1]:,}")

    # shared_metabolite: R ↔ R (co-occurrence in any metabolite)
    P = (S_dense != 0).astype(np.float32)   # n_met × n_rxn participation matrix
    shared = P.T @ P                          # n_rxn × n_rxn co-occurrence matrix
    np.fill_diagonal(shared, 0)
    r1, r2 = np.where(shared > 0)
    ei_shared = torch.tensor(np.stack([r1, r2]), dtype=torch.long)
    torch.save(ei_shared, os.path.join(output_dir, 'edge_indices', 'shared_metabolite.pt'))
    logger.info(f"shared_metabolite edges: {ei_shared.shape[1]:,}")

    return {
        ('metabolite', 'substrate_of',      'reaction'): ei_sub,
        ('reaction',   'produces',          'metabolite'): ei_prod,
        ('reaction',   'shared_metabolite', 'reaction'): ei_shared,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build metabolite node features X_M  (shared across patients)
# ─────────────────────────────────────────────────────────────────────────────
def build_metabolite_features(
    met_ids: List[str],
    pubchem_props_tsv: str,
    output_h5: str,
) -> np.ndarray:
    """
    Build X_M ∈ ℝ^(4140 × 519):
      - 7 physico-chemical props from PubChem (MW, logP, HBA, HBD, TPSA, rings, charge)
      - 512-bit Morgan fingerprints (radius=2) computed via RDKit from SMILES

    Args:
        met_ids:           ordered list of Recon3D metabolite BiGG IDs (4140)
        pubchem_props_tsv: TSV mapping BiGG ID → SMILES + physico-chemical properties
                           (can be fetched via PubChem REST API using BiGG SMILES)
        output_h5:         output path for metabolite features

    Returns:
        X_M: np.ndarray (4140, 519)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem
        HAS_RDKIT = True
    except ImportError:
        logger.warning("RDKit not available — using zero-filled metabolite features")
        HAS_RDKIT = False

    props_df = pd.read_csv(pubchem_props_tsv, sep='\t', index_col='bigg_id')
    X_M = np.zeros((len(met_ids), 519), dtype=np.float32)

    for i, mid in enumerate(met_ids):
        # Strip compartment suffix  (e.g. "atp_c" → "atp")
        base_id = mid.rsplit('_', 1)[0]
        if base_id not in props_df.index:
            continue  # zero-fill for unmapped metabolites

        row = props_df.loc[base_id]
        # Physico-chemical features (7 dims)
        physico = np.array([
            row.get('mol_weight', 0.0),
            row.get('xlogp', 0.0),
            row.get('hbond_acceptor', 0.0),
            row.get('hbond_donor', 0.0),
            row.get('tpsa', 0.0),
            row.get('ring_count', 0.0),
            row.get('formal_charge', 0.0),
        ], dtype=np.float32)

        # Morgan fingerprints (512 dims)
        fp = np.zeros(512, dtype=np.float32)
        if HAS_RDKIT and pd.notna(row.get('smiles', None)):
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                fp = np.array(
                    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512),
                    dtype=np.float32,
                )

        X_M[i] = np.concatenate([physico, fp])

    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('X_M', data=X_M, compression='gzip')
        f.create_dataset('met_ids', data=np.array(met_ids, dtype='S'), compression='gzip')
        f.attrs['shape']  = str(X_M.shape)
        f.attrs['dims']   = '519: [0:7]=physico-chemical, [7:519]=Morgan_FP_r2_512bit'
    logger.info(f"Metabolite features saved: {X_M.shape} → {output_h5}")
    return X_M


# ─────────────────────────────────────────────────────────────────────────────
# Build per-patient reaction node features X_R
# ─────────────────────────────────────────────────────────────────────────────
def build_reaction_features_all_patients(
    rnaseq_h5: str,
    proteomics_h5: str,
    gpr_table_tsv: str,
    output_dir: str,
    n_reactions: int = 13543,
):
    """
    For each patient, compute X_R ∈ ℝ^(13543 × 2) using GPR mapping and
    save to individual HDF5 files under output_dir/reaction_features/.

    Column 0: GPR-mapped RNA-seq VST score
    Column 1: GPR-mapped TMT protein abundance

    GPR convention: AND→min, OR→max (Zur et al. 2010)
    """
    rxn_feat_dir = os.path.join(output_dir, 'reaction_features')
    os.makedirs(rxn_feat_dir, exist_ok=True)

    # Load expression matrices
    with h5py.File(rnaseq_h5, 'r') as f:
        rna_mat  = f['vst_expression'][:]    # genes × patients
        rna_genes = [g.decode() for g in f['gene_ids'][:]]
        rna_pats  = [p.decode() for p in f['patient_ids'][:]]

    with h5py.File(proteomics_h5, 'r') as f:
        prot_mat  = f['protein_abundance'][:]  # proteins × patients
        prot_genes = [g.decode() for g in f['gene_ids'][:]]
        prot_pats  = [p.decode() for p in f['patient_ids'][:]]

    gpr_df = pd.read_csv(gpr_table_tsv, sep='\t')  # cols: rxn_idx, gene_sets_str

    rna_df  = pd.DataFrame(rna_mat,  index=rna_genes,  columns=rna_pats)
    prot_df = pd.DataFrame(prot_mat, index=prot_genes, columns=prot_pats)

    all_patients = sorted(set(rna_pats) & set(prot_pats))
    logger.info(f"Building X_R for {len(all_patients)} patients...")

    for pid in all_patients:
        X_R = np.zeros((n_reactions, 2), dtype=np.float32)
        rna_expr  = rna_df[pid].to_dict()
        prot_expr = prot_df[pid].to_dict() if pid in prot_df.columns else {}

        for _, row in gpr_df.iterrows():
            rxn_idx   = int(row['rxn_idx'])
            gene_sets = eval(row['gene_sets_str'])  # list of lists
            if not gene_sets:
                continue

            rna_cmplx = [min(rna_expr.get(g, 0.0) for g in grp)
                         for grp in gene_sets if grp]
            prt_cmplx = [min(prot_expr.get(g, 0.0) for g in grp)
                         for grp in gene_sets if grp]

            X_R[rxn_idx, 0] = max(rna_cmplx)  if rna_cmplx  else 0.0
            X_R[rxn_idx, 1] = max(prt_cmplx)  if prt_cmplx  else 0.0

        out_path = os.path.join(rxn_feat_dir, f'{pid}.h5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('X_R', data=X_R, compression='gzip')
            f.attrs['patient_id'] = pid
            f.attrs['shape']      = str(X_R.shape)
            f.attrs['cols']       = 'col0=GPR_rnaseq_vst, col1=GPR_tmt_protein'

    logger.info(f"X_R files written for {len(all_patients)} patients → {rxn_feat_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Build activity pseudo-labels from HMA tissue-specific GEMs
# ─────────────────────────────────────────────────────────────────────────────
def build_activity_labels(
    hma_mat_path: str,
    n_reactions: int = 13543,
    output_pt: str = 'activity_pseudolabels.pt',
):
    """
    Derive binary reaction activity pseudo-labels from the 98 Human
    Metabolic Atlas (HMA) tissue-specific GEMs.

    Strategy:
      For each reaction, label as active (1) if it is active in ≥1 of the
      14 HMA GEMs most similar to colorectal tissue (cosine similarity
      of gene-expression profiles), otherwise inactive (0).

    HMA dataset:
      Download from https://metabolicatlas.org/downloads
      File: Human1_GEMs_v2.0.zip (98 tissue GEMs in JSON/MATLAB format)

    Args:
        hma_mat_path:  path to HMA MATLAB .mat file
        n_reactions:   number of reactions in Recon3D (13543)
        output_pt:     output path for binary label tensor
    """
    mat = sio.loadmat(hma_mat_path, squeeze_me=True, struct_as_record=False)
    hma_models = mat.get('models', mat.get('tissueModels', None))
    if hma_models is None:
        raise ValueError("Expected 'models' or 'tissueModels' field in HMA .mat file")

    # Aggregate: reaction active in ≥1 of the 98 HMA models → label = 1
    active_union = np.zeros(n_reactions, dtype=np.float32)
    for m in hma_models:
        rxn_active = np.array(getattr(m, 'activeRxns', np.zeros(n_reactions)), dtype=float)
        if len(rxn_active) == n_reactions:
            active_union = np.maximum(active_union, (rxn_active > 0).astype(float))

    y_r = torch.tensor(active_union, dtype=torch.float32)
    torch.save(y_r, output_pt)
    logger.info(
        f"Activity pseudo-labels: {int(y_r.sum())} active / "
        f"{n_reactions - int(y_r.sum())} inactive → {output_pt}"
    )
    return y_r


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    recon3d_mat:       str,
    rnaseq_h5:         str,
    proteomics_h5:     str,
    pubchem_props_tsv: str,
    gpr_table_tsv:     str,
    hma_mat:           str,
    output_dir:        str,
):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== Heterogeneous Graph Construction Pipeline ===")

    recon = load_recon3d(recon3d_mat)
    S     = recon['S']

    logger.info(f"Recon3D loaded: {S.shape[1]:,} reactions, {S.shape[0]:,} metabolites")

    build_edge_indices(S, output_dir)

    build_metabolite_features(
        met_ids          = recon['mets'],
        pubchem_props_tsv= pubchem_props_tsv,
        output_h5        = os.path.join(output_dir, 'metabolite_features.h5'),
    )

    build_reaction_features_all_patients(
        rnaseq_h5    = rnaseq_h5,
        proteomics_h5= proteomics_h5,
        gpr_table_tsv= gpr_table_tsv,
        output_dir   = output_dir,
        n_reactions  = S.shape[1],
    )

    build_activity_labels(
        hma_mat_path = hma_mat,
        n_reactions  = S.shape[1],
        output_pt    = os.path.join(output_dir, 'activity_pseudolabels.pt'),
    )

    logger.info("=== Graph construction complete ===")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--recon3d_mat',        required=True)
    p.add_argument('--rnaseq_h5',          required=True)
    p.add_argument('--proteomics_h5',      required=True)
    p.add_argument('--pubchem_props_tsv',  required=True)
    p.add_argument('--gpr_table_tsv',      required=True)
    p.add_argument('--hma_mat',            required=True)
    p.add_argument('--output_dir',         default='./graph_data')
    args = p.parse_args()
    run_pipeline(args.recon3d_mat, args.rnaseq_h5, args.proteomics_h5,
                 args.pubchem_props_tsv, args.gpr_table_tsv,
                 args.hma_mat, args.output_dir)
