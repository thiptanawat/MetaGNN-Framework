"""
Build Recon3D Graph Structure (Self-Contained)
==============================================
Generates all shared Recon3D graph files from scratch, eliminating
the dependency on MetaGNN-CRC/code/processed_690/.

Downloads Recon3D v3.0 via COBRApy/BiGG and builds:
  - edge_indices/substrate_of.pt    (M→R stoichiometric edges)
  - edge_indices/produces.pt        (R→M stoichiometric edges)
  - edge_indices/shared_metabolite.pt (R↔R co-occurrence edges)
  - metabolite_features.h5          (X_M: 5835 × 519)
  - recon3d_stoich.h5               (stoichiometric matrix S)
  - gpr_table.tsv                   (reaction → gene sets mapping)

These files are tissue-independent and can be reused for ANY
cancer type (BRCA, CRC, LUAD, etc.).

Usage:
  python 00_build_recon3d_graph.py --output_dir ./data_brca/processed/

Requirements:
  pip install cobra rdkit-pypi torch numpy scipy pandas h5py

Author: MetaGNN Team
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Load or Download Recon3D
# ═════════════════════════════════════════════════════════════════════════════
def load_recon3d(model_path: Optional[str] = None) -> 'cobra.Model':
    """
    Load Recon3D v3.0. If no path provided, attempts download from BiGG.

    Download sources (in order of preference):
      1. Local .mat or .xml file (if --recon3d_path provided)
      2. BiGG Models: http://bigg.ucsd.edu/models/Recon3D
      3. VMH: https://www.vmh.life/#downloadview
    """
    import cobra

    if model_path and os.path.exists(model_path):
        logger.info(f"Loading Recon3D from: {model_path}")
        if model_path.endswith('.mat'):
            model = cobra.io.load_matlab_model(model_path)
        elif model_path.endswith('.xml') or model_path.endswith('.xml.gz'):
            model = cobra.io.read_sbml_model(model_path)
        elif model_path.endswith('.json'):
            model = cobra.io.load_json_model(model_path)
        else:
            raise ValueError(f"Unsupported format: {model_path}")
    else:
        logger.info("Downloading Recon3D from BiGG Models...")
        try:
            import requests
            # BiGG Models SBML download
            url = "http://bigg.ucsd.edu/static/models/Recon3D.xml.gz"
            local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'Recon3D.xml.gz')
            if not os.path.exists(local_path):
                logger.info(f"Downloading from {url} ...")
                resp = requests.get(url, stream=True, timeout=300)
                resp.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded: {local_path}")
            model = cobra.io.read_sbml_model(local_path)
        except Exception as e:
            logger.error(f"Failed to download Recon3D: {e}")
            logger.error("Please download manually from https://www.vmh.life/#downloadview")
            logger.error("  or http://bigg.ucsd.edu/models/Recon3D")
            logger.error("Then run: python 00_build_recon3d_graph.py --recon3d_path /path/to/Recon3D.xml")
            sys.exit(1)

    logger.info(f"Recon3D loaded: {len(model.reactions)} reactions, "
                f"{len(model.metabolites)} metabolites, "
                f"{len(model.genes)} genes")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Build Stoichiometric Matrix
# ═════════════════════════════════════════════════════════════════════════════
def build_stoichiometric_matrix(model, output_dir: str):
    """
    Build stoichiometric matrix S (n_met × n_rxn) and save.
    """
    rxn_ids = [r.id for r in model.reactions]
    met_ids = [m.id for m in model.metabolites]
    n_met = len(met_ids)
    n_rxn = len(rxn_ids)

    logger.info(f"Building S matrix: {n_met} metabolites × {n_rxn} reactions")

    met_idx = {m: i for i, m in enumerate(met_ids)}
    rxn_idx = {r: i for i, r in enumerate(rxn_ids)}

    rows, cols, vals = [], [], []
    for rxn in model.reactions:
        j = rxn_idx[rxn.id]
        for met, coeff in rxn.metabolites.items():
            i = met_idx[met.id]
            rows.append(i)
            cols.append(j)
            vals.append(float(coeff))

    S = sp.csr_matrix((vals, (rows, cols)), shape=(n_met, n_rxn))
    S_dense = np.array(S.todense(), dtype=np.float32)

    stoich_path = os.path.join(output_dir, 'recon3d_stoich.h5')
    with h5py.File(stoich_path, 'w') as f:
        f.create_dataset('S', data=S_dense, compression='gzip')
        f.create_dataset('met_ids',
                         data=np.array(met_ids, dtype='S'),
                         compression='gzip')
        f.create_dataset('rxn_ids',
                         data=np.array(rxn_ids, dtype='S'),
                         compression='gzip')
    logger.info(f"Stoichiometric matrix saved: {S_dense.shape} → {stoich_path}")

    return S_dense, met_ids, rxn_ids


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Build Edge Indices
# ═════════════════════════════════════════════════════════════════════════════
def build_edge_indices(S: np.ndarray, output_dir: str):
    """
    Derive three edge sets from S (n_met × n_rxn):
      substrate_of : M → R  (S[m,r] < 0)
      produces     : R → M  (S[m,r] > 0)
      shared_metabolite : R ↔ R  (share ≥1 non-currency metabolite)
    """
    import torch

    edge_dir = os.path.join(output_dir, 'edge_indices')
    os.makedirs(edge_dir, exist_ok=True)

    # substrate_of: M → R (negative stoichiometric coefficients)
    met_sub, rxn_sub = np.where(S < 0)
    ei_sub = torch.tensor(np.stack([met_sub, rxn_sub]), dtype=torch.long)
    torch.save(ei_sub, os.path.join(edge_dir, 'substrate_of.pt'))
    logger.info(f"substrate_of edges: {ei_sub.shape[1]:,}")

    # produces: R → M (positive stoichiometric coefficients)
    met_prod, rxn_prod = np.where(S > 0)
    ei_prod = torch.tensor(np.stack([rxn_prod, met_prod]), dtype=torch.long)
    torch.save(ei_prod, os.path.join(edge_dir, 'produces.pt'))
    logger.info(f"produces edges: {ei_prod.shape[1]:,}")

    # shared_metabolite: R ↔ R (co-occurrence matrix)
    # Exclude currency metabolites (degree > 150)
    met_degree = (S != 0).astype(np.float32).sum(axis=1)
    currency_mask = met_degree > 150
    n_currency = currency_mask.sum()
    logger.info(f"Currency metabolites (degree > 150): {n_currency}")

    # Build participation matrix excluding currency metabolites
    S_filtered = S.copy()
    S_filtered[currency_mask.astype(bool).ravel(), :] = 0
    P = (S_filtered != 0).astype(np.float32)
    shared = P.T @ P  # n_rxn × n_rxn
    np.fill_diagonal(shared, 0)

    # Full shared_metabolite edge set
    r1, r2 = np.where(shared > 0)
    ei_shared = torch.tensor(np.stack([r1, r2]), dtype=torch.long)
    torch.save(ei_shared, os.path.join(edge_dir, 'shared_metabolite.pt'))
    logger.info(f"shared_metabolite edges: {ei_shared.shape[1]:,}")

    return {
        'substrate_of': ei_sub.shape[1],
        'produces': ei_prod.shape[1],
        'shared_metabolite': ei_shared.shape[1],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Build Metabolite Features (X_M)
# ═════════════════════════════════════════════════════════════════════════════
def build_metabolite_features(model, met_ids: List[str], output_dir: str):
    """
    Build X_M ∈ ℝ^(n_met × 519):
      - 7 physico-chemical properties
      - 512-bit Morgan fingerprints (radius=2)

    Uses RDKit for molecular descriptors. Falls back to zero-fill
    for metabolites without SMILES or when RDKit is unavailable.
    """
    n_met = len(met_ids)
    X_M = np.zeros((n_met, 519), dtype=np.float32)

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem
        HAS_RDKIT = True
        logger.info("RDKit available — computing molecular features")
    except ImportError:
        HAS_RDKIT = False
        logger.warning("RDKit not available — using zero-filled metabolite features")
        logger.warning("Install: pip install rdkit-pypi")

    # Try to get formulae/annotations from the COBRA model
    met_with_features = 0
    for i, mid in enumerate(met_ids):
        met_obj = model.metabolites.get_by_id(mid)

        if HAS_RDKIT:
            # Try to get SMILES from annotation
            smiles = None
            if hasattr(met_obj, 'annotation'):
                for key in ['smiles', 'SMILES', 'inchi']:
                    if key in met_obj.annotation:
                        val = met_obj.annotation[key]
                        if isinstance(val, list):
                            val = val[0]
                        if key.lower() == 'inchi':
                            mol = Chem.MolFromInchi(str(val))
                            if mol:
                                smiles = Chem.MolToSmiles(mol)
                        else:
                            smiles = str(val)
                        break

            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # 7 physico-chemical properties
                    X_M[i, 0] = Descriptors.MolWt(mol)
                    X_M[i, 1] = Descriptors.MolLogP(mol)
                    X_M[i, 2] = Descriptors.NumHAcceptors(mol)
                    X_M[i, 3] = Descriptors.NumHDonors(mol)
                    X_M[i, 4] = Descriptors.TPSA(mol)
                    X_M[i, 5] = Descriptors.RingCount(mol)
                    X_M[i, 6] = Chem.GetFormalCharge(mol)

                    # 512-bit Morgan fingerprint
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
                    X_M[i, 7:519] = np.array(fp, dtype=np.float32)
                    met_with_features += 1

        # Use formula-based features as fallback
        if X_M[i].sum() == 0 and met_obj.formula:
            formula = met_obj.formula
            # Crude MW estimate from formula
            mw = 0.0
            atom_weights = {'C': 12.01, 'H': 1.008, 'N': 14.01,
                           'O': 16.00, 'P': 30.97, 'S': 32.06, 'Fe': 55.85}
            import re
            for atom, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
                count = int(count) if count else 1
                mw += atom_weights.get(atom, 0) * count
            X_M[i, 0] = mw
            if mw > 0:
                met_with_features += 1

    logger.info(f"Metabolite features: {met_with_features}/{n_met} with non-zero features")

    met_h5 = os.path.join(output_dir, 'metabolite_features.h5')
    with h5py.File(met_h5, 'w') as f:
        f.create_dataset('X_M', data=X_M, compression='gzip')
        f.create_dataset('met_ids',
                         data=np.array(met_ids, dtype='S'),
                         compression='gzip')
        f.attrs['shape'] = str(X_M.shape)
        f.attrs['dims'] = '519: [0:7]=physico-chemical, [7:519]=Morgan_FP_r2_512bit'
    logger.info(f"Metabolite features saved: {X_M.shape} → {met_h5}")


# ═════════════════════════════════════════════════════════════════════════════
# Step 5: Build GPR Table
# ═════════════════════════════════════════════════════════════════════════════
def build_gpr_table(model, rxn_ids: List[str], output_dir: str):
    """
    Build GPR table mapping reaction index → gene sets.

    Format: rxn_idx, gene_sets_str
    gene_sets_str is a Python list-of-lists string:
      [['gene1', 'gene2'], ['gene3']]
    where inner lists are AND (complex subunits), outer list is OR (isozymes).
    """
    records = []

    for i, rxn_id in enumerate(rxn_ids):
        rxn = model.reactions.get_by_id(rxn_id)
        gpr = rxn.gene_reaction_rule.strip()

        if not gpr:
            records.append({'rxn_idx': i, 'gene_sets_str': "[['[]']]"})
            continue

        # Parse GPR rule into list-of-lists
        # OR groups (isozymes): split by ' or '
        # AND groups (complex): split by ' and '
        try:
            # Remove parentheses for simple parsing
            clean = gpr.replace('(', '').replace(')', '')
            or_groups = clean.split(' or ')
            gene_sets = []
            for grp in or_groups:
                genes = [g.strip() for g in grp.split(' and ') if g.strip()]
                if genes:
                    gene_sets.append(genes)

            if gene_sets:
                records.append({'rxn_idx': i, 'gene_sets_str': str(gene_sets)})
            else:
                records.append({'rxn_idx': i, 'gene_sets_str': "[['[]']]"})
        except:
            records.append({'rxn_idx': i, 'gene_sets_str': "[['[]']]"})

    gpr_df = pd.DataFrame(records)
    gpr_path = os.path.join(output_dir, 'gpr_table.tsv')
    gpr_df.to_csv(gpr_path, sep='\t', index=False)

    n_with_genes = sum(1 for r in records
                       if r['gene_sets_str'] != "[['[]']]")
    logger.info(f"GPR table: {len(records)} reactions, "
                f"{n_with_genes} with GPR rules → {gpr_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Build Recon3D graph structure (self-contained, no CRC dependency)"
    )
    parser.add_argument("--output_dir", default="./data_brca/processed/",
                        help="Output directory for graph files")
    parser.add_argument("--recon3d_path", default=None,
                        help="Path to Recon3D model file (.mat, .xml, .json). "
                             "If not provided, downloads from BiGG.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Building Recon3D Graph Structure (Self-Contained)")
    logger.info("=" * 60)

    # 1. Load Recon3D
    model = load_recon3d(args.recon3d_path)

    # 2. Build stoichiometric matrix
    S, met_ids, rxn_ids = build_stoichiometric_matrix(model, args.output_dir)

    # 3. Build edge indices
    edge_stats = build_edge_indices(S, args.output_dir)

    # 4. Build metabolite features
    build_metabolite_features(model, met_ids, args.output_dir)

    # 5. Build GPR table
    build_gpr_table(model, rxn_ids, args.output_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("Recon3D Graph Build Complete!")
    logger.info("=" * 60)
    logger.info(f"  Reactions:             {len(rxn_ids):,}")
    logger.info(f"  Metabolites:           {len(met_ids):,}")
    logger.info(f"  substrate_of edges:    {edge_stats['substrate_of']:,}")
    logger.info(f"  produces edges:        {edge_stats['produces']:,}")
    logger.info(f"  shared_met edges:      {edge_stats['shared_metabolite']:,}")
    logger.info(f"  Output:                {args.output_dir}")
    logger.info("=" * 60)
    logger.info("These graph files are tissue-independent and reusable")
    logger.info("for ANY cancer type (BRCA, LUAD, LIHC, etc.)")

    # Save build metadata
    meta = {
        'recon3d_source': args.recon3d_path or 'BiGG (auto-download)',
        'n_reactions': len(rxn_ids),
        'n_metabolites': len(met_ids),
        'edge_counts': edge_stats,
    }
    with open(os.path.join(args.output_dir, 'graph_build_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
