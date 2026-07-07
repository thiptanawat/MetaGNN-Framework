"""
CPTAC-CRC Proteomics Preprocessing Pipeline
Processes TMT-10plex LC-MS/MS protein abundance data from the
Proteomics Data Commons (PDC) portal into log2-normalised matrices
compatible with MetaGNN reaction feature engineering.

Data source:
  CPTAC Colorectal Cancer Cohort (CPTAC-3)
  Proteomics Data Commons: https://pdc.cancer.gov/
  Study: PDC000116 (CPTAC COAD), PDC000220 (CPTAC READ)
  Access: Requires free PDC account; bulk download via PDC API
  Quantification: TMT-10plex isobaric labelling (Paulovich lab protocol)
  Instrument: Orbitrap Fusion Lumos, 2h gradient

Author: Thiptanawat Phongwattana
Affiliation: School of Information Technology, KMUTT
"""

import os
import logging
import numpy as np
import pandas as pd
import h5py
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load PDC TMT abundance matrix
# ─────────────────────────────────────────────────────────────────────────────
def load_pdc_tmt_matrix(
    pdc_tsv_path: str,
    id_column: str = 'Gene',
) -> pd.DataFrame:
    """
    Load TMT protein abundance matrix downloaded from PDC.

    PDC download format (TSV):
      - Rows: protein/gene identifiers
      - Columns: sample aliquot IDs (e.g. 'C3L-00032.N', 'C3L-00032.T')
      - Values: TMT reporter ion ratio (log2-normalised within each plex)

    Args:
        pdc_tsv_path: path to downloaded PDC abundance TSV
        id_column:    column name containing gene/protein identifiers

    Returns:
        prot_df: DataFrame (proteins × samples)
    """
    df = pd.read_csv(pdc_tsv_path, sep='\t', index_col=id_column)
    # Drop non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    logger.info(f"Loaded PDC matrix: {df.shape[0]} proteins × {df.shape[1]} samples")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Select tumour samples and match to TCGA barcodes
# ─────────────────────────────────────────────────────────────────────────────
def select_tumour_samples(
    prot_df: pd.DataFrame,
    pdc_clinical: pd.DataFrame,
    tcga_matched_ids: List[str],
) -> pd.DataFrame:
    """
    Retain only primary tumour aliquots (.T suffix in CPTAC) that have
    matched TCGA RNA-seq profiles in our cohort.

    Args:
        prot_df:         proteins × samples matrix (all aliquots)
        pdc_clinical:    PDC clinical table linking aliquot_id → case_id → TCGA barcode
        tcga_matched_ids: list of TCGA barcodes with matched RNA-seq data

    Returns:
        tumour_df: proteins × patients (TCGA barcodes as columns)
    """
    # Map aliquot IDs to TCGA barcodes
    aliquot_to_tcga = pdc_clinical.set_index('aliquot_id')['tcga_barcode'].to_dict()

    # Keep tumour samples (.T) only
    tumour_cols = [c for c in prot_df.columns
                   if str(c).endswith('.T') or 'Tumor' in str(c)]

    prot_tumour = prot_df[tumour_cols].copy()
    prot_tumour.columns = [aliquot_to_tcga.get(c, c) for c in prot_tumour.columns]

    # Retain patients with matched RNA-seq
    matched = [c for c in prot_tumour.columns if c in set(tcga_matched_ids)]
    prot_tumour = prot_tumour[matched]

    logger.info(
        f"Tumour samples retained: {prot_tumour.shape[1]} / {len(tcga_matched_ids)} "
        f"matched to RNA-seq cohort"
    )
    return prot_tumour


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Within-sample median normalisation + missing value imputation
# ─────────────────────────────────────────────────────────────────────────────
def normalise_and_impute(
    prot_df: pd.DataFrame,
    impute_method: str = 'knn',
    knn_k: int = 5,
    min_completeness: float = 0.70,
) -> pd.DataFrame:
    """
    1. Subtract per-sample median (correct loading bias between TMT plexes).
    2. Filter proteins below completeness threshold.
    3. Impute remaining missing values.

    Args:
        prot_df:          proteins × patients (log2 TMT ratios, may contain NaN)
        impute_method:    'knn' (default) or 'min_shifted' (half-minimum per protein)
        knn_k:            number of nearest neighbours for KNN imputation
        min_completeness: minimum fraction of patients with valid quantification

    Returns:
        norm_df: normalised and imputed DataFrame (proteins × patients), no NaN
    """
    # Within-sample median centering
    sample_medians = prot_df.median(axis=0)
    prot_centred   = prot_df.subtract(sample_medians, axis=1)

    # Filter by completeness
    completeness = prot_centred.notna().mean(axis=1)
    prot_filtered = prot_centred.loc[completeness >= min_completeness].copy()
    logger.info(
        f"Completeness filter (≥{min_completeness:.0%}): "
        f"{prot_centred.shape[0]:,} → {prot_filtered.shape[0]:,} proteins"
    )

    # Imputation
    if impute_method == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=knn_k)
        imputed = imputer.fit_transform(prot_filtered.T).T
        prot_imputed = pd.DataFrame(
            imputed, index=prot_filtered.index, columns=prot_filtered.columns
        )
    elif impute_method == 'min_shifted':
        # Half-minimum per protein (reflects MNAR assumption)
        prot_imputed = prot_filtered.copy()
        for prot_id in prot_imputed.index:
            row = prot_imputed.loc[prot_id]
            fill_val = row.min() / 2.0
            prot_imputed.loc[prot_id] = row.fillna(fill_val)
    else:
        raise ValueError(f"Unknown impute_method: {impute_method}")

    logger.info(f"Imputation ({impute_method}) complete. Remaining NaN: {prot_imputed.isna().sum().sum()}")
    return prot_imputed


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Map proteins to Recon3D GPR-associated genes
# ─────────────────────────────────────────────────────────────────────────────
def map_to_recon3d_proteins(
    prot_df: pd.DataFrame,
    recon3d_gene_list: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retain proteins with HGNC symbols matching Recon3D GPR associations.

    Returns:
        mapped_df:  filtered DataFrame
        coverage_df: summary of coverage per gene (for data_processing Figure 2C)
    """
    recon3d_set = set(recon3d_gene_list)
    mapped = prot_df.loc[prot_df.index.isin(recon3d_set)]

    coverage = pd.DataFrame({
        'gene': list(recon3d_set),
        'found_in_proteomics': [g in set(mapped.index) for g in recon3d_set],
    })
    logger.info(
        f"Mapped to Recon3D GPR genes: {mapped.shape[0]:,} / {len(recon3d_set):,}"
    )
    return mapped, coverage


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Save to HDF5
# ─────────────────────────────────────────────────────────────────────────────
def save_to_hdf5(prot_df: pd.DataFrame, output_h5: str):
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('protein_abundance', data=prot_df.values.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('gene_ids',    data=np.array(prot_df.index, dtype='S'),
                         compression='gzip')
        f.create_dataset('patient_ids', data=np.array(prot_df.columns, dtype='S'),
                         compression='gzip')
        f.attrs['n_proteins'] = prot_df.shape[0]
        f.attrs['n_patients'] = prot_df.shape[1]
        f.attrs['normalisation'] = 'TMT-10plex, within-plex median centering, KNN imputation'
        f.attrs['source'] = 'PDC CPTAC-3 (PDC000116, PDC000220)'
    logger.info(f"Saved proteomics matrix to: {output_h5}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    pdc_tsv_path: str,
    pdc_clinical_path: str,
    tcga_barcodes_path: str,
    recon3d_gene_list_path: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    prot_df     = load_pdc_tmt_matrix(pdc_tsv_path)
    pdc_clin    = pd.read_csv(pdc_clinical_path, sep='\t')
    tcga_ids    = pd.read_csv(tcga_barcodes_path, header=None)[0].tolist()
    recon3d_g   = pd.read_csv(recon3d_gene_list_path, header=None)[0].tolist()

    prot_tumour  = select_tumour_samples(prot_df, pdc_clin, tcga_ids)
    prot_norm    = normalise_and_impute(prot_tumour, impute_method='knn', knn_k=5)
    prot_mapped, cov_df = map_to_recon3d_proteins(prot_norm, recon3d_g)

    save_to_hdf5(prot_mapped, os.path.join(output_dir, 'cptac_crc_protein_tmt.h5'))
    cov_df.to_csv(os.path.join(output_dir, 'protein_recon3d_coverage.csv'), index=False)
    logger.info("Proteomics preprocessing pipeline completed.")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='CPTAC-CRC proteomics preprocessing')
    p.add_argument('--pdc_tsv',         required=True)
    p.add_argument('--pdc_clinical',    required=True)
    p.add_argument('--tcga_barcodes',   required=True)
    p.add_argument('--recon3d_genes',   required=True)
    p.add_argument('--output_dir',      default='./processed_proteomics')
    args = p.parse_args()
    run_pipeline(args.pdc_tsv, args.pdc_clinical, args.tcga_barcodes,
                 args.recon3d_genes, args.output_dir)
