"""
TCGA-CRC RNA-seq Preprocessing Pipeline
Processes raw STAR-counts from the GDC portal to VST-normalised
gene expression matrices ready for MetaGNN reaction feature engineering.

Data source:
  TCGA-COAD and TCGA-READ cohorts via GDC Data Portal
  (https://portal.gdc.cancer.gov/)
  Access: Requires free GDC account + gdc-client download tool
  Project IDs: TCGA-COAD, TCGA-READ
  Data type: Gene Expression Quantification (STAR - Counts)
  Workflow: STAR 2-pass alignment, hg38 reference genome

Author: Thiptanawat Phongwattana
Affiliation: School of Information Technology, KMUTT
"""

import os
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Merge per-sample STAR count files into a gene × patient matrix
# ─────────────────────────────────────────────────────────────────────────────
def merge_star_counts(
    gdc_download_dir: str,
    manifest_tsv: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Merge individual GDC STAR count files into a single counts matrix.

    GDC download directory structure:
        gdc_download_dir/
            <file_uuid>/
                <sample_id>.star_gene_counts.tsv
            ...
        manifest_tsv:  GDC file manifest linking file_uuid → case_id/TCGA barcode

    Args:
        gdc_download_dir: path to folder produced by gdc-client download
        manifest_tsv:     GDC file manifest TSV (downloaded alongside data)
        output_path:      where to save the merged raw_counts.tsv

    Returns:
        counts_df: DataFrame (n_genes × n_patients) of unstranded read counts
    """
    manifest = pd.read_csv(manifest_tsv, sep='\t')
    # GDC manifests contain columns: id, filename, md5, size, state
    # We join to the clinical file to get TCGA barcodes
    counts_dict = {}

    for _, row in manifest.iterrows():
        file_path = os.path.join(gdc_download_dir, row['id'], row['filename'])
        if not os.path.exists(file_path):
            logger.warning(f"Missing file: {file_path}")
            continue

        # GDC STAR count files: skip 4 summary rows, then gene counts
        df = pd.read_csv(file_path, sep='\t', comment='N',
                         names=['gene_id', 'unstranded', 'stranded_first', 'stranded_second'],
                         skiprows=6)
        df = df.set_index('gene_id')
        sample_id = row['filename'].split('.')[0]
        counts_dict[sample_id] = df['unstranded']

    counts_df = pd.DataFrame(counts_dict)
    counts_df.index.name = 'gene_id'
    counts_df.to_csv(output_path, sep='\t')
    logger.info(f"Merged counts: {counts_df.shape[0]} genes × {counts_df.shape[1]} patients")
    return counts_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Filter lowly-expressed genes
# ─────────────────────────────────────────────────────────────────────────────
def filter_low_expression(
    counts_df: pd.DataFrame,
    min_count: int = 10,
    min_samples_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Retain genes with at least `min_count` reads in ≥ min_samples_frac of samples.

    Args:
        min_count:         minimum read count threshold
        min_samples_frac:  minimum fraction of samples that must pass threshold

    Returns:
        filtered_df: filtered counts DataFrame
    """
    min_samples = int(min_samples_frac * counts_df.shape[1])
    mask = (counts_df >= min_count).sum(axis=1) >= min_samples
    filtered = counts_df.loc[mask]
    logger.info(
        f"Gene filtering: {counts_df.shape[0]:,} → {filtered.shape[0]:,} genes "
        f"(min_count={min_count}, min_samples={min_samples})"
    )
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: VST normalisation (variance-stabilising transformation)
# ─────────────────────────────────────────────────────────────────────────────
def vst_normalise(
    counts_df: pd.DataFrame,
    method: str = 'pydeseq2',
) -> pd.DataFrame:
    """
    Apply Variance Stabilising Transformation (VST) to RNA-seq counts.

    Two implementations available:
      'pydeseq2' — uses the Python port of DESeq2 (Love et al. 2014);
                   recommended for reproducibility with the manuscript.
      'log1p'    — fallback: log2(CPM + 1) pseudo-VST; faster but less rigorous.

    Args:
        counts_df: raw counts matrix (genes × patients), integer values
        method:    'pydeseq2' (default) or 'log1p'

    Returns:
        vst_df: VST-normalised expression matrix (genes × patients)
    """
    if method == 'pydeseq2':
        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.default_inference import DefaultInference

            # PyDESeq2 expects samples × genes
            counts_T = counts_df.T.astype(int)
            metadata = pd.DataFrame(
                {'condition': ['tumour'] * len(counts_T)},
                index=counts_T.index,
            )
            inference = DefaultInference(n_cpus=os.cpu_count())
            dds = DeseqDataSet(
                counts=counts_T,
                metadata=metadata,
                design_factors='condition',
                inference=inference,
            )
            dds.deseq2()
            vst_T = dds.vst(fit_type='parametric')
            return pd.DataFrame(
                vst_T.X,
                index=vst_T.obs_names,
                columns=vst_T.var_names,
            ).T   # back to genes × patients
        except ImportError:
            logger.warning("pydeseq2 not installed — falling back to log1p CPM")
            method = 'log1p'

    if method == 'log1p':
        # Log2(CPM + 1) as approximate VST
        lib_sizes = counts_df.sum(axis=0)
        cpm = counts_df.div(lib_sizes, axis=1) * 1e6
        return np.log2(cpm + 1)

    raise ValueError(f"Unknown normalisation method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Map ENSG IDs to HGNC gene symbols and filter to Recon3D GPR genes
# ─────────────────────────────────────────────────────────────────────────────
def map_to_recon3d_genes(
    vst_df: pd.DataFrame,
    recon3d_gene_list: List[str],
    ensg_to_symbol: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Retain only genes present in Recon3D GPR associations.

    Args:
        vst_df:           VST-normalised expression (genes × patients);
                          index may be ENSG IDs or gene symbols
        recon3d_gene_list: list of HGNC gene symbols from Recon3D GPRs
        ensg_to_symbol:   optional mapping dict ENSG_ID → gene_symbol;
                          if None, assumes index already contains gene symbols

    Returns:
        mapped_df: filtered DataFrame (recon3d_genes × patients)
    """
    if ensg_to_symbol is not None:
        vst_df.index = vst_df.index.map(
            lambda x: ensg_to_symbol.get(x.split('.')[0], x)
        )
        vst_df = vst_df[~vst_df.index.duplicated(keep='first')]

    recon3d_set = set(recon3d_gene_list)
    mapped = vst_df.loc[vst_df.index.isin(recon3d_set)]
    logger.info(
        f"Mapped to Recon3D: {mapped.shape[0]:,} / {len(recon3d_gene_list):,} "
        f"GPR-associated genes found in RNA-seq data"
    )
    return mapped


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Save to HDF5
# ─────────────────────────────────────────────────────────────────────────────
def save_to_hdf5(
    vst_df: pd.DataFrame,
    output_h5: str,
    dataset_name: str = 'vst_expression',
):
    """
    Save the normalised expression matrix to HDF5 format.
    Stores gene IDs and patient barcodes as string datasets for indexing.

    Args:
        vst_df:       DataFrame (genes × patients)
        output_h5:    output file path (.h5)
        dataset_name: HDF5 dataset name for the expression matrix
    """
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset(dataset_name, data=vst_df.values.astype(np.float32),
                         compression='gzip', compression_opts=4)
        f.create_dataset('gene_ids',    data=np.array(vst_df.index, dtype='S'),
                         compression='gzip')
        f.create_dataset('patient_ids', data=np.array(vst_df.columns, dtype='S'),
                         compression='gzip')
        f.attrs['n_genes']    = vst_df.shape[0]
        f.attrs['n_patients'] = vst_df.shape[1]
        f.attrs['normalisation'] = 'DESeq2 VST (pydeseq2 v0.4.3)'
        f.attrs['genome_build']  = 'GRCh38 / hg38'
    logger.info(f"Saved VST matrix to: {output_h5}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    gdc_download_dir: str,
    manifest_tsv: str,
    recon3d_gene_list_path: str,
    output_dir: str,
    vst_method: str = 'pydeseq2',
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Merge STAR counts
    raw_path = os.path.join(output_dir, 'raw_counts.tsv')
    counts_df = merge_star_counts(gdc_download_dir, manifest_tsv, raw_path)

    # 2. Filter
    filtered_df = filter_low_expression(counts_df)

    # 3. VST normalise
    vst_df = vst_normalise(filtered_df, method=vst_method)

    # 4. Map to Recon3D genes
    recon3d_genes = pd.read_csv(recon3d_gene_list_path, header=None)[0].tolist()
    mapped_df     = map_to_recon3d_genes(vst_df, recon3d_genes)

    # 5. Save
    save_to_hdf5(mapped_df, os.path.join(output_dir, 'tcga_crc_rnaseq_vst.h5'))

    # 6. QC summary
    qc = pd.DataFrame({
        'median_library_size': counts_df.sum().median(),
        'n_genes_raw':         counts_df.shape[0],
        'n_genes_filtered':    filtered_df.shape[0],
        'n_genes_recon3d':     mapped_df.shape[0],
        'n_patients':          vst_df.shape[1],
        'vst_mean':            vst_df.values.mean(),
        'vst_std':             vst_df.values.std(),
    }, index=[0])
    qc.to_csv(os.path.join(output_dir, 'rnaseq_qc_summary.csv'), index=False)
    logger.info("RNA-seq preprocessing pipeline completed.")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='TCGA-CRC RNA-seq preprocessing')
    p.add_argument('--gdc_dir',       required=True)
    p.add_argument('--manifest',      required=True)
    p.add_argument('--recon3d_genes', required=True)
    p.add_argument('--output_dir',    default='./processed_rnaseq')
    p.add_argument('--vst_method',    default='pydeseq2', choices=['pydeseq2','log1p'])
    args = p.parse_args()
    run_pipeline(args.gdc_dir, args.manifest, args.recon3d_genes,
                 args.output_dir, args.vst_method)
