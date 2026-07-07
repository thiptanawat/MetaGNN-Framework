"""
TCGA-LUAD RNA-seq Data Acquisition Pipeline for MetaGNN
========================================================
Downloads STAR gene counts from the GDC Data Portal for TCGA-LUAD
(Lung Adenocarcinoma) using the GDC REST API.

This is the cross-cancer generalisability extension of MetaGNN,
originally validated on TCGA-CRC (colorectal cancer, 624 patients).

Data source:
  TCGA-LUAD via GDC Data Portal (https://portal.gdc.cancer.gov/)
  Project ID: TCGA-LUAD
  Data type: Gene Expression Quantification (STAR - Counts)
  Workflow: STAR 2-pass alignment, hg38 reference genome
  Access: Open access (no dbGaP authorization required for counts)

Expected yield: ~515 unique patients (primary tumour, 01A samples)

Usage:
  python 01_download_tcga_luad.py --output_dir ./data_luad/raw

Author: MetaGNN Team
"""

import os
import json
import logging
import argparse
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Query GDC for TCGA-LUAD STAR gene count file UUIDs
# ─────────────────────────────────────────────────────────────────────────────
def query_gdc_luad_files(sample_type: str = "Primary Tumor") -> pd.DataFrame:
    """
    Query the GDC API for all STAR gene count files in TCGA-LUAD.

    Returns a DataFrame with columns:
        file_id, file_name, file_size, case_id, submitter_id (TCGA barcode),
        sample_type
    """
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-LUAD"}},
            {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "data_format", "value": "TSV"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join([
            "file_id", "file_name", "file_size",
            "cases.case_id", "cases.submitter_id",
            "cases.samples.sample_type",
            "cases.samples.portions.analytes.aliquots.submitter_id",
        ]),
        "format": "JSON",
        "size": 2000,
    }

    logger.info("Querying GDC API for TCGA-LUAD STAR gene count files...")
    resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["data"]["hits"]
    logger.info(f"GDC returned {len(data)} files")

    records = []
    for hit in data:
        file_id   = hit["file_id"]
        file_name = hit["file_name"]
        file_size = hit.get("file_size", 0)

        for case in hit.get("cases", []):
            case_id      = case["case_id"]
            submitter_id = case["submitter_id"]   # TCGA barcode, e.g. TCGA-A1-A0SB

            # Determine sample type
            sample_types = set()
            for s in case.get("samples", []):
                sample_types.add(s.get("sample_type", "Unknown"))

            for st in sample_types:
                records.append({
                    "file_id":      file_id,
                    "file_name":    file_name,
                    "file_size":    file_size,
                    "case_id":      case_id,
                    "submitter_id": submitter_id,
                    "sample_type":  st,
                })

    df = pd.DataFrame(records)

    # Filter to primary tumour samples only
    if sample_type:
        df = df[df["sample_type"] == sample_type].copy()
        logger.info(f"After filtering to '{sample_type}': {len(df)} files")

    # Deduplicate: keep one file per patient (latest/largest)
    df = df.sort_values("file_size", ascending=False).drop_duplicates(
        subset="submitter_id", keep="first"
    ).reset_index(drop=True)
    logger.info(f"Unique patients after deduplication: {len(df)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Download STAR count files in batches via GDC data endpoint
# ─────────────────────────────────────────────────────────────────────────────
def download_star_counts(
    manifest_df: pd.DataFrame,
    output_dir: str,
    batch_size: int = 50,
    max_retries: int = 3,
) -> List[str]:
    """
    Download STAR gene count files from GDC, one per patient.

    Uses the GDC /data endpoint with a list of file UUIDs.
    Files are saved as <submitter_id>.star_gene_counts.tsv
    """
    os.makedirs(output_dir, exist_ok=True)
    downloaded = []

    for i in range(0, len(manifest_df), batch_size):
        batch = manifest_df.iloc[i:i+batch_size]
        batch_ids = batch["file_id"].tolist()

        for _, row in batch.iterrows():
            fid = row["file_id"]
            pid = row["submitter_id"]
            out_path = os.path.join(output_dir, f"{pid}.star_gene_counts.tsv")

            if os.path.exists(out_path):
                downloaded.append(out_path)
                continue

            for attempt in range(max_retries):
                try:
                    resp = requests.get(
                        f"{GDC_DATA_ENDPOINT}/{fid}",
                        headers={"Content-Type": "application/json"},
                        timeout=120,
                    )
                    resp.raise_for_status()

                    with open(out_path, 'wb') as f:
                        f.write(resp.content)
                    downloaded.append(out_path)
                    break
                except Exception as e:
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {pid}: {e}")
                    time.sleep(2 ** attempt)

        logger.info(f"Downloaded {len(downloaded)}/{len(manifest_df)} files")

    return downloaded


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Merge individual STAR count files into gene × patient matrix
# ─────────────────────────────────────────────────────────────────────────────
def merge_star_counts(
    download_dir: str,
    output_tsv: str,
) -> pd.DataFrame:
    """
    Merge individual STAR count TSV files into a unified gene × patient TPM matrix.

    GDC STAR count files have columns:
        gene_id | gene_name | gene_type | unstranded | stranded_first_read_count |
        stranded_second_read_count | tpm_unstranded | fpkm_unstranded | fpkm_uq_unstranded

    We extract tpm_unstranded for consistency with MetaGNN-CRC.
    """
    files = sorted(Path(download_dir).glob("*.star_gene_counts.tsv"))
    logger.info(f"Merging {len(files)} STAR count files...")

    counts_dict = {}
    gene_lengths = None

    for fpath in files:
        pid = fpath.stem.replace(".star_gene_counts", "")
        try:
            # Read the STAR count file robustly:
            #  - Line-by-line to handle variable header formats
            #  - DO NOT use comment='N' — it truncates 'ENSG...' to 'E'
            #    because pandas treats 'N' as an in-field comment character
            rows = []
            with open(fpath, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 7 and parts[0].startswith('ENSG'):
                        # cols: gene_id, gene_name, gene_type,
                        #       unstranded, stranded_first, stranded_second,
                        #       tpm_unstranded, [fpkm_unstranded, fpkm_uq_unstranded]
                        try:
                            tpm = float(parts[6])
                            rows.append((parts[0], tpm))
                        except ValueError:
                            continue

            if not rows:
                logger.warning(f"No ENSG rows in {fpath.name}, skipping")
                continue

            gene_ids, tpm_vals = zip(*rows)
            series = pd.Series(tpm_vals, index=gene_ids, dtype=float, name=pid)
            counts_dict[pid] = series
        except Exception as e:
            logger.warning(f"Skipping {fpath.name}: {e}")
            continue

    tpm_df = pd.DataFrame(counts_dict)
    tpm_df.index.name = 'gene_id'

    # log2(TPM + 1) transform
    tpm_log = np.log2(tpm_df + 1)

    tpm_log.to_csv(output_tsv, sep='\t')
    logger.info(f"Merged TPM matrix: {tpm_log.shape[0]} genes × {tpm_log.shape[1]} patients → {output_tsv}")

    # ── Extract gene_id → gene_name mapping (Ensembl → HUGO symbol) ──
    # Use a SEPARATE robust read of the first file (not the comment/skiprows
    # parse used for TPM, which can misalign columns).
    gene_id_to_name = _extract_gene_name_mapping(files[0] if files else None)

    mapping_path = str(Path(output_tsv).parent / 'gene_id_to_name.tsv')
    mapping_df = pd.DataFrame([
        {'gene_id': gid, 'gene_name': gname}
        for gid, gname in gene_id_to_name.items()
    ])
    mapping_df.to_csv(mapping_path, sep='\t', index=False)
    logger.info(f"Gene ID mapping: {len(mapping_df)} entries → {mapping_path}")

    return tpm_log


def _extract_gene_name_mapping(star_file: Optional[Path]) -> Dict[str, str]:
    """
    Robustly extract gene_id → gene_name from a GDC STAR count file.

    GDC STAR count files have varying formats:
      Format A: header on line 0, N_ summary rows on lines 1-4, gene data from line 5
      Format B: N_ summary rows on lines 0-3, header on line 4, gene data from line 5
      Format C: header on line 0, gene data from line 1, N_ rows at the end

    This function handles ALL formats by:
      1. Reading the full file with header detection
      2. Keeping only rows where gene_id starts with 'ENSG'
    """
    if star_file is None:
        return {}

    logger.info(f"Extracting gene_id → gene_name mapping from: {star_file.name}")

    mapping = {}

    # Strategy 1: Read with pandas header detection
    try:
        # Read entire file, let pandas detect the header
        df_full = pd.read_csv(star_file, sep='\t', header=0)
        cols = list(df_full.columns)
        logger.info(f"  File columns: {cols[:4]}...")

        # Find the gene_id and gene_name columns by name or position
        gid_col = None
        gname_col = None

        for c in cols:
            cl = str(c).lower().strip()
            if cl == 'gene_id':
                gid_col = c
            elif cl == 'gene_name':
                gname_col = c

        if gid_col and gname_col:
            for _, row in df_full.iterrows():
                gid = str(row[gid_col]).strip()
                gname = str(row[gname_col]).strip()
                if gid.startswith('ENSG') and gname and gname != 'nan':
                    mapping[gid] = gname
            logger.info(f"  Strategy 1 (header detection): {len(mapping)} ENSG genes mapped")
            if mapping:
                return mapping
    except Exception as e:
        logger.warning(f"  Strategy 1 failed: {e}")

    # Strategy 2: Raw line-by-line parsing
    try:
        with open(star_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[0].startswith('ENSG'):
                    mapping[parts[0]] = parts[1]
        logger.info(f"  Strategy 2 (line-by-line): {len(mapping)} ENSG genes mapped")
    except Exception as e:
        logger.warning(f"  Strategy 2 failed: {e}")

    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download TCGA-LUAD RNA-seq for MetaGNN")
    parser.add_argument("--output_dir", default="./data_luad/raw",
                        help="Output directory for downloaded files")
    parser.add_argument("--merged_output", default="./data_luad/tcga_luad_tpm_log2.tsv",
                        help="Path for merged gene × patient matrix")
    parser.add_argument("--manifest_output", default="./data_luad/gdc_manifest_luad.tsv",
                        help="Path to save GDC file manifest")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, only merge existing files")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.merged_output), exist_ok=True)

    if not args.skip_download:
        # Query GDC for file list
        manifest_df = query_gdc_luad_files(sample_type="Primary Tumor")
        manifest_df.to_csv(args.manifest_output, sep='\t', index=False)
        logger.info(f"Manifest saved: {args.manifest_output}")

        # Download
        download_star_counts(manifest_df, args.output_dir)

    # Merge into matrix
    tpm_log = merge_star_counts(args.output_dir, args.merged_output)
    logger.info(f"DONE. {tpm_log.shape[1]} patients ready for MetaGNN preprocessing.")


if __name__ == "__main__":
    main()
