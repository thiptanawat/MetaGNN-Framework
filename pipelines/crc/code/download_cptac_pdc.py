#!/usr/bin/env python3
"""
Quick standalone script to download CPTAC colon proteomics from PDC API.
Study: PDC000111 — TCGA Colon Cancer Proteome (Label Free, 90 cases)

Usage:
    python download_cptac_pdc.py --output_dir ./raw_downloads
"""

import os
import sys
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def download_from_pdc(output_dir: str):
    import requests
    import pandas as pd
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    prot_tsv = os.path.join(output_dir, 'cptac_colon_proteomics.tsv')
    clin_tsv = os.path.join(output_dir, 'cptac_colon_clinical.tsv')

    if os.path.exists(prot_tsv):
        logger.info(f"Already exists: {prot_tsv}")
        return

    pdc_url = "https://pdc.cancer.gov/graphql"

    # ── Step 1: Get the study ID for PDC000111 ────────────────────────────
    logger.info("Querying PDC for study PDC000111...")

    # Use the known study_id from PDC000111
    study_id = "b998098f-57b8-11e8-b07a-00a098d917f8"
    logger.info(f"Study ID: {study_id}")

    # ── Step 2: Try quantDataMatrix (processed log2 ratios) ──────────────
    logger.info("Requesting quantitative data matrix (log2_ratio)...")
    logger.info("This may take 1-3 minutes...")

    for data_type in ["log2_ratio", "unshared_log2_ratio", "spectral_count"]:
        logger.info(f"  Trying data_type='{data_type}'...")

        quant_query = """
        {{
            quantDataMatrix(
                study_id: "{sid}"
                data_type: "{dtype}"
                acceptDUA: true
            )
        }}
        """.format(sid=study_id, dtype=data_type)

        try:
            resp = requests.post(
                pdc_url,
                json={"query": quant_query},
                timeout=300
            )
            resp.raise_for_status()
            result = resp.json()

            matrix_data = result.get('data', {}).get('quantDataMatrix', [])

            if matrix_data and len(matrix_data) > 1:
                logger.info(f"  Got matrix: {len(matrix_data)} rows (including header)")

                # First row = header (Gene + aliquot IDs)
                header = matrix_data[0]
                rows = matrix_data[1:]

                gene_names = [row[0] for row in rows]
                values = []
                for row in rows:
                    vals = []
                    for v in row[1:]:
                        try:
                            vals.append(float(v) if v not in (None, '', 'NA', 'N/A') else np.nan)
                        except (ValueError, TypeError):
                            vals.append(np.nan)
                    values.append(vals)

                prot_df = pd.DataFrame(
                    values,
                    index=gene_names,
                    columns=header[1:]
                )
                prot_df.index.name = 'Gene'

                # Remove fully empty rows/columns
                prot_df = prot_df.dropna(how='all', axis=0)
                prot_df = prot_df.dropna(how='all', axis=1)

                logger.info(f"Proteomics matrix: {prot_df.shape[0]} genes × {prot_df.shape[1]} samples")
                logger.info(f"Sample IDs (first 5): {list(prot_df.columns[:5])}")
                logger.info(f"Non-NaN fraction: {prot_df.notna().mean().mean():.1%}")

                prot_df.to_csv(prot_tsv, sep='\t')
                logger.info(f"Saved: {prot_tsv}")

                # Create clinical mapping
                patient_ids = list(prot_df.columns)
                clin_df = pd.DataFrame({
                    'aliquot_id': patient_ids,
                    'tcga_barcode': [pid.split('.')[0] if '.' in pid else pid
                                     for pid in patient_ids],
                })
                clin_df.index = patient_ids
                clin_df.index.name = 'Patient_ID'
                clin_df.to_csv(clin_tsv, sep='\t')
                logger.info(f"Saved: {clin_tsv}")

                return prot_tsv, clin_tsv

            else:
                logger.warning(f"  data_type='{data_type}' returned empty matrix")

        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout for data_type='{data_type}'")
        except Exception as e:
            logger.warning(f"  Failed for data_type='{data_type}': {e}")

    # ── Step 3: Fallback — download individual protein assembly files ─────
    logger.info("\nquantDataMatrix failed. Trying file-based download...")

    file_query = """
    {{
        filesPerStudy(
            study_id: "{sid}"
            acceptDUA: true
        ) {{
            file_id
            file_name
            file_type
            data_category
            file_size
            md5sum
            signedUrl {{
                url
            }}
        }}
    }}
    """.format(sid=study_id)

    try:
        resp = requests.post(pdc_url, json={"query": file_query}, timeout=120)
        resp.raise_for_status()
        files = resp.json().get('data', {}).get('filesPerStudy', [])

        logger.info(f"Found {len(files)} files in study")

        # Filter for Protein Assembly text files
        prot_files = [f for f in files
                      if 'protein' in f.get('data_category', '').lower()
                      and f.get('file_type', '').lower() in ('text', 'tsv', 'csv')]

        if not prot_files:
            # Try broader filter
            prot_files = [f for f in files
                          if 'protein assembly' in f.get('data_category', '').lower()]

        logger.info(f"Protein assembly files: {len(prot_files)}")
        for pf in prot_files:
            logger.info(f"  {pf.get('file_name')} ({pf.get('file_size', 0) / 1024:.0f} KB)")
            url_info = pf.get('signedUrl', {})
            if url_info and url_info.get('url'):
                dl_url = url_info['url']
                fname = pf['file_name']
                out_path = os.path.join(output_dir, fname)

                logger.info(f"  Downloading {fname}...")
                dl_resp = requests.get(dl_url, stream=True, timeout=120)
                dl_resp.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in dl_resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                logger.info(f"  Saved: {out_path}")

        if prot_files:
            logger.info("\nProtein assembly files downloaded.")
            logger.info("You may need to manually combine them into a single matrix.")
            return None, None

    except Exception as e:
        logger.error(f"File download also failed: {e}")

    logger.error(
        "\n"
        "Could not download CPTAC data automatically.\n"
        "This is OK — only 90/690 patients have CPTAC proteomics.\n"
        "You can proceed without it (595/690 patients are zero-filled anyway).\n"
        "\n"
        "To skip CPTAC in the main pipeline:\n"
        "  python 00_download_and_generate_690.py --skip_cptac ...\n"
    )
    return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download CPTAC PDC000111 proteomics')
    parser.add_argument('--output_dir', default='./raw_downloads',
                        help='Output directory (default: ./raw_downloads)')
    args = parser.parse_args()

    download_from_pdc(args.output_dir)
