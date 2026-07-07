#!/usr/bin/env python3
"""
Download the CORRECT CPTAC proteomics from PDC000111 (TCGA Retrospective).
Run this on your Mac (not in VM — PDC is blocked there).

Usage:
    cd MetaGNN-CRC/code
    python download_cptac_correct.py

This downloads from PDC000111 (study_id: b998098f-57b8-11e8-b07a-00a098d917f8)
which has TCGA barcodes matching our 690 TCGA-COAD/READ patients.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = './raw_downloads'
STUDY_ID = 'b998098f-57b8-11e8-b07a-00a098d917f8'  # PDC000111
PDC_URL = 'https://pdc.cancer.gov/graphql'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prot_tsv = os.path.join(OUTPUT_DIR, 'cptac_colon_proteomics.tsv')
    clin_tsv = os.path.join(OUTPUT_DIR, 'cptac_colon_clinical.tsv')

    # ── Try quantDataMatrix with different data types ────────────────────
    for data_type in ['log2_ratio', 'unshared_log2_ratio', 'spectral_count',
                      'precursor_area', 'log2_intensity']:
        logger.info(f"Trying quantDataMatrix(data_type='{data_type}')...")

        query = """
        {{
            quantDataMatrix(
                study_id: "{sid}"
                data_type: "{dtype}"
                acceptDUA: true
            )
        }}
        """.format(sid=STUDY_ID, dtype=data_type)

        try:
            resp = requests.post(PDC_URL, json={"query": query}, timeout=300)
            resp.raise_for_status()
            result = resp.json()

            # Check for errors in response
            if 'errors' in result:
                logger.warning(f"  GraphQL errors: {result['errors']}")
                continue

            matrix = result.get('data', {}).get('quantDataMatrix', [])

            if matrix and len(matrix) > 1:
                header = matrix[0]
                rows = matrix[1:]
                logger.info(f"  Got {len(rows)} genes × {len(header)-1} samples")

                # Check if sample IDs are TCGA barcodes
                sample_ids = header[1:]
                tcga_count = sum(1 for s in sample_ids if str(s).startswith('TCGA-'))
                logger.info(f"  TCGA barcodes: {tcga_count}/{len(sample_ids)}")
                logger.info(f"  First 5 IDs: {sample_ids[:5]}")

                # Build DataFrame
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

                prot_df = pd.DataFrame(values, index=gene_names, columns=sample_ids)
                prot_df.index.name = 'Gene'
                prot_df = prot_df.dropna(how='all', axis=0).dropna(how='all', axis=1)

                logger.info(f"  Final matrix: {prot_df.shape[0]} genes × {prot_df.shape[1]} samples")
                logger.info(f"  Non-NaN: {prot_df.notna().mean().mean():.1%}")

                prot_df.to_csv(prot_tsv, sep='\t')
                logger.info(f"  Saved: {prot_tsv}")

                # Create clinical mapping from biospecimen if available
                bio_path = os.path.join(OUTPUT_DIR,
                    'PDC_study_biospecimen_03092026_220837.csv')
                if not os.path.exists(bio_path):
                    # Try uploads folder
                    for d in ['.', '..', '../uploads']:
                        candidate = os.path.join(d, 'PDC_study_biospecimen_03092026_220837.csv')
                        if os.path.exists(candidate):
                            bio_path = candidate
                            break

                if os.path.exists(bio_path):
                    bio_df = pd.read_csv(bio_path)
                    clin_out = bio_df[['Aliquot Submitter ID', 'Case Submitter ID',
                                       'Sample Type']].copy()
                    clin_out.columns = ['aliquot_id', 'tcga_barcode', 'sample_type']
                    clin_out.to_csv(clin_tsv, sep='\t', index=False)
                    logger.info(f"  Clinical mapping saved: {clin_tsv}")
                else:
                    # Create from column names
                    clin_data = []
                    for sid in prot_df.columns:
                        tcga = sid.split('-01A')[0] if '-01A' in sid else sid[:12]
                        clin_data.append({'aliquot_id': sid, 'tcga_barcode': tcga})
                    pd.DataFrame(clin_data).to_csv(clin_tsv, sep='\t', index=False)
                    logger.info(f"  Clinical mapping (auto-generated): {clin_tsv}")

                logger.info("\nDone! Now re-run the pipeline:")
                logger.info("  python 00_download_and_generate_690.py \\")
                logger.info("      --cptac_tsv ./raw_downloads/cptac_colon_proteomics.tsv \\")
                logger.info("      --cptac_clinical ./raw_downloads/cptac_colon_clinical.tsv \\")
                logger.info("      ...")
                return

            else:
                logger.warning(f"  Empty matrix for data_type='{data_type}'")

        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout (try again or increase timeout)")
        except Exception as e:
            logger.warning(f"  Error: {e}")

    # ── If all quant types failed, list available files ───────────────────
    logger.info("\nAll quantDataMatrix attempts failed. Checking available files...")

    file_query = """
    {{
        filesPerStudy(study_id: "{sid}" acceptDUA: true) {{
            file_id
            file_name
            file_type
            data_category
            file_size
        }}
    }}
    """.format(sid=STUDY_ID)

    try:
        resp = requests.post(PDC_URL, json={"query": file_query}, timeout=120)
        files = resp.json().get('data', {}).get('filesPerStudy', [])

        categories = {}
        for f in files:
            cat = f.get('data_category', 'unknown')
            categories.setdefault(cat, []).append(f)

        for cat, flist in categories.items():
            logger.info(f"  {cat}: {len(flist)} files")
            for ff in flist[:3]:
                logger.info(f"    {ff.get('file_name')} ({ff.get('file_size',0)/1024:.0f} KB)")

    except Exception as e:
        logger.error(f"File listing failed: {e}")

    logger.error(
        "\nCould not download quantitative data from PDC000111.\n"
        "You can proceed without CPTAC — add --skip_cptac to the pipeline command.\n"
        "Only 90/690 patients have CPTAC data; the rest are zero-filled anyway."
    )


if __name__ == '__main__':
    main()
