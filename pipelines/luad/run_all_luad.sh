#!/bin/bash
# MetaGNN-LUAD: Full pipeline
# Run from MetaGNN-LUAD/ directory on H100
set -e

export CRC_DATA_DIR=${CRC_DATA_DIR:-~/code/MetaGNN_H100_Package/data/processed_690/}

echo "=== Step 1: Download TCGA-LUAD ==="
python 01_download_tcga_luad.py --output_dir ./data_luad/raw

echo "=== Step 2: Preprocess for MetaGNN ==="
python 02_preprocess_luad_for_metagnn.py \
    --tpm_matrix ./data_luad/tcga_luad_tpm_log2.tsv \
    --crc_processed $CRC_DATA_DIR \
    --output_dir ./data_luad/processed/

echo "=== Step 3: Train MetaGNN on LUAD ==="
python 03_train_metagnn_luad.py \
    --data_dir ./data_luad/processed/ \
    --config full_cohort \
    --output_dir ./results_luad/full_cohort/

echo "=== DONE ==="
echo "Results: results_luad/full_cohort/results_luad.json"
