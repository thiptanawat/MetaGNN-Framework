#!/bin/bash
# The frozen percentile-only interface on the main backbone (Qwen3.8-27B, remote endpoint):
# development cohort first, then the two external cohorts, in the three configurations. Every
# collection is interleaved, fully archived and restart-safe (a rerun skips answered calls).
cd "$(dirname "$0")/.."
R=results/msi2; mkdir -p $R
for cfg in zero_shot evidence tool; do
  python3 code/msi_probe2.py --cohort tcga --endpoint nonmsih --config $cfg --donor_seed 11 --out $R/tcga_qwen_${cfg}_s11 --workers 10
  python3 code/msi_probe2.py --cohort gse39582 --config $cfg --donor_seed 11 --out $R/gse39582_qwen_${cfg}_s11 --workers 10
  python3 code/msi_probe2.py --cohort gse13294 --config $cfg --donor_seed 11 --out $R/gse13294_qwen_${cfg}_s11 --workers 10
done
echo "$(date -Is) qwen msi2 collections done"
