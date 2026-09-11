#!/usr/bin/env bash
# Donor arms under the donor-distinct within-lineage schedule, for all three schedule seeds.
# Seed 11 is primary; 22 and 33 are the donor-schedule sensitivity. `own` and `mean` do not depend
# on the schedule and are copied in rather than recomputed.
set -eu
cd "$HOME/metagnn/external/deepmeta_audit"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
PY="$HOME/metagnn/venv/bin/python"
ARMS="within cross within_expr within_graph"
for S in 11 22 33; do
  echo "=== seed $S: build_arms ==="
  $PY build_arms.py --seed "$S" --arms $ARMS --batch_size 20
  echo "=== seed $S: run_arms ==="
  $PY run_arms.py --seed "$S" --arms $ARMS --batch_size 4 --cores 8 --device cuda
  mkdir -p "preds/seed$S"
  for A in own mean; do
    [ -f "preds/seed$S/preds_$A.csv" ] || cp "preds/seed11/preds_$A.csv" "preds/seed$S/preds_$A.csv"
  done
  echo "=== seed $S: metrics ==="
  if [ "$S" = "11" ]; then
    $PY metrics.py --seed 11 --rf_jobs 24
  else
    $PY metrics.py --seed "$S" --arms own mean within cross within_expr within_graph \
        --rf_jobs 24 --out "audit_results_seed$S.json"
  fi
  rm -rf "work/seed$S"
done
echo "=== ALL DONE ==="
