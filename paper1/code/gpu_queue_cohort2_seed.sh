#!/bin/bash
# Outside SECOND_COHORT_PLAN_2026-11-13.md: the training seed on the second cohort, at one cell.
# Section 6 of the plan fixes the default training seed. This check repeats, on the second cohort,
# the one made on the first (gpu_queue8.sh): cell p0r0 of the family-disjoint split under the
# matched selection rule, own-expression and cohort-mean arms, at the training seeds 2025 and 2026,
# with the folds fixed: 4 jobs on one accelerator. Everything else is as in gpu_queue_cohort2.sh.
# It was run after the 30 runs of Section 6 had ended and their outcome was known, and it enters
# none of the plan's quantities (code/second_cohort_seed.py reads it).
#
#   GPU=1 DATA=~/metagnn/repl/cohort2/brca OUT=~/metagnn/out/cohort2_fam_seed bash gpu_queue_cohort2_seed.sh
cd ~/metagnn/work
DATA=${DATA:-$HOME/metagnn/repl/cohort2/brca}; OUT=${OUT:-$HOME/metagnn/out/cohort2_fam_seed}
L=~/metagnn/logs; GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT" "$L"
F="python3 double_holdout_ctrl.py --data $DATA --out $OUT --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --inner_val_rxn 0.2 --family_map families_union.json --family_tag fam --only_pfold 0 --only_rfold 0"
i=0
for seed in 2025 2026; do
  for mode in real mean; do
    echo "$(date -Is) cohort2 seed check launching [$i]: $F --feature_mode $mode --train_seed $seed" >> $L/gpu_queue_cohort2_seed.log
    nohup bash -c "$F --feature_mode $mode --train_seed $seed" > $L/cohort2_seed_job_$i.log 2>&1 &
    i=$((i+1)); sleep 150
  done
done
wait
echo "$(date -Is) cohort2 seed check done" >> $L/gpu_queue_cohort2_seed.log
