#!/bin/bash
# Section 7 of SECOND_COHORT_PLAN_2026-11-13.md: the graph attention scorer on the second cohort at a
# second training seed, on the full grid. Not run when this file was committed, except for cell p0r0,
# whose two runs at this seed were made in the check of 14 November (gpu_queue_cohort2_seed.sh) and
# are reused: copy their result files into OUT before starting, and the driver skips them.
# Everything else is gpu_queue_cohort2.sh: the family-disjoint split under the matched selection rule,
# the own-expression arm and the cohort-mean arm on every cell of the 5 x 3 grid, with
# --train_seed 2025 (the folds stay on the default seed): 30 jobs, 28 of them new.
#
#   GPU=1 DATA=~/metagnn/repl/cohort2/brca OUT=~/metagnn/out/cohort2_fam_ts2025 bash gpu_queue_cohort2_seed2025.sh
#
# The driver skips a cell whose result file already exists, so the queue can be stopped and started
# again. It runs on one accelerator (GPU, default 0) and starts a job only while that accelerator has
# at least FREE_MB of memory free.
cd ~/metagnn/work
DATA=${DATA:-$HOME/metagnn/repl/cohort2/brca}; OUT=${OUT:-$HOME/metagnn/out/cohort2_fam_ts2025}
L=~/metagnn/logs; MAXJ=${MAXJ:-4}; GPU=${GPU:-0}; FREE_MB=${FREE_MB:-12000}; SEED=2025
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT" "$L"
F="python3 double_holdout_ctrl.py --data $DATA --out $OUT --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --inner_val_rxn 0.2 --family_map families_union.json --family_tag fam --train_seed $SEED"
Q=()
for pf in 0 1 2 3 4; do
  for rf in 0 1 2; do
    for mode in real mean; do
      Q+=("$F --only_pfold $pf --only_rfold $rf --feature_mode $mode")
    done
  done
done
echo "$(date -Is) cohort2 seed $SEED queue starting with ${#Q[@]} jobs on GPU $GPU" >> $L/gpu_queue_cohort2_ts$SEED.log
i=0
while [ $i -lt ${#Q[@]} ]; do
  mine=$(pgrep -f "double_holdout_ctrl.py --data $DATA" | wc -l)
  free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "$mine" -lt "$MAXJ" ] && [ "${free:-0}" -ge "$FREE_MB" ]; then
    echo "$(date -Is) cohort2 seed $SEED queue launching [$i]: ${Q[$i]}" >> $L/gpu_queue_cohort2_ts$SEED.log
    nohup bash -c "${Q[$i]}" > $L/cohort2_ts${SEED}_job_$i.log 2>&1 &
    i=$((i+1)); sleep 150
  else
    sleep 60
  fi
done
echo "$(date -Is) cohort2 seed $SEED queue drained" >> $L/gpu_queue_cohort2_ts$SEED.log
