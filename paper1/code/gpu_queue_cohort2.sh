#!/bin/bash
# Planned extension of SECOND_COHORT_PLAN_2026-11-13.md (Section 6): the graph attention scorer on the
# second cohort. Not run when this file was committed. The family-disjoint split under the matched
# selection rule (inner-validation reactions withheld from the loss), the default training seed, the
# own-expression arm and the cohort-mean arm on every cell of the 5 x 3 grid: 30 jobs.
#
#   DATA=~/metagnn/repl/cohort2/brca OUT=~/metagnn/out/cohort2_fam bash gpu_queue_cohort2.sh
#
# The driver skips a cell whose result file already exists, so the queue can be stopped and started
# again. It runs on one accelerator (GPU, default 0) and starts a job only while that accelerator has
# at least FREE_MB of memory free, so it waits instead of competing with another process.
cd ~/metagnn/work
DATA=${DATA:-$HOME/metagnn/repl/cohort2/brca}; OUT=${OUT:-$HOME/metagnn/out/cohort2_fam}
L=~/metagnn/logs; MAXJ=${MAXJ:-4}; GPU=${GPU:-0}; FREE_MB=${FREE_MB:-12000}
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT" "$L"
F="python3 double_holdout_ctrl.py --data $DATA --out $OUT --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --inner_val_rxn 0.2 --family_map families_union.json --family_tag fam"
Q=()
for pf in 0 1 2 3 4; do
  for rf in 0 1 2; do
    for mode in real mean; do
      Q+=("$F --only_pfold $pf --only_rfold $rf --feature_mode $mode")
    done
  done
done
echo "$(date -Is) cohort2 queue starting with ${#Q[@]} jobs on GPU $GPU" >> $L/gpu_queue_cohort2.log
i=0
while [ $i -lt ${#Q[@]} ]; do
  mine=$(pgrep -f "double_holdout_ctrl.py --data $DATA" | wc -l)
  free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "$mine" -lt "$MAXJ" ] && [ "${free:-0}" -ge "$FREE_MB" ]; then
    echo "$(date -Is) cohort2 queue launching [$i]: ${Q[$i]}" >> $L/gpu_queue_cohort2.log
    nohup bash -c "${Q[$i]}" > $L/cohort2_job_$i.log 2>&1 &
    i=$((i+1)); sleep 150
  else
    sleep 60
  fi
done
echo "$(date -Is) cohort2 queue drained" >> $L/gpu_queue_cohort2.log
