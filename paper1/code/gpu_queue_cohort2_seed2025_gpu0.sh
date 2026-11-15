#!/bin/bash
# Section 7 of SECOND_COHORT_PLAN_2026-11-13.md: the last jobs of the list of gpu_queue_cohort2_seed2025.sh,
# issued on a second accelerator of the same host to shorten the wall-clock time. Written on 15 November
# 2026 while the queue of gpu_queue_cohort2_seed2025.sh was running on the first accelerator; a departure
# from the plan's launcher, recorded with the results. The job list is that launcher's, built by the same
# loops. The jobs LAST down to FIRST of that list (numbered from 0) are issued here in descending order, so
# that the two queues work toward each other. A job is skipped if its result file exists or if it is
# already running. The arguments of a job are those of the published launcher; --out is given before
# --data, so that the other queue, which counts its own jobs by their command line, does not count the
# jobs issued here.
#
#   GPU=0 FIRST=22 LAST=29 DATA=~/metagnn/repl/cohort2/brca OUT=~/metagnn/out/cohort2_fam_ts2025 bash gpu_queue_cohort2_seed2025_gpu0.sh
cd ~/metagnn/work
DATA=${DATA:-$HOME/metagnn/repl/cohort2/brca}; OUT=${OUT:-$HOME/metagnn/out/cohort2_fam_ts2025}
L=~/metagnn/logs; MAXJ=${MAXJ:-4}; GPU=${GPU:-0}; FREE_MB=${FREE_MB:-12000}; SEED=2025
FIRST=${FIRST:?give the first job index}; LAST=${LAST:?give the last job index}
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT" "$L"
F="python3 double_holdout_ctrl.py --out $OUT --data $DATA --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --inner_val_rxn 0.2 --family_map families_union.json --family_tag fam --train_seed $SEED"
Q=(); T=(); N=()
for pf in 0 1 2 3 4; do
  for rf in 0 1 2; do
    for mode in real mean; do
      Q+=("$F --only_pfold $pf --only_rfold $rf --feature_mode $mode")
      T+=("--train_seed $SEED --only_pfold $pf --only_rfold $rf --feature_mode $mode")
      if [ $mode = real ]; then N+=("gnn_B_mb0.0_p${pf}_r${rf}_ts${SEED}_ivr_fam.json"); else N+=("gnn_B_mb0.0_p${pf}_r${rf}_${mode}_ts${SEED}_ivr_fam.json"); fi
    done
  done
done
LOG=$L/gpu_queue_cohort2_ts${SEED}_gpu$GPU.log
echo "$(date -Is) cohort2 seed $SEED second-accelerator queue: jobs $LAST down to $FIRST of ${#Q[@]} on GPU $GPU" >> $LOG
i=$LAST
while [ $i -ge $FIRST ]; do
  if [ -f "$OUT/${N[$i]}" ]; then echo "$(date -Is) [$i] result file exists, skipped" >> $LOG; i=$((i-1)); continue; fi
  if pgrep -f -- "${T[$i]}\$" > /dev/null; then echo "$(date -Is) [$i] already running, skipped" >> $LOG; i=$((i-1)); continue; fi
  mine=$(pgrep -f "double_holdout_ctrl.py --out $OUT --data" | wc -l)
  free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "$mine" -lt "$MAXJ" ] && [ "${free:-0}" -ge "$FREE_MB" ]; then
    echo "$(date -Is) cohort2 seed $SEED second-accelerator queue launching [$i]: ${Q[$i]}" >> $LOG
    nohup bash -c "${Q[$i]}" > $L/cohort2_ts${SEED}_job_$i.log 2>&1 &
    i=$((i-1)); sleep 150
  else
    sleep 60
  fi
done
echo "$(date -Is) cohort2 seed $SEED second-accelerator queue drained" >> $LOG
