#!/bin/bash
# FIRST_COHORT_SEED_PLAN_2026-11-15.md: the graph attention scorer on the first cohort at a second training
# seed, on the full grid. Everything is gpu_queue8.sh (the family-disjoint split under the matched selection
# rule), with --train_seed 2025 (the folds stay on the default seed) and the own-expression and cohort-mean
# arms only: 30 jobs. Not run when this file was fixed.
#
#   GPU=0 bash gpu_queue_cohort1_seed2025.sh                      # jobs 0 to 29, ascending
#   GPU=1 ORDER=down bash gpu_queue_cohort1_seed2025.sh           # the same list from its end, on a second accelerator
#
# A job is skipped when its result file exists or when a job with the same arguments is running, so two
# copies of the queue can work through one list from its two ends, and the queue can be stopped and started
# again. It starts a job only while its accelerator has at least FREE_MB of memory free and fewer than MAXJ
# of its own jobs are running there. --out is given before --data, so that the queues of the second cohort,
# which count their own jobs by their command line, do not count the jobs issued here.
cd ~/metagnn/work
DATA=${DATA:-crc_624}; OUT=${OUT:-$HOME/metagnn/out/cohort1_fam_ts2025}
L=~/metagnn/logs; MAXJ=${MAXJ:-4}; GPU=${GPU:-0}; FREE_MB=${FREE_MB:-12000}; ORDER=${ORDER:-up}; SEED=2025
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT" "$L"
F="python3 double_holdout_ctrl.py --out $OUT --data $DATA --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --inner_val_rxn 0.2 --family_map families_union.json --family_tag fam --train_seed $SEED"
Q=(); T=(); N=()
for pf in 0 1 2 3 4; do for rf in 0 1 2; do for mode in real mean; do
  Q+=("$F --only_pfold $pf --only_rfold $rf --feature_mode $mode")
  T+=("--out $OUT --data $DATA .* --train_seed $SEED --only_pfold $pf --only_rfold $rf --feature_mode $mode")
  if [ $mode = real ]; then N+=("gnn_B_mb0.0_p${pf}_r${rf}_ts${SEED}_ivr_fam.json"); else N+=("gnn_B_mb0.0_p${pf}_r${rf}_${mode}_ts${SEED}_ivr_fam.json"); fi
done; done; done
LOG=$L/gpu_queue_cohort1_ts${SEED}_gpu$GPU.log
if [ "$ORDER" = down ]; then i=$((${#Q[@]}-1)); step=-1; else i=0; step=1; fi
echo "$(date -Is) cohort1 seed $SEED queue: ${#Q[@]} jobs, order $ORDER, on GPU $GPU" >> $LOG
while [ $i -ge 0 ] && [ $i -lt ${#Q[@]} ]; do
  if [ -f "$OUT/${N[$i]}" ]; then echo "$(date -Is) [$i] result file exists, skipped" >> $LOG; i=$((i+step)); continue; fi
  if pgrep -f -- "${T[$i]}\$" > /dev/null; then echo "$(date -Is) [$i] already running, skipped" >> $LOG; i=$((i+step)); continue; fi
  mine=0   # the jobs of this list that run on this accelerator (two copies of the queue may share the list)
  for p in $(pgrep -f "double_holdout_ctrl.py --out $OUT --data"); do
    { tr '\0' '\n' < /proc/$p/environ; } 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPU" && mine=$((mine+1))
  done
  free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "$mine" -lt "$MAXJ" ] && [ "${free:-0}" -ge "$FREE_MB" ]; then
    echo "$(date -Is) cohort1 seed $SEED queue launching [$i] on GPU $GPU: ${Q[$i]}" >> $LOG
    nohup bash -c "${Q[$i]}" > $L/cohort1_ts${SEED}_job_$i.log 2>&1 &
    i=$((i+step)); sleep 150
  else
    sleep 60
  fi
done
echo "$(date -Is) cohort1 seed $SEED queue on GPU $GPU reached the end of its list" >> $LOG
