#!/bin/bash
# Section 8 of SECOND_COHORT_PLAN_2026-11-13.md: the two arms of the graph attention scorer on the second
# cohort at equal training length. Not run when this file was committed.
# The driver is double_holdout_len.py: no early stopping and no checkpoint selection; every cell is trained
# for 13 epochs and scored after 1, 7 and 13 epochs. Everything else is gpu_queue_cohort2.sh and
# gpu_queue_cohort2_seed2025.sh: the family-disjoint split with the inner validation reactions, the
# own-expression arm and the cohort-mean arm on every cell of the 5 x 3 grid. SEED picks the training
# seed: 2024, the seed of Section 6 (the driver's default, no --train_seed), or 2025, the seed of
# Section 7 (--train_seed 2025). 30 jobs per seed, 60 in all.
#
#   SEED=2024 GPU=1 bash gpu_queue_cohort2_len.sh
#   SEED=2025 GPU=0 bash gpu_queue_cohort2_len.sh
#
# One call issues the 30 jobs of one seed on one accelerator; the two calls may run at the same time on
# two accelerators or one after the other on one. The driver skips a cell whose result files exist, so a
# queue can be stopped and started again. A job is started only while the accelerator has at least
# FREE_MB of memory free and fewer than MAXJ jobs of this queue are running.
cd ~/metagnn/work
SEED=${SEED:?give the training seed, 2024 or 2025}
DATA=${DATA:-$HOME/metagnn/repl/cohort2/brca}; OUT=${OUT:-$HOME/metagnn/out/cohort2_fam_len_ts$SEED}
L=~/metagnn/logs; MAXJ=${MAXJ:-4}; GPU=${GPU:-0}; FREE_MB=${FREE_MB:-12000}; EPOCHS=1,7,13
case $SEED in
  2024) TS="" ;;
  2025) TS=" --train_seed 2025" ;;
  *) echo "SEED must be 2024 or 2025"; exit 1 ;;
esac
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT" "$L"
F="python3 double_holdout_len.py --data $DATA --out $OUT --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --inner_val_rxn 0.2 --family_map families_union.json --family_tag fam$TS --score_epochs $EPOCHS"
Q=()
for pf in 0 1 2 3 4; do
  for rf in 0 1 2; do
    for mode in real mean; do
      Q+=("$F --only_pfold $pf --only_rfold $rf --feature_mode $mode")
    done
  done
done
LOG=$L/gpu_queue_cohort2_len_ts$SEED.log
echo "$(date -Is) cohort2 equal-length queue, seed $SEED, starting with ${#Q[@]} jobs on GPU $GPU" >> $LOG
i=0
while [ $i -lt ${#Q[@]} ]; do
  mine=$(pgrep -f "double_holdout_len.py --data $DATA --out $OUT " | wc -l)
  free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "$mine" -lt "$MAXJ" ] && [ "${free:-0}" -ge "$FREE_MB" ]; then
    echo "$(date -Is) cohort2 equal-length queue, seed $SEED, launching [$i]: ${Q[$i]}" >> $LOG
    nohup bash -c "${Q[$i]}" > $L/cohort2_len_ts${SEED}_job_$i.log 2>&1 &
    i=$((i+1)); sleep 150
  else
    sleep 60
  fi
done
echo "$(date -Is) cohort2 equal-length queue, seed $SEED, drained" >> $LOG
