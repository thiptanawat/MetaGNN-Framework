#!/bin/bash
# Prioritized queue for the Reaction-Scoring Audit robustness runs on GPU 0 (the driver skips a cell whose result
# file already exists, so finished cells are never rerun). Order: the inner-validation-reaction early
# stopping check on the remaining patient folds (unsubstituted and cohort-mean arms, so that the check
# covers the full 15-cell grid), then the remaining second-seed and rank-normalized cells, the remaining
# tissue-matched-label cells, and finally the entries that are already complete (kept so the list
# documents the full design).
cd ~/metagnn/work
L=~/metagnn/logs; MAXJ=${MAXJ:-5}
export CUDA_VISIBLE_DEVICES=0
mkdir -p ~/metagnn/out/ht29 ~/metagnn/out/seedrep ~/metagnn/out/rankgnn ~/metagnn/out/permute ~/metagnn/out/ivr
H="python3 double_holdout_ctrl.py --data crc_624_ht29 --out ~/metagnn/out/ht29 --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0"
S="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/seedrep --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0 --train_seed 1"
K="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/rankgnn --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0 --expr_transform rank"
P="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/permute --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0 --feature_mode permute"
V="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/ivr --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --inner_val_rxn 0.2"
V0="$V --only_pfold 0"
Q=(
"$V --only_pfold 1 --only_rfold 0 --feature_mode real"
"$V --only_pfold 1 --only_rfold 0 --feature_mode mean"
"$V --only_pfold 1 --only_rfold 1 --feature_mode real"
"$V --only_pfold 1 --only_rfold 1 --feature_mode mean"
"$V --only_pfold 1 --only_rfold 2 --feature_mode real"
"$V --only_pfold 1 --only_rfold 2 --feature_mode mean"
"$V --only_pfold 2 --only_rfold 0 --feature_mode real"
"$V --only_pfold 2 --only_rfold 0 --feature_mode mean"
"$V --only_pfold 2 --only_rfold 1 --feature_mode real"
"$V --only_pfold 2 --only_rfold 1 --feature_mode mean"
"$V --only_pfold 2 --only_rfold 2 --feature_mode real"
"$V --only_pfold 2 --only_rfold 2 --feature_mode mean"
"$V --only_pfold 3 --only_rfold 0 --feature_mode real"
"$V --only_pfold 3 --only_rfold 0 --feature_mode mean"
"$V --only_pfold 3 --only_rfold 1 --feature_mode real"
"$V --only_pfold 3 --only_rfold 1 --feature_mode mean"
"$V --only_pfold 3 --only_rfold 2 --feature_mode real"
"$V --only_pfold 3 --only_rfold 2 --feature_mode mean"
"$V --only_pfold 4 --only_rfold 0 --feature_mode real"
"$V --only_pfold 4 --only_rfold 0 --feature_mode mean"
"$V --only_pfold 4 --only_rfold 1 --feature_mode real"
"$V --only_pfold 4 --only_rfold 1 --feature_mode mean"
"$V --only_pfold 4 --only_rfold 2 --feature_mode real"
"$V --only_pfold 4 --only_rfold 2 --feature_mode mean"
"$S --only_rfold 1 --feature_mode real"
"$S --only_rfold 1 --feature_mode mean"
"$S --only_rfold 2 --feature_mode real"
"$S --only_rfold 2 --feature_mode mean"
"$K --only_rfold 1 --feature_mode real"
"$K --only_rfold 1 --feature_mode mean"
"$K --only_rfold 2 --feature_mode real"
"$K --only_rfold 2 --feature_mode mean"
"$H --only_rfold 1 --feature_mode zero"
"$H --only_rfold 2 --feature_mode zero"
"$V0 --only_rfold 0 --feature_mode real"
"$V0 --only_rfold 0 --feature_mode mean"
"$V0 --only_rfold 1 --feature_mode real"
"$V0 --only_rfold 1 --feature_mode mean"
"$V0 --only_rfold 2 --feature_mode real"
"$V0 --only_rfold 2 --feature_mode mean"
"$P --only_rfold 0"
"$P --only_rfold 1"
"$P --only_rfold 2"
"$S --only_rfold 0 --feature_mode real"
"$S --only_rfold 0 --feature_mode mean"
"$K --only_rfold 0 --feature_mode real"
"$K --only_rfold 0 --feature_mode mean"
"$H --only_rfold 2 --feature_mode real"
"$H --only_rfold 2 --feature_mode mean"
)
i=0
while [ $i -lt ${#Q[@]} ]; do
  n=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -c .)
  if [ "$n" -lt "$MAXJ" ]; then
    echo "$(date -Is) queue6 launching [$i]: ${Q[$i]}" >> $L/gpu_queue.log
    nohup bash -c "${Q[$i]}" > $L/queue6_job_$i.log 2>&1 &
    i=$((i+1)); sleep 150
  else
    sleep 60
  fi
done
echo "$(date -Is) queue6 drained" >> $L/gpu_queue.log
