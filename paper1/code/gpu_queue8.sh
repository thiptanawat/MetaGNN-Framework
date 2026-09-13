#!/bin/bash
# Third queue for the Reaction-Scoring Audit runs on GPU 0: the reaction-family-disjoint evaluation. The reaction
# folds and the inner-validation partition are drawn family by family from the union family map
# (build_families.py, family_folds.py), under the matched selection rule (inner-validation reactions
# withheld from the loss). Order: the paired pilot on cell (p0, f0), unsubstituted and cohort-mean arms
# under the default training seed and two further seeds, so that the seed dispersion of the contrast
# is known before the grid; then the five arms (own, cohort mean, zeroed, indicator, wrong patient) on
# every cell of the 5 x 3 grid. The driver skips a cell whose result file already exists.
cd ~/metagnn/work
L=~/metagnn/logs; MAXJ=${MAXJ:-5}
export CUDA_VISIBLE_DEVICES=0
mkdir -p ~/metagnn/out/fam
F="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/fam --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --inner_val_rxn 0.2 --family_map families_union.json --family_tag fam"
Q=(
"$F --only_pfold 0 --only_rfold 0 --feature_mode real"
"$F --only_pfold 0 --only_rfold 0 --feature_mode mean"
"$F --only_pfold 0 --only_rfold 0 --feature_mode real --train_seed 2025"
"$F --only_pfold 0 --only_rfold 0 --feature_mode mean --train_seed 2025"
"$F --only_pfold 0 --only_rfold 0 --feature_mode real --train_seed 2026"
"$F --only_pfold 0 --only_rfold 0 --feature_mode mean --train_seed 2026"
)
for pf in 0 1 2 3 4; do
  for rf in 0 1 2; do
    for mode in real mean zero indicator permute; do
      if [ "$pf$rf" = "00" ] && { [ "$mode" = "real" ] || [ "$mode" = "mean" ]; }; then continue; fi
      Q+=("$F --only_pfold $pf --only_rfold $rf --feature_mode $mode")
    done
  done
done
echo "$(date -Is) queue8 starting with ${#Q[@]} jobs" >> $L/gpu_queue.log
i=0
while [ $i -lt ${#Q[@]} ]; do
  n=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -c .)
  if [ "$n" -lt "$MAXJ" ]; then
    echo "$(date -Is) queue8 launching [$i]: ${Q[$i]}" >> $L/gpu_queue.log
    nohup bash -c "${Q[$i]}" > $L/queue8_job_$i.log 2>&1 &
    i=$((i+1)); sleep 150
  else
    sleep 60
  fi
done
echo "$(date -Is) queue8 drained" >> $L/gpu_queue.log
