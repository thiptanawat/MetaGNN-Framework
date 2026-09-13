#!/bin/bash
# Second queue for the Reaction-Scoring Audit robustness runs on GPU 0. Order: the synthetic patient-varying
# positive control for the audit (a cohort-median label on all or half of the expression-bearing
# reactions, unsubstituted and cohort-mean arms, patient fold 0, first under inner-validation-reaction
# early stopping and then under the main grid's selection rule); then the inner-validation-reaction
# check on the zeroed, indicator and wrong-patient arms over the full 15-cell grid, so that the whole
# substitution ladder exists under the selection criterion that asks the test question (the
# unsubstituted and cohort-mean arms are in gpu_queue6.sh). The driver skips a cell whose result file
# already exists. Started by run_queue7_after6.sh once the first queue drains.
cd ~/metagnn/work
L=~/metagnn/logs; MAXJ=${MAXJ:-5}
export CUDA_VISIBLE_DEVICES=0
mkdir -p ~/metagnn/out/ht29 ~/metagnn/out/seedrep ~/metagnn/out/rankgnn ~/metagnn/out/permute ~/metagnn/out/ivr ~/metagnn/out/synth
H="python3 double_holdout_ctrl.py --data crc_624_ht29 --out ~/metagnn/out/ht29 --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0"
S="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/seedrep --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0 --train_seed 1"
K="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/rankgnn --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0 --expr_transform rank"
P="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/permute --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0 --feature_mode permute"
V="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/ivr --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --inner_val_rxn 0.2"
V0="$V --only_pfold 0"
Y="python3 double_holdout_ctrl.py --data crc_624 --out ~/metagnn/out/synth --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0 --synth_target cohortmedian"
Q=(
"$Y --synth_frac 1.0 --inner_val_rxn 0.2 --only_rfold 0 --feature_mode real"
"$Y --synth_frac 1.0 --inner_val_rxn 0.2 --only_rfold 0 --feature_mode mean"
"$Y --synth_frac 1.0 --inner_val_rxn 0.2 --only_rfold 1 --feature_mode real"
"$Y --synth_frac 1.0 --inner_val_rxn 0.2 --only_rfold 1 --feature_mode mean"
"$Y --synth_frac 1.0 --inner_val_rxn 0.2 --only_rfold 2 --feature_mode real"
"$Y --synth_frac 1.0 --inner_val_rxn 0.2 --only_rfold 2 --feature_mode mean"
"$Y --synth_frac 0.5 --inner_val_rxn 0.2 --only_rfold 0 --feature_mode real"
"$Y --synth_frac 0.5 --inner_val_rxn 0.2 --only_rfold 0 --feature_mode mean"
"$Y --synth_frac 0.5 --inner_val_rxn 0.2 --only_rfold 1 --feature_mode real"
"$Y --synth_frac 0.5 --inner_val_rxn 0.2 --only_rfold 1 --feature_mode mean"
"$Y --synth_frac 0.5 --inner_val_rxn 0.2 --only_rfold 2 --feature_mode real"
"$Y --synth_frac 0.5 --inner_val_rxn 0.2 --only_rfold 2 --feature_mode mean"
"$Y --synth_frac 1.0 --only_rfold 0 --feature_mode real"
"$Y --synth_frac 1.0 --only_rfold 0 --feature_mode mean"
"$Y --synth_frac 1.0 --only_rfold 1 --feature_mode real"
"$Y --synth_frac 1.0 --only_rfold 1 --feature_mode mean"
"$Y --synth_frac 1.0 --only_rfold 2 --feature_mode real"
"$Y --synth_frac 1.0 --only_rfold 2 --feature_mode mean"
"$Y --synth_frac 0.5 --only_rfold 0 --feature_mode real"
"$Y --synth_frac 0.5 --only_rfold 0 --feature_mode mean"
"$Y --synth_frac 0.5 --only_rfold 1 --feature_mode real"
"$Y --synth_frac 0.5 --only_rfold 1 --feature_mode mean"
"$Y --synth_frac 0.5 --only_rfold 2 --feature_mode real"
"$Y --synth_frac 0.5 --only_rfold 2 --feature_mode mean"
"$V --only_pfold 0 --only_rfold 0 --feature_mode zero"
"$V --only_pfold 0 --only_rfold 1 --feature_mode zero"
"$V --only_pfold 0 --only_rfold 2 --feature_mode zero"
"$V --only_pfold 1 --only_rfold 0 --feature_mode zero"
"$V --only_pfold 1 --only_rfold 1 --feature_mode zero"
"$V --only_pfold 1 --only_rfold 2 --feature_mode zero"
"$V --only_pfold 2 --only_rfold 0 --feature_mode zero"
"$V --only_pfold 2 --only_rfold 1 --feature_mode zero"
"$V --only_pfold 2 --only_rfold 2 --feature_mode zero"
"$V --only_pfold 3 --only_rfold 0 --feature_mode zero"
"$V --only_pfold 3 --only_rfold 1 --feature_mode zero"
"$V --only_pfold 3 --only_rfold 2 --feature_mode zero"
"$V --only_pfold 4 --only_rfold 0 --feature_mode zero"
"$V --only_pfold 4 --only_rfold 1 --feature_mode zero"
"$V --only_pfold 4 --only_rfold 2 --feature_mode zero"
"$V --only_pfold 0 --only_rfold 0 --feature_mode indicator"
"$V --only_pfold 0 --only_rfold 1 --feature_mode indicator"
"$V --only_pfold 0 --only_rfold 2 --feature_mode indicator"
"$V --only_pfold 1 --only_rfold 0 --feature_mode indicator"
"$V --only_pfold 1 --only_rfold 1 --feature_mode indicator"
"$V --only_pfold 1 --only_rfold 2 --feature_mode indicator"
"$V --only_pfold 2 --only_rfold 0 --feature_mode indicator"
"$V --only_pfold 2 --only_rfold 1 --feature_mode indicator"
"$V --only_pfold 2 --only_rfold 2 --feature_mode indicator"
"$V --only_pfold 3 --only_rfold 0 --feature_mode indicator"
"$V --only_pfold 3 --only_rfold 1 --feature_mode indicator"
"$V --only_pfold 3 --only_rfold 2 --feature_mode indicator"
"$V --only_pfold 4 --only_rfold 0 --feature_mode indicator"
"$V --only_pfold 4 --only_rfold 1 --feature_mode indicator"
"$V --only_pfold 4 --only_rfold 2 --feature_mode indicator"
"$V --only_pfold 0 --only_rfold 0 --feature_mode permute"
"$V --only_pfold 0 --only_rfold 1 --feature_mode permute"
"$V --only_pfold 0 --only_rfold 2 --feature_mode permute"
"$V --only_pfold 1 --only_rfold 0 --feature_mode permute"
"$V --only_pfold 1 --only_rfold 1 --feature_mode permute"
"$V --only_pfold 1 --only_rfold 2 --feature_mode permute"
"$V --only_pfold 2 --only_rfold 0 --feature_mode permute"
"$V --only_pfold 2 --only_rfold 1 --feature_mode permute"
"$V --only_pfold 2 --only_rfold 2 --feature_mode permute"
"$V --only_pfold 3 --only_rfold 0 --feature_mode permute"
"$V --only_pfold 3 --only_rfold 1 --feature_mode permute"
"$V --only_pfold 3 --only_rfold 2 --feature_mode permute"
"$V --only_pfold 4 --only_rfold 0 --feature_mode permute"
"$V --only_pfold 4 --only_rfold 1 --feature_mode permute"
"$V --only_pfold 4 --only_rfold 2 --feature_mode permute"
)
i=0
while [ $i -lt ${#Q[@]} ]; do
  n=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -c .)
  if [ "$n" -lt "$MAXJ" ]; then
    echo "$(date -Is) queue7 launching [$i]: ${Q[$i]}" >> $L/gpu_queue.log
    nohup bash -c "${Q[$i]}" > $L/queue7_job_$i.log 2>&1 &
    i=$((i+1)); sleep 150
  else
    sleep 60
  fi
done
echo "$(date -Is) queue7 drained" >> $L/gpu_queue.log
