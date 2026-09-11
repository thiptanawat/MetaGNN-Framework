#!/bin/bash
# Input-invariance control for the graph model, patient fold 0 only.
#
# Config B reaches ~0.854 on held-out reactions, against 0.634 for ranking by
# expression and 0.737 for a structure-only logistic regression. This asks where
# that margin comes from, retraining the identical model on the identical folds
# with the expression column replaced:
#   mean  every patient sees the TRAIN-patient cohort mean -> patient specificity gone
#   zero  no expression at all                             -> only topology remains
# Three reaction folds per mode, each with a direct counterpart in the main grid.
source $HOME/metagnn/env.sh
echo $$ > $BASE/.ctrl.pid
cd $BASE/work
C="--data crc_624 --out $BASE/out/ctrl --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0 --only_pfold 0"
for MODE in zero mean; do
  echo "### CONTROL feature_mode=$MODE $(date -Is)"
  python double_holdout_ctrl.py $C --feature_mode $MODE
done
echo "### CONTROLS DONE $(date -Is)"
rm -f $BASE/.ctrl.pid
