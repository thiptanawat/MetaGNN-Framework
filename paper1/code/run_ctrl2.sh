#!/bin/bash
# Extend the input-invariance control to further patient folds, so the headline
# decomposition carries fold-level error bars rather than resting on one fold.
# Queues behind the first control run rather than competing with it for the GPU.
source $HOME/metagnn/env.sh
while [ -f $BASE/.ctrl.pid ] && kill -0 $(cat $BASE/.ctrl.pid) 2>/dev/null; do sleep 120; done
echo $$ > $BASE/.ctrl2.pid
cd $BASE/work
C="--data crc_624 --out $BASE/out/ctrl --device cuda --pfolds 5 --rfolds 3 --models gnn_B:0.0"
for PF in 1 2 3 4; do
  for MODE in mean zero; do
    echo "### CONTROL pfold=$PF mode=$MODE $(date -Is)"
    python double_holdout_ctrl.py $C --feature_mode $MODE --only_pfold $PF
  done
done
echo "### CONTROL EXTENSION DONE $(date -Is)"
rm -f $BASE/.ctrl2.pid
