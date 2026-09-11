#!/bin/bash
# Paper 1 full-scale double hold-out. GPU 0 only: GPU 1 runs a vLLM/Ray service.
# Restart-safe at cell granularity, so re-running resumes rather than repeats.
source $HOME/metagnn/env.sh
echo $$ > $BASE/.pipeline.pid
cd $BASE/work
C="--data crc_624 --out $BASE/out/dh --device cuda --pfolds 5 --rfolds 3"
echo "### STAGE 1 memorization contrast $(date -Is)"
python double_holdout.py $C --models mlp:0.0 mlp_emb:0.0
echo "### STAGE 2 full grid $(date -Is)"
python double_holdout.py $C --preset full
echo "### ALL DONE $(date -Is)"
rm -f $BASE/.pipeline.pid
