#!/bin/bash
# The linear reference rows on the family-disjoint folds (CPU): every unfitted and fitted row of
# naive_baselines_allfolds.py, including the nearest-neighbor label lookup, and the pooled frozen
# rows of pooled_baselines.py, with the reaction folds and the inner tuning partition drawn family
# by family from the union family map. Writes naive_allfolds_fam.json and naive_pooled_fam.json.
cd ~/metagnn/work
source ~/metagnn/env.sh
export P1_FAMILY_MAP=~/metagnn/work/families_union.json
export NJOBS=${NJOBS:-8}
P1_OUT=~/metagnn/out/naive_allfolds_fam.json python3 naive_baselines_allfolds.py > ~/metagnn/logs/naive_fam.log 2>&1
P1_OUT=~/metagnn/out/naive_pooled_fam.json python3 pooled_baselines.py > ~/metagnn/logs/pooled_fam.log 2>&1
echo "$(date -Is) family linear rows done" >> ~/metagnn/logs/gpu_queue.log
