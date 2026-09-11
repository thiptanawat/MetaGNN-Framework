#!/usr/bin/env bash
# End-to-end smoke test of the DeepMeta audit pipeline on 6 held-out lines, arms `own` and
# `within` only, primary schedule (seed 11). Runs in about 2.5 minutes on 2 CPUs and never
# holds more than ~600 MB of PyG scratch on disk.
#
# The 6 lines are the first six held-out Lung lines by ModelID: one lineage keeps the `within`
# donors inside the lineage (as the real arm does) and the Lung tissue network is one of the
# small ones, so the run fits the time budget. metrics.py emits all three concordance domains
# (primary_template_domain, secondary_explicit_only, tertiary_zero_fill_all). Confidence
# intervals from 6 lines are wide by construction -- the point of the smoke test is that every
# stage runs and the numbers are well-formed, not that they are conclusive.
#
#   bash smoke.sh              # normal run
#   SMOKE_N=12 SMOKE_RF_TREES=50 bash smoke.sh   # override size / cost
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
SEED=${SEED:-11}
N=${SMOKE_N:-6}
RF=${SMOKE_RF_TREES:-200}         # same as the full run (~30 s on these 2 CPUs)
BOOT=${SMOKE_BOOT:-2000}           # same as the full run; with 6 lines the bootstrap costs seconds
PREDS_DIR=preds/smoke
t0=$(date +%s)

$PY selfcheck.py                              # concordance + domain-mask + bootstrap arithmetic
[ -f manifest.json ]  || $PY manifest.py
[ -f schedules.json ] || $PY arms.py

CELLS=$($PY - "$N" <<'EOF'
import json, sys
n = int(sys.argv[1])
man = json.load(open("manifest.json"))
ho = sorted(s["ModelID"] for s in man["samples"]
            if s["heldout"] in (True, "True") and s["OncotreeLineage"] == "Lung")
print(" ".join(ho[:n]))
EOF
)
echo "== smoke cells: $CELLS"

$PY build_arms.py --seed "$SEED" --arms own within --tag _smoke --batch_size 3 --cells $CELLS
$PY run_arms.py  --seed "$SEED" --arms own within --tag _smoke --batch_size 3 --cores 1 \
                 --out "$PREDS_DIR"
$PY metrics.py   --seed "$SEED" --arms own within --preds_dir "$PREDS_DIR" \
                 --boot "$BOOT" --rf_trees "$RF" --out smoke_results.json

echo "== smoke test finished in $(( $(date +%s) - t0 ))s; results in smoke_results.json"
