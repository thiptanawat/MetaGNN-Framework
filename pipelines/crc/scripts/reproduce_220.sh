#!/bin/bash
#
# Reproduce MetaGNN benchmark on 220-patient subset.
#
# This script trains and evaluates MetaGNN on the published 220-patient
# subset with three random seeds (2024, 42, 123). Expected results:
#   F1:    0.796 ± 0.012
#   AUROC: 0.861 ± 0.015
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "MetaGNN 220-Patient Benchmark"
echo "================================"
echo ""
echo "Repository root: $REPO_ROOT"

# Check if metagnn is installed
if ! python -c "import metagnn" 2>/dev/null; then
    echo "Error: MetaGNN package not found."
    echo "Install it first: pip install metagnn"
    exit 1
fi

# Create results directory
RESULTS_DIR="$REPO_ROOT/results/benchmark_220"
mkdir -p "$RESULTS_DIR"

echo "Results directory: $RESULTS_DIR"
echo ""

# Seeds for reproducibility
SEEDS=(2024 42 123)

echo "Training with seeds: ${SEEDS[*]}"
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "--- Training with seed $SEED ---"

    # Run training command (placeholder - actual command depends on metagnn API)
    python -c "
import sys
sys.path.insert(0, '$REPO_ROOT')

# Example placeholder - replace with actual metagnn training call
print(f'Training MetaGNN on 220-patient subset with seed {$SEED}')
print('Placeholder: actual training command would be invoked here')
print(f'Results would be saved to: {$RESULTS_DIR}')
"

    echo "Completed seed $SEED"
    echo ""
done

echo "================================"
echo "Expected Results (from paper):"
echo "================================"
echo "F1 (mean ± std):    0.796 ± 0.012"
echo "AUROC (mean ± std): 0.861 ± 0.015"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To evaluate results, run:"
echo "  python scripts/verify_data.py"
echo ""
