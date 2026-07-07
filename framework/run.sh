#!/bin/bash

################################################################################
# MetaGNN Pipeline Orchestrator
#
# Full pipeline execution: data acquisition -> graph construction -> training ->
# evaluation -> figure generation
#
# Usage: bash run.sh configs/crc_220.yaml
################################################################################

set -e  # Exit on first error

# Color output for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get config file from command line
CONFIG_FILE="${1}"

if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not provided${NC}"
    echo "Usage: bash run.sh <config_file>"
    echo "Example: bash run.sh configs/crc_220.yaml"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Get absolute path of script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR="$SCRIPT_DIR/code"

# Determine data_root and output_dir from config
# Simple YAML parsing (assumes "data_root:" and "output_dir:" on their own lines)
DATA_ROOT=$(grep "^data_root:" "$CONFIG_FILE" | cut -d' ' -f2 | tr -d '"')
OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_FILE" | cut -d' ' -f2 | tr -d '"')

# Use defaults if not specified in config
DATA_ROOT=${DATA_ROOT:-.$(basename "${CONFIG_FILE%.yaml}")/data}
OUTPUT_DIR=${OUTPUT_DIR:-.$(basename "${CONFIG_FILE%.yaml}")/output}

# Create directories
mkdir -p "$DATA_ROOT"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}MetaGNN Pipeline Orchestrator${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""
echo "Config file: $CONFIG_FILE"
echo "Data root: $DATA_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Step 1: Data Acquisition
echo -e "${YELLOW}[1/5] Data Acquisition${NC}"
if python3 "$CODE_DIR/01_data_acquisition.py" \
    --config "$CONFIG_FILE" \
    --data_root "$DATA_ROOT"; then
    echo -e "${GREEN}✓ Data acquisition completed${NC}"
else
    echo -e "${RED}✗ Data acquisition failed${NC}"
    exit 1
fi
echo ""

# Step 2: Graph Construction
echo -e "${YELLOW}[2/5] Graph Construction${NC}"
if python3 "$CODE_DIR/02_graph_construction.py" \
    --config "$CONFIG_FILE" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR"; then
    echo -e "${GREEN}✓ Graph construction completed${NC}"
else
    echo -e "${RED}✗ Graph construction failed${NC}"
    exit 1
fi
echo ""

# Step 3: Model Training
echo -e "${YELLOW}[3/5] Model Training${NC}"
if python3 "$CODE_DIR/03_train_metagnn.py" \
    --config "$CONFIG_FILE" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR"; then
    echo -e "${GREEN}✓ Model training completed${NC}"
else
    echo -e "${RED}✗ Model training failed${NC}"
    exit 1
fi
echo ""

# Step 4: Model Evaluation
echo -e "${YELLOW}[4/5] Model Evaluation${NC}"
# Find latest checkpoint
LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -name "best_model.pt" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo -e "${YELLOW}Warning: No checkpoint found, creating placeholder for evaluation${NC}"
    LATEST_CHECKPOINT="$OUTPUT_DIR/placeholder_checkpoint.pt"
fi

if python3 "$CODE_DIR/04_evaluate_metagnn.py" \
    --config "$CONFIG_FILE" \
    --checkpoint "$LATEST_CHECKPOINT" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR"; then
    echo -e "${GREEN}✓ Model evaluation completed${NC}"
else
    echo -e "${RED}✗ Model evaluation failed${NC}"
    exit 1
fi
echo ""

# Step 5: Figure Generation
echo -e "${YELLOW}[5/5] Figure Generation${NC}"
if python3 "$CODE_DIR/05_generate_figures.py" \
    --config "$CONFIG_FILE" \
    --results_dir "$OUTPUT_DIR"; then
    echo -e "${GREEN}✓ Figure generation completed${NC}"
else
    echo -e "${RED}✗ Figure generation failed${NC}"
    exit 1
fi
echo ""

# Final summary
echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""
echo "Results location: $OUTPUT_DIR"
echo ""
echo "Generated outputs:"
echo "  - Figures: $OUTPUT_DIR/figures/"
echo "  - Results: $OUTPUT_DIR/evaluation_results.json"
echo "  - Predictions: $OUTPUT_DIR/predictions.json"
echo ""
