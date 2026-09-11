#!/bin/bash
# The frozen percentile-only interface on the two locally served backbones (Mistral Small 3.2 24B,
# the primary confirmatory backbone, and Gemma 4 31B-it), on the development cohort and the two
# external cohorts, in the three configurations. Waits for the family GNN queue to free GPU 0, then
# serves each backbone in turn (same flags as the development collection) and runs msi_probe2.py.
# Restart-safe: a rerun skips answered calls. Runs from a directory holding _paths.py, msi_probe2.py,
# frozen_scorer.py, aligned_rxn.py, recon3d_aligned.json, rxn_context.npz, clinical_metadata_msi.tsv,
# msi_int/design.json and results/external/{panel_frozen.json,gse39582_panel.npz,gse13294_panel.npz}.
set -u
export CUDA_HOME=/usr/local/cuda-12.4 PATH=$HOME/.local/bin:/usr/local/cuda-12.4/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
cd ~/metagnn/llm_v3
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
L=~/metagnn/logs; mkdir -p $L ~/metagnn/results/msi2
python3 frozen_scorer.py > $L/frozen_scorer.log 2>&1 || true

collect () {   # backbone_name port plain_chat_flag
  local NAME=$1 PORT=$2 PLAIN=$3
  for cfg in zero_shot evidence tool; do
    python3 msi_probe2.py --url http://127.0.0.1:$PORT/v1/chat/completions --model $NAME $PLAIN \
      --cohort tcga --endpoint nonmsih --config $cfg --donor_seed 11 --out ~/metagnn/results/msi2/tcga_${NAME}_${cfg}_s11 --workers 10 > $L/msi2_tcga_${NAME}_${cfg}.log 2>&1
    python3 msi_probe2.py --url http://127.0.0.1:$PORT/v1/chat/completions --model $NAME $PLAIN \
      --cohort gse39582 --config $cfg --donor_seed 11 --out ~/metagnn/results/msi2/gse39582_${NAME}_${cfg}_s11 --workers 10 > $L/msi2_gse39582_${NAME}_${cfg}.log 2>&1
    python3 msi_probe2.py --url http://127.0.0.1:$PORT/v1/chat/completions --model $NAME $PLAIN \
      --cohort gse13294 --config $cfg --donor_seed 11 --out ~/metagnn/results/msi2/gse13294_${NAME}_${cfg}_s11 --workers 10 > $L/msi2_gse13294_${NAME}_${cfg}.log 2>&1
  done
}

serve_and_collect () {   # model served_name port extra_serve plain_chat_flag
  local MODEL=$1 NAME=$2 PORT=$3 EXTRA=$4 PLAIN=$5
  echo "$(date -Is) msi2: serve $MODEL as $NAME" >> $L/gpu_queue.log
  nohup vllm serve $MODEL --served-model-name $NAME --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 --enable-prefix-caching --max-logprobs 0 \
    $EXTRA --uvicorn-log-level warning > $L/serve_msi2_$NAME.log 2>&1 &
  local SPID=$!
  for i in $(seq 1 180); do curl -s http://127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\"" && break; sleep 10; done
  if curl -s http://127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\""; then
    collect $NAME $PORT "$PLAIN"
    echo "$(date -Is) msi2: $NAME done" >> $L/gpu_queue.log
  else
    echo "$(date -Is) msi2: $NAME server did not come up" >> $L/gpu_queue.log
  fi
  kill $SPID 2>/dev/null; sleep 15; pkill -f "vllm serve $MODEL"; sleep 10
}

while pgrep -f "[d]ouble_holdout_ctrl.py" > /dev/null; do sleep 120; done
echo "$(date -Is) msi2: GPU free, starting" >> $L/gpu_queue.log
serve_and_collect mistralai/Mistral-Small-3.2-24B-Instruct-2506 mistral-small-3.2-24b-instruct 8002 \
  "--tokenizer_mode mistral --config_format mistral --load_format mistral --limit-mm-per-prompt {\"image\":0}" "--plain_chat"
serve_and_collect google/gemma-4-31B-it gemma-4-31b-it 8001 \
  "--default-chat-template-kwargs {\"enable_thinking\":false}" ""
echo "$(date -Is) msi2: all local backbones done" >> $L/gpu_queue.log
