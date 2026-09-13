#!/bin/bash
# Language-Model Evidence: the microsatellite instability probe re-collected with its arms interleaved, so that the
# own-versus-stranger comparison on the patient-level target is read under the same protocol as the
# reaction-activity designs (the earlier collections sent the own arm first and the stranger's arm
# second). Same seed, panel, donors and prompts as before; only the dispatch order changes, and every
# reply is archived complete. Each served backbone also answers the value-echo positive control
# (positive_control.py --tasks echoval: copy the printed three-decimal expression value), which the
# main backbone answers through its endpoint separately. Runs on the main backbone's endpoint (no
# GPU), then serves the second and third backbones on GPU 0 in turn. Meant to run between the first
# and second the Reaction-Scoring Audit GPU queues (run_queue7_after6.sh waits for this script before starting the
# second queue).
set -u
export CUDA_HOME=/usr/local/cuda-12.4 PATH=$HOME/.local/bin:/usr/local/cuda-12.4/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
cd ~/metagnn/llm_v3
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
L=~/metagnn/logs; mkdir -p $L

# main backbone, through the endpoint
echo "$(date -Is) msi interleaved: main backbone start" >> $L/gpu_queue.log
python3 msi_probe.py --url http://ray-serve.203.156.3.39.nip.io/qwen3-8-27b/v1/chat/completions --model qwen3-8-27b \
  --interleave --full_archive --out msi_int --workers 10 > $L/msi_int.log 2>&1
echo "$(date -Is) msi interleaved: main backbone done (msi_int)" >> $L/gpu_queue.log

serve_and_probe () {
  local MODEL=$1 NAME=$2 PORT=$3 OUT=$4 EXTRA_SERVE=$5 EXTRA_PROBE=$6 PCOUT=$7
  echo "$(date -Is) msi interleaved: serve $MODEL as $NAME" >> $L/gpu_queue.log
  nohup vllm serve $MODEL --served-model-name $NAME --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 --enable-prefix-caching --max-logprobs 0 \
    $EXTRA_SERVE --uvicorn-log-level warning > $L/serve_$OUT.log 2>&1 &
  local SERVE_PID=$!
  for i in $(seq 1 180); do
    curl -s http://127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\"" && break; sleep 10
  done
  if curl -s http://127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\""; then
    python3 msi_probe.py --url http://127.0.0.1:$PORT/v1/chat/completions --model $NAME \
      $EXTRA_PROBE --interleave --full_archive --out $OUT --workers 10 > $L/$OUT.log 2>&1
    echo "$(date -Is) msi interleaved: $NAME done ($OUT)" >> $L/gpu_queue.log
    python3 positive_control.py --url http://127.0.0.1:$PORT/v1/chat/completions --model $NAME \
      $EXTRA_PROBE --tasks echoval --full_archive --out $PCOUT --workers 12 > $L/$PCOUT.log 2>&1
    echo "$(date -Is) value-echo control: $NAME done ($PCOUT)" >> $L/gpu_queue.log
  else
    echo "$(date -Is) msi interleaved: $NAME server did not come up" >> $L/gpu_queue.log
  fi
  kill $SERVE_PID; sleep 15; pkill -f "vllm serve $MODEL"; sleep 10
}

# the GPU must be free of training cells before a model is served
while pgrep -f "[d]ouble_holdout_ctrl.py" > /dev/null; do sleep 60; done
serve_and_probe google/gemma-4-31B-it gemma-4-31b-it 8001 msi_b2int \
  "--default-chat-template-kwargs {\"enable_thinking\":false}" "" pc_echoval_b2
serve_and_probe mistralai/Mistral-Small-3.2-24B-Instruct-2506 mistral-small-3.2-24b-instruct 8002 msi_b3int \
  "--tokenizer_mode mistral --config_format mistral --load_format mistral --limit-mm-per-prompt {\"image\":0}" "--plain_chat" pc_echoval_b3
echo "$(date -Is) msi interleaved: all done" >> $L/gpu_queue.log
