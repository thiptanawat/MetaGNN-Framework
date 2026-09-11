#!/bin/bash
# Paper 2, second backbone (google/gemma-4-31B-it) re-collected the way the third backbone and the
# endpoint runs were: Design A with the three arms interleaved and the complete reply archived
# (repr_b2il), then the MSI probe with the complete archive (msi_b2il). Same seeds, same reactions,
# patients and donors as the original second-backbone run. Serves the model locally on GPU 0 with vLLM
# in bfloat16, waits for the Paper 1 robustness cells on that GPU to finish first, and restarts their
# queue when done. GPU 1 is left alone for the Ray service.
set -u
export CUDA_HOME=/usr/local/cuda-12.4 PATH=$HOME/.local/bin:/usr/local/cuda-12.4/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
cd ~/metagnn/llm_v3
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
L=~/metagnn/logs; mkdir -p $L
MODEL=google/gemma-4-31B-it; NAME=gemma-4-31b-it; PORT=8001
while pgrep -f "double_holdout_ctrl.py" > /dev/null; do sleep 60; done
echo "$(date -Is) backbone2 interleaved re-collection serve start: $MODEL as $NAME" >> $L/gpu_queue.log
nohup vllm serve $MODEL --served-model-name $NAME --host 127.0.0.1 --port $PORT \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --max-logprobs 0 \
  --default-chat-template-kwargs "{\"enable_thinking\": false}" --uvicorn-log-level warning \
  > $L/serve_b2il.log 2>&1 &
SERVE_PID=$!
for i in $(seq 1 180); do
  curl -s http://127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\"" && break; sleep 10
done
curl -s http://127.0.0.1:$PORT/v1/models > $L/models_b2il.json
if grep -q "\"$NAME\"" $L/models_b2il.json; then
  URL=http://127.0.0.1:$PORT/v1/chat/completions
  python3 llm_probe.py --url $URL --model $NAME --n 300 --patients 10 --sampling random --seed 2024 \
    --interleave --full_archive --out repr_b2il --workers 12 > $L/repr_b2il.log 2>&1
  python3 msi_probe.py --url $URL --model $NAME --full_archive --out msi_b2il --workers 10 > $L/msi_b2il.log 2>&1
  echo "$(date -Is) backbone2 interleaved re-collection done (repr_b2il, msi_b2il)" >> $L/gpu_queue.log
else
  echo "$(date -Is) backbone2 server did not come up for the re-collection" >> $L/gpu_queue.log
fi
kill $SERVE_PID; sleep 15; pkill -f "vllm serve $MODEL"; sleep 5
echo "$(date -Is) resuming queue6" >> $L/gpu_queue.log
nohup bash ~/metagnn/work/gpu_queue6.sh > $L/queue6d.log 2>&1 &
