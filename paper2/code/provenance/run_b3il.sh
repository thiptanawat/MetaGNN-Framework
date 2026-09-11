#!/bin/bash
# Paper 2, third backbone (mistralai/Mistral-Small-3.2-24B-Instruct-2506) collected under the clean
# protocol: Design A with the three arms interleaved and the complete reply archived (repr_b3il), the
# three positive controls (pc_b3il) and the MSI probe (msi_b3il) with the complete archive, all with
# --plain_chat because the Mistral chat template has no thinking switch. The first collection of this
# backbone (repr_b3, pc_b3, msi_b3) was made in blocked order without the complete archive by an
# earlier version of run_backbone3.sh and is kept as it was.
# Pauses the Paper 1 robustness queue (the running cells finish; no new cell starts), waits for GPU 0
# to be free of training cells, serves the model, runs the three drivers, and restarts the queue.
# GPU 1 is left alone for the Ray service.
set -u
export CUDA_HOME=/usr/local/cuda-12.4 PATH=$HOME/.local/bin:/usr/local/cuda-12.4/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
cd ~/metagnn/llm_v3
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
L=~/metagnn/logs; mkdir -p $L
MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506; NAME=mistral-small-3.2-24b-instruct; PORT=8002
# let the queue launch the last cell of the inner-validation set (six cells; five run at once) before
# pausing it: wait until one of the first five finishes, give the loop time to launch the sixth
while [ "$(pgrep -fc "[d]ouble_holdout_ctrl.py")" -ge 5 ]; do sleep 60; done
sleep 240
pkill -f "[b]ash $HOME/metagnn/work/gpu_queue6.sh"
echo "$(date -Is) queue6 paused for the third backbone's clean collection (run_b3il.sh); running cells finish first" >> $L/gpu_queue.log
while pgrep -f "[d]ouble_holdout_ctrl.py" > /dev/null; do sleep 60; done
echo "$(date -Is) backbone3 clean collection serve start: $MODEL as $NAME" >> $L/gpu_queue.log
nohup vllm serve $MODEL --served-model-name $NAME --host 127.0.0.1 --port $PORT \
  --tokenizer_mode mistral --config_format mistral --load_format mistral \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --max-logprobs 0 \
  --limit-mm-per-prompt "{\"image\":0}" --uvicorn-log-level warning \
  > $L/serve_b3il.log 2>&1 &
SERVE_PID=$!
for i in $(seq 1 180); do
  curl -s http://127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\"" && break; sleep 10
done
curl -s http://127.0.0.1:$PORT/v1/models > $L/models_b3il.json
if grep -q "\"$NAME\"" $L/models_b3il.json; then
  URL=http://127.0.0.1:$PORT/v1/chat/completions
  python3 llm_probe.py --url $URL --model $NAME --n 300 --patients 10 --sampling random --seed 2024 \
    --plain_chat --interleave --full_archive --out repr_b3il --workers 12 > $L/repr_b3il.log 2>&1
  python3 positive_control.py --url $URL --model $NAME --plain_chat --full_archive --out pc_b3il --workers 12 > $L/pc_b3il.log 2>&1
  python3 msi_probe.py --url $URL --model $NAME --plain_chat --full_archive --out msi_b3il --workers 10 > $L/msi_b3il.log 2>&1
  echo "$(date -Is) backbone3 clean collection done (repr_b3il, pc_b3il, msi_b3il)" >> $L/gpu_queue.log
else
  echo "$(date -Is) backbone3 server did not come up for the clean collection" >> $L/gpu_queue.log
fi
kill $SERVE_PID; sleep 15; pkill -f "vllm serve $MODEL"; sleep 5
echo "$(date -Is) resuming queue6" >> $L/gpu_queue.log
nohup bash ~/metagnn/work/gpu_queue6.sh > $L/queue6e.log 2>&1 &
