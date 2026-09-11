#!/bin/bash
# Paper 2, third backbone: mistralai/Mistral-Small-3.2-24B-Instruct-2506 served locally on GPU 0 with
# vLLM, then Design A (same seed, same 300 reactions and 10 patients as the main run), the three
# positive controls and the MSI probe, all through the released drivers with --plain_chat (the Mistral
# chat template has no thinking switch). Waits for the Paper 1 robustness cells on GPU 0 to finish,
# and restarts their queue when done. GPU 1 is left alone for the Ray service.
set -u
export CUDA_HOME=/usr/local/cuda-12.4 PATH=$HOME/.local/bin:/usr/local/cuda-12.4/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}
cd ~/metagnn/llm_v3
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
L=~/metagnn/logs; mkdir -p $L
MODEL=${MODEL:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}; NAME=${NAME:-mistral-small-3.2-24b-instruct}; SUF=${SUF:-b3}; PORT=${PORT:-8002}
# wait for the download and for GPU 0 to be free of training cells
while pgrep -f "download mistralai/Mistral-Small-3.2-24B-Instruct-2506" > /dev/null; do sleep 30; done
ls ~/.cache/huggingface/hub/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/*/consolidated.safetensors > /dev/null 2>&1 || { echo "$(date -Is) backbone3: model files missing after download" >> $L/gpu_queue.log; exit 1; }
while pgrep -f "double_holdout_ctrl.py" > /dev/null; do sleep 60; done
echo "$(date -Is) backbone3 serve start: $MODEL as $NAME" >> $L/gpu_queue.log
nohup vllm serve $MODEL --served-model-name $NAME --host 127.0.0.1 --port $PORT \
  --tokenizer_mode mistral --config_format mistral --load_format mistral \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching --max-logprobs 0 \
  --limit-mm-per-prompt '{"image":0}' --uvicorn-log-level warning \
  > $L/serve_$SUF.log 2>&1 &
SERVE_PID=$!
for i in $(seq 1 180); do
  curl -s http://127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\"" && break; sleep 10
done
curl -s http://127.0.0.1:$PORT/v1/models > $L/models_$SUF.json
if ! grep -q "\"$NAME\"" $L/models_$SUF.json; then
  echo "$(date -Is) backbone3 server did not come up" >> $L/gpu_queue.log; kill $SERVE_PID
  nohup bash ~/metagnn/work/gpu_queue6.sh > $L/queue6c.log 2>&1 &
  exit 1
fi
URL=http://127.0.0.1:$PORT/v1/chat/completions
python3 - <<PY > $L/smoke_$SUF.txt 2>&1
import json, urllib.request
payload = {"model": "$NAME", "temperature": 0.0, "max_tokens": 160,
           "messages": [{"role": "system", "content": "Answer with a JSON object only: {\\"p_active\\": <number between 0 and 1>, \\"reason\\": \\"<one short sentence>\\"}"},
                        {"role": "user", "content": "Reaction ID: PGI\\nName: glucose-6-phosphate isomerase\\nSubsystem: Glycolysis\\nEquation: 1 g6p_c <=> 1 f6p_c\\nReversible: yes\\nGene-protein-reaction rule: GPI"}]}
req = urllib.request.Request("$URL", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=180).read()); print(json.dumps(r, indent=1)[:1500])
PY
# every new run is collected with the arms interleaved and the complete reply archived
python3 llm_probe.py --url $URL --model $NAME --n 300 --patients 10 --sampling random --seed 2024 \
  --plain_chat --interleave --full_archive --out repr_$SUF --workers 12 > $L/repr_$SUF.log 2>&1
python3 positive_control.py --url $URL --model $NAME --plain_chat --full_archive --out pc_$SUF --workers 12 > $L/pc_$SUF.log 2>&1
python3 msi_probe.py --url $URL --model $NAME --plain_chat --full_archive --out msi_$SUF --workers 10 > $L/msi_$SUF.log 2>&1
kill $SERVE_PID; sleep 15; pkill -f "vllm serve $MODEL" ; sleep 5
echo "$(date -Is) backbone3 done ($SUF)" >> $L/gpu_queue.log
# the second backbone is re-collected the same way by run_b2il.sh
echo "$(date -Is) resuming queue6" >> $L/gpu_queue.log
nohup bash ~/metagnn/work/gpu_queue6.sh > $L/queue6c.log 2>&1 &
