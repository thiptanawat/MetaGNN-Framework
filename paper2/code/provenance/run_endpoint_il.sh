#!/bin/bash
# Paper 2, main backbone (Qwen3.8-27B endpoint): Design A re-collected with the arms interleaved and the
# complete reply archived, then the MSI probe with the complete archive. Same seeds as the original runs.
set -u
cd ~/metagnn/llm_v3
L=~/metagnn/logs; URL=http://ray-serve.203.156.3.39.nip.io/qwen3-8-27b/v1/chat/completions; M=qwen3-8-27b
python3 llm_probe.py --url $URL --model $M --n 300 --patients 10 --sampling random --seed 2024 \
  --interleave --full_archive --out repr_il --workers 12 > $L/repr_il.log 2>&1
python3 msi_probe.py --url $URL --model $M --full_archive --out msi_il --workers 10 > $L/msi_il.log 2>&1
echo "$(date -Is) endpoint interleaved re-collection done (repr_il, msi_il)" >> $L/gpu_queue.log
