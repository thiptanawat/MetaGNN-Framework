#!/bin/bash
# Language-Model Evidence, main backbone (Qwen3.8-27B endpoint), after run_endpoint_il.sh: Design E re-collected with
# the arms interleaved and the complete reply archived (the 40 patients of the original Design E, the
# 10 of Design A extended by the same seed; same reactions; same donors), then Design A interleaved once
# more with an independently chosen donor schedule (donor seed 4048) as a sensitivity run for the swap.
set -u
cd ~/metagnn/llm_v3
L=~/metagnn/logs; URL=http://ray-serve.203.156.3.39.nip.io/qwen3-8-27b/v1/chat/completions; M=qwen3-8-27b
while pgrep -f "[r]un_endpoint_il.sh" > /dev/null; do sleep 60; done
python3 llm_probe.py --url $URL --model $M --n 300 --patients 40 --patients_from repr_patients.json \
  --sampling random --seed 2024 --interleave --full_archive --out repr40_il --workers 12 > $L/repr40_il.log 2>&1
echo "$(date -Is) endpoint Design E interleaved re-collection done (repr40_il)" >> $L/gpu_queue.log
python3 llm_probe.py --url $URL --model $M --n 300 --patients 10 --sampling random --seed 2024 \
  --donor_seed 4048 --interleave --full_archive --out repr_il_d2 --workers 12 > $L/repr_il_d2.log 2>&1
echo "$(date -Is) endpoint Design A second donor schedule done (repr_il_d2)" >> $L/gpu_queue.log
