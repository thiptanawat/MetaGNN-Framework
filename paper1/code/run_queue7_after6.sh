#!/bin/bash
# Waits for the first queue (gpu_queue6.sh) to run and drain, then runs the interleaved
# re-collection of the Language-Model Evidence's microsatellite instability probe (paper2/code/run_msi_interleaved.sh,
# which itself waits for the last training cells to finish before serving a model on the GPU), and
# then the second queue (gpu_queue7.sh). The first wait covers the pause in which the queue is
# restarted after the third backbone's clean collection; the second wait is the queue's own lifetime.
L=~/metagnn/logs
Q6="[b]ash $HOME/metagnn/work/gpu_queue6.sh"
until pgrep -f "$Q6" > /dev/null; do sleep 60; done
echo "$(date -Is) queue7 chain: queue6 seen running" >> $L/gpu_queue.log
while pgrep -f "$Q6" > /dev/null; do sleep 120; done
echo "$(date -Is) queue7 chain: queue6 exited; running the interleaved MSI probe collections" >> $L/gpu_queue.log
bash $HOME/metagnn/llm_v3/run_msi_interleaved.sh
echo "$(date -Is) queue7 starting" >> $L/gpu_queue.log
bash $HOME/metagnn/work/gpu_queue7.sh
