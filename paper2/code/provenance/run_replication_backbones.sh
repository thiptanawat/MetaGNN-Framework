#!/bin/bash
# The replication of the own-versus-donor contrast on the two further backbones
# (REPLICATION_BACKBONES_PLAN_2026-11-13.md), as issued on the second host.
#
#   bash run_replication_backbones.sh          # Mistral (b3), then Gemma (b2), the order of the plan
#   bash run_replication_backbones.sh b3       # one backbone
#
# One local vLLM process at a time on accelerator 1 of that host; accelerator 0 there belongs to
# another user and is never used. For each backbone: the files of the downloaded snapshot are compared
# with results/model_records.json (a backbone whose files differ is not served), the server is started
# with the flags of the earlier collections of that backbone, the serving record is written, 600
# patient-blind prompts of an earlier design measure the rate and show that answers parse, the three
# collection steps run in the order of the first plan, and the server is stopped.
#
# The script starts only when ~/metagnn/GO_REPLICATION_BACKBONES exists (created once the plan is
# public) and when no training job of the Reaction-Scoring Audit is left on the accelerator. Every
# step can be repeated: the collector skips answered calls and completed blocks.
set -u
BASE=$HOME/metagnn; STUDY=$BASE/lme/paper2; L=$BASE/logs; LOG=$L/replication_backbones.log
GPU=1; WORKERS=24
export CUDA_VISIBLE_DEVICES=$GPU VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
export HF_HOME=$BASE/hf HF_HUB_OFFLINE=1
source $BASE/venv_vllm/bin/activate
mkdir -p $L
say () { echo "$(date -Is) $*" >> $LOG; }

while [ ! -f $BASE/GO_REPLICATION_BACKBONES ]; do sleep 60; done
while pgrep -f "[g]pu_queue_cohort2.sh" > /dev/null || pgrep -f "[d]ouble_holdout_ctrl.py" > /dev/null; do sleep 60; done
say "accelerator $GPU free of training jobs; study checkout at $(git -C $STUDY rev-parse HEAD)"
cd $STUDY
[ -f recon3d_aligned.json ] || gunzip -k recon3d_aligned.json.gz

run_backbone () {   # tag  hub directory  revision  served name  port  serve flags  collector flags
  local TAG=$1 DIR=$2 REV=$3 NAME=$4 PORT=$5 SERVE_EXTRA=$6 PROBE_EXTRA=$7
  local SNAP=$HF_HOME/hub/$DIR/snapshots/$REV OUT=results/replication_$TAG
  local URL=http://127.0.0.1:$PORT/v1/chat/completions
  mkdir -p $OUT
  # the weight files against the record of the earlier collections; no server if they differ
  if ! python3 code/replication_serving_record.py --tag $TAG --served-name $NAME --snapshot $SNAP \
        --out $OUT/serving_record.json >> $LOG 2>&1; then
    say "$TAG: the snapshot differs from results/model_records.json; not served"; return 1
  fi
  local CMD="vllm serve $SNAP --served-model-name $NAME --host 127.0.0.1 --port $PORT $SERVE_EXTRA --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-logprobs 0 --uvicorn-log-level warning"
  say "$TAG: serve start: $CMD"
  nohup $CMD > $L/serve_repl_$TAG.log 2>&1 &
  local SERVE_PID=$!
  for i in $(seq 1 240); do
    curl -s http://127.0.0.1:$PORT/v1/models | grep -q "\"$NAME\"" && break
    kill -0 $SERVE_PID 2>/dev/null || break
    sleep 10
  done
  curl -s http://127.0.0.1:$PORT/v1/models > $L/models_repl_$TAG.json
  if ! grep -q "\"$NAME\"" $L/models_repl_$TAG.json; then
    say "$TAG: the server did not come up"; kill $SERVE_PID 2>/dev/null; sleep 15; pkill -f "vllm serve $SNAP"; return 1
  fi
  python3 code/replication_serving_record.py --tag $TAG --served-name $NAME --snapshot $SNAP \
      --out $OUT/serving_record.json --command "$CMD" --models-json $L/models_repl_$TAG.json >> $LOG 2>&1
  local PR="python3 code/replication_probe.py --url $URL --model $NAME $PROBE_EXTRA --workers $WORKERS --block_size 10 --blind_blocks 2"
  # the rate, and whether answers parse, on prompts of an earlier design (nothing is stored)
  $PR --bench 600 --out $OUT/unused > $L/repl_${TAG}_bench.log 2>&1
  local PARSED=$(sed -n 's/.*parsed \([0-9]*\).*/\1/p' $L/repl_${TAG}_bench.log | tail -1)
  say "$TAG: $(tail -1 $L/repl_${TAG}_bench.log)"
  if [ "${PARSED:-0}" -lt 570 ]; then
    say "$TAG: fewer than 95% of the 600 check answers parsed; the collection does not start"
  else
    # if the server stops, the collector is interrupted at once, so that it does not go on to record a
    # failed call for every remaining cell; what was answered is kept, the block in progress stays
    # incomplete, and the collection is continued by running this script again
    ( while kill -0 $SERVE_PID 2>/dev/null; do sleep 5; done
      pkill -INT -f "[r]eplication_probe.py --url $URL"; say "$TAG: the server stopped during the collection; collector interrupted" ) &
    local WATCH_PID=$!
    say "$TAG: collection start"
    # the three steps in the order of the first plan; a step runs only if the one before it ended normally
    $PR --wording 1 --blocks 1-4  --arms reaction patient shuffled shuffled_d2 shuffled_d3 --out $OUT/w1 >> $L/repl_${TAG}_w1.log 2>&1 && \
    $PR --wording 2 --blocks 1-4  --arms reaction patient shuffled --out $OUT/w2 >> $L/repl_${TAG}_w2.log 2>&1 && \
    $PR --wording 1 --blocks 5-10 --arms reaction patient shuffled shuffled_d2 shuffled_d3 --out $OUT/w1 >> $L/repl_${TAG}_w1.log 2>&1
    say "$TAG: collection end, status $? ($(tail -1 $L/repl_${TAG}_w1.log))"
    kill $WATCH_PID 2>/dev/null
  fi
  kill $SERVE_PID 2>/dev/null; sleep 15; pkill -f "vllm serve $SNAP"; sleep 10
  # the next backbone starts only after the accelerator has released the memory
  for i in $(seq 1 60); do
    [ "$(nvidia-smi -i $GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 2000 ] && break; sleep 5
  done
  say "$TAG: server stopped"
}

WHICH=${1:-all}
if [ "$WHICH" = all ] || [ "$WHICH" = b3 ]; then
  run_backbone b3 models--mistralai--Mistral-Small-3.2-24B-Instruct-2506 95a6d26c4bfb886c58daf9d3f7332c857cb27b43 \
    mistral-small-3.2-24b-instruct 18002 \
    "--tokenizer_mode mistral --config_format mistral --load_format mistral --limit-mm-per-prompt {\"image\":0}" "--plain_chat"
fi
if [ "$WHICH" = all ] || [ "$WHICH" = b2 ]; then
  run_backbone b2 models--google--gemma-4-31B-it 439edf5652646a0d1bd8b46bfdc1d3645761a445 \
    gemma-4-31b-it 18001 \
    "--default-chat-template-kwargs {\"enable_thinking\":false}" ""
fi
say "ALLDONE ($WHICH)"
