#!/bin/bash
# The second replication of the own-versus-donor contrast on the main backbone
# (REPLICATION2_PLAN_2026-11-14.md), as issued on the first host through the service that issued the
# first replication.
#
#   SERVICE=http://HOST:PORT DASHBOARD=http://HOST:PORT bash run_replication2_service.sh
#
# SERVICE is the address of the service's HTTP proxy and DASHBOARD that of its status interface. Both
# are read: the script sends chat requests and reads the status, and changes nothing in the service.
# It starts only when $BASE/GO_REPLICATION2 exists (created once the plan is public). Before the first
# call it saves the service's own record of its applications and its version and reduces them to
# results/replication2_b1/serving_record.json; if that record names another model source than the one
# of the first replication, nothing is collected. Then 600 patient-blind prompts of an earlier design
# measure the rate and show that answers parse, and then wording 1, blocks 1 to 20, is collected.
#
# Night window. Other clients use the service by day. A block is started only between 22:00 and 06:00
# local time, and a block still running at 06:30 is left incomplete; the script then waits for 22:00
# and goes on with the calls still to be sent. These limits depend on the clock and on no answer.
#
# Watchdog. While the service does not report the application RUNNING, the deployment HEALTHY and two
# replicas RUNNING, the collector is paused (SIGSTOP); it is resumed 60 s after the service is healthy
# again. Every step can be repeated: the collector skips answered calls and completed blocks.
set -u
BASE=${BASE:-$HOME/metagnn}; STUDY=${STUDY:-$BASE/lme2/paper2}; L=$BASE/logs; LOG=$L/replication2_service.log
: "${SERVICE:?set SERVICE to the address of the service proxy}"
: "${DASHBOARD:?set DASHBOARD to the address of the status interface}"
APP=qwen3-8-27b; SOURCE=Qwen/Qwen3.8-27B-FP8; URL=$SERVICE/$APP/v1/chat/completions; WORKERS=24; BLOCKS=20
OUT=results/replication2_b1; MASTER=results/replication2/design_master.json
mkdir -p $L
say () { echo "$(date -Is) $*" >> $LOG; }

while [ ! -f $BASE/GO_REPLICATION2 ]; do sleep 60; done
cd $STUDY || exit 1
export PYTHONPATH=code
[ -f recon3d_aligned.json ] || gunzip -k recon3d_aligned.json.gz
mkdir -p $OUT
say "study checkout at $(git -C $STUDY rev-parse HEAD)"

window () {   # prints the two limits of the current night window, or nothing by day
  local h; h=$(date +%-H)
  if [ "$h" -ge 22 ]; then echo "$(date -d tomorrow +%F)T06:00 $(date -d tomorrow +%F)T06:30"
  elif [ "$h" -lt 6 ]; then echo "$(date +%F)T06:00 $(date +%F)T06:30"; fi
}
status () {   # "RUNNING HEALTHY 2" when the service is in order
  curl -s -m 8 $DASHBOARD/api/serve/applications/ | python3 -c "import sys,json; d=json.load(sys.stdin); a=d['applications']['$APP']; e=a['deployments']['LLMServer:$APP']; print(a.get('status'), e.get('status'), sum(1 for r in e.get('replicas', []) if r.get('state') == 'RUNNING'))" 2>/dev/null
}
watchdog () {
  local paused=0 healthy_since=0 S P now
  while true; do
    S=$(status); [ -z "$S" ] && S="UNKNOWN UNKNOWN 0"
    P=$(pgrep -f "[r]eplication_probe.py --url $URL" | head -1); now=$(date +%s)
    if [ "$S" != "RUNNING HEALTHY 2" ]; then
      healthy_since=0
      if [ -n "$P" ] && [ $paused -eq 0 ]; then kill -STOP $P; paused=1; say "watchdog: service $S; collector $P paused"; fi
    else
      [ $healthy_since -eq 0 ] && healthy_since=$now
      if [ $paused -eq 1 ] && [ $((now - healthy_since)) -ge 60 ]; then
        [ -n "$P" ] && kill -CONT $P; paused=0; say "watchdog: service healthy for 60 s; collector $P resumed"
      fi
    fi
    sleep 20
  done
}

while [ -z "$(window)" ]; do sleep 60; done
if [ "$(status)" != "RUNNING HEALTHY 2" ]; then say "the service is not healthy with two replicas; not started"; exit 1; fi
if [ ! -f $OUT/serving_record.json ]; then
  T=$(date +%Y-%m-%dT%H:%M:%S%z)
  curl -s -m 20 $DASHBOARD/api/serve/applications/ > $L/replication2_serve_applications.json
  curl -s -m 20 $DASHBOARD/api/version > $L/replication2_ray_version.json
  if ! python3 code/replication_service_record.py --applications $L/replication2_serve_applications.json \
        --version $L/replication2_ray_version.json --read-at $T --out $OUT/serving_record.json >> $LOG 2>&1; then
    say "the service record could not be written; nothing collected"; exit 1
  fi
fi
if ! grep -q "\"model_source\": \"$SOURCE\"" $OUT/serving_record.json; then
  say "the service names another model source than $SOURCE; nothing collected"; exit 1
fi

PR="python3 code/replication_probe.py --url $URL --model $APP --master $MASTER --workers $WORKERS --block_size 10 --blind_blocks 2"
if [ ! -f $OUT/w1/raw.jsonl ]; then
  # the rate, and whether answers parse, on prompts of an earlier design (nothing is stored)
  $PR --bench 600 --out $OUT/unused > $L/repl2_b1_bench.log 2>&1
  PARSED=$(sed -n 's/.*parsed \([0-9]*\).*/\1/p' $L/repl2_b1_bench.log | tail -1)
  say "b1: $(tail -1 $L/repl2_b1_bench.log)"
  if [ "${PARSED:-0}" -lt 570 ]; then say "b1: fewer than 95% of the 600 check answers parsed; the collection does not start"; exit 1; fi
fi
watchdog & WD=$!
say "b1: collection start"
while true; do
  W=$(window)
  if [ -z "$W" ]; then sleep 300; continue; fi
  set -- $W
  $PR --wording 1 --blocks 1-$BLOCKS --arms reaction patient shuffled shuffled_d2 shuffled_d3 \
      --no_start_after $1 --abandon_after $2 --out $OUT/w1 >> $L/repl2_b1_w1.log 2>&1
  RC=$?
  DONE=$(python3 -c "import json; print(len(json.load(open('$OUT/w1/design.json'))['blocks']))" 2>/dev/null || echo 0)
  say "b1: collector ended with status $RC; $DONE of $BLOCKS blocks complete ($(tail -1 $L/repl2_b1_w1.log))"
  [ "$DONE" -ge $BLOCKS ] && break
  [ $RC -ne 0 ] && { say "b1: the collector failed; stopped"; break; }
  sleep 600
done
kill $WD 2>/dev/null
say "ALLDONE (b1)"
