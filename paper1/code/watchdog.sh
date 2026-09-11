#!/bin/bash
# Detached supervisor. Its own session leader via setsid, so it outlives every
# terminal, tmux server and SSH session. Liveness is tested against a PID file
# the pipeline writes, not a process-name pattern: any command mentioning
# double_holdout.py -- including a status query typed from elsewhere -- would
# otherwise look like a running pipeline and suppress the relaunch.
source $HOME/metagnn/env.sh
exec 9>"$BASE/.watchdog.lock"; flock -n 9 || exit 0
echo $$ > $BASE/.watchdog.pid
TOTAL=75
alive() { p=$(cat $BASE/.pipeline.pid 2>/dev/null) && [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
while :; do
  cells=$(ls -1 $BASE/out/dh/*.json 2>/dev/null | wc -l | tr -d ' ')
  if alive; then run=true; else run=false; fi
  prog=$(grep -oE "\[[0-9]+/[0-9]+ cells[^]]*\]" $BASE/logs/run_all.log 2>/dev/null | tail -1)
  cat > $BASE/STATUS.json <<EOJ
{"heartbeat_utc":"$(date -u +%FT%TZ)","host":"$(hostname -s)","cells_done":$cells,"cells_total":$TOTAL,
 "pipeline_running":$run,"progress":"$prog",
 "gpu0":"$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 0 2>/dev/null|tr -d ' ')",
 "disk_avail":"$(df -hP $HOME|tail -1|awk '{print $4}')","watchdog_pid":$$}
EOJ
  if ! alive && [ "$cells" -lt "$TOTAL" ]; then
    echo "$(date -Is) relaunching at $cells/$TOTAL" >> $BASE/logs/watchdog.log
    setsid nohup $BASE/work/run_all.sh >> $BASE/logs/run_all.log 2>&1 < /dev/null &
    sleep 45
  fi
  sleep 60
done
