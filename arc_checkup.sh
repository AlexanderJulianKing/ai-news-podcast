#!/bin/bash
# Arc-continuity monitor: did the latest run read the dedup tags and chain arcs?
cd /home/alex/ai-news-podcast || exit 1
D=$(date +%y_%m_%d)
LOG="logs/log_$D.txt"
TS=$(date "+%Y-%m-%d %H:%M")
if [ ! -f "$LOG" ]; then
  echo "[$TS] arc-checkup $D: WARN  no log for today (run may not have happened yet)"
  exit 0
fi
MAP=$(grep -oE "Built headline->arc map with [0-9]+" "$LOG" | grep -oE "[0-9]+" | tail -1)
INJ=$(grep -c "Injected update context" "$LOG")
SIDE=$(grep -c "is a continuation; injecting prior audience_state" "$LOG")
MULTI=$(python3 -c "import json;a=json.load(open(\"stories_chosen/story_ledger.json\"))[\"arcs\"];print(sum(1 for v in a.values() if len(v.get(\"episodes\",[]))>1))" 2>/dev/null)
MAP=${MAP:-0}; MULTI=${MULTI:-NA}
if [ "$MAP" -gt 0 ]; then V="PASS  parser read $MAP tags"; else V="FAIL  map empty (parser not reading tags)"; fi
echo "[$TS] arc-checkup $D: $V | main_update_framing=$INJ side_continuations=$SIDE multi_episode_arcs=$MULTI"
