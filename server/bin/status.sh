#!/usr/bin/env bash
set -euo pipefail

PID_DIR="/tmp/obsidian_assistant_pids"
LLAMA_PID="$PID_DIR/llama.pid"
ASSIST_PID="$PID_DIR/assistant.pid"

check_pid() {
  local name="$1"
  local pid_file="$2"
  if [ -f "$pid_file" ]; then
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$name: running (pid $pid)"
      return
    fi
  fi
  echo "$name: not running"
}

check_pid "LLM server" "$LLAMA_PID"
check_pid "Assistant server" "$ASSIST_PID"

if [ -f /tmp/obsidian_llama.log ]; then
  echo "LLM log: /tmp/obsidian_llama.log"
fi
if [ -f /tmp/obsidian_assistant.log ]; then
  echo "Assistant log: /tmp/obsidian_assistant.log"
fi
