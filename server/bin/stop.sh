#!/usr/bin/env bash
set -euo pipefail

PID_DIR="/tmp/obsidian_assistant_pids"
LLAMA_PID="$PID_DIR/llama.pid"
ASSIST_PID="$PID_DIR/assistant.pid"

stop_pid() {
  local name="$1"
  local pid_file="$2"
  if [ -f "$pid_file" ]; then
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      echo "Stopped $name (pid $pid)"
    else
      echo "$name not running (stale pid $pid)"
    fi
    rm -f "$pid_file"
  else
    echo "$name pid file not found"
  fi
}

stop_pid "LLM server" "$LLAMA_PID"
stop_pid "Assistant server" "$ASSIST_PID"
