#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="/tmp/obsidian_assistant_pids"
LLAMA_PID="$PID_DIR/llama.pid"
ASSIST_PID="$PID_DIR/assistant.pid"
LLAMA_LOG="/tmp/obsidian_llama.log"
ASSIST_LOG="/tmp/obsidian_assistant.log"

mkdir -p "$PID_DIR"

LLAMA_MODEL="${LLAMA_MODEL:-/path/to/model.gguf}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-1234}"
LLAMA_ALIAS="${LLAMA_ALIAS:-obsidian-qwen2.5-3b}"
LLAMA_CTX="${LLAMA_CTX:-4096}"
LLAMA_THREADS="${LLAMA_THREADS:--1}"

if [ -f "$LLAMA_PID" ] && kill -0 "$(cat "$LLAMA_PID")" 2>/dev/null; then
  echo "LLM server already running (pid $(cat "$LLAMA_PID"))"
else
  nohup python3 -u -m llama_cpp.server \
    --model "$LLAMA_MODEL" \
    --host "$LLAMA_HOST" --port "$LLAMA_PORT" \
    --model_alias "$LLAMA_ALIAS" \
    --n_ctx "$LLAMA_CTX" --n_threads "$LLAMA_THREADS" \
    > "$LLAMA_LOG" 2>&1 &
  echo $! > "$LLAMA_PID"
  echo "Started LLM server (pid $(cat "$LLAMA_PID"))"
fi

if [ -f "$ASSIST_PID" ] && kill -0 "$(cat "$ASSIST_PID")" 2>/dev/null; then
  echo "Assistant server already running (pid $(cat "$ASSIST_PID"))"
else
  nohup python3 "$ROOT/server.py" > "$ASSIST_LOG" 2>&1 &
  echo $! > "$ASSIST_PID"
  echo "Started assistant server (pid $(cat "$ASSIST_PID"))"
fi

echo "Logs: $LLAMA_LOG and $ASSIST_LOG"
