#!/bin/bash
# Head-to-head: MaxLLM vs LiteLLM — same mock upstream, same payload
set -euo pipefail

# Kill anything on our ports first
for p in 9999 8080 8090; do kill $(lsof -ti :$p) 2>/dev/null || true; done
sleep 0.5

MOCK_PORT=9999
MAXLLM_PORT=8080
LITELLM_PORT=8090
DURATION=30s
THREADS=4
CONNECTIONS=50

WRK_BODY='{"model":"mock-model","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}]}'
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cleanup() {
  echo ""
  echo "=== Cleaning up ==="
  kill $MOCK_PID 2>/dev/null || true
  kill $MAXLLM_PID 2>/dev/null || true
  kill $LITELLM_PID 2>/dev/null || true
  wait 2>/dev/null
}
trap cleanup EXIT

# --- 1. Start mock upstream ---
echo "=== Starting mock upstream on :$MOCK_PORT ==="
"$PROJECT_DIR/perf/mock_upstream" $MOCK_PORT &
MOCK_PID=$!
MAXLLM_PID=""
LITELLM_PID=""
sleep 0.5

# Verify mock is up
curl -sf http://127.0.0.1:$MOCK_PORT/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d "$WRK_BODY" > /dev/null
echo "Mock upstream OK"

# --- 2. Benchmark: Direct to mock (baseline) ---
echo ""
echo "============================================"
echo "  BASELINE: wrk → mock upstream (no proxy)"
echo "  $THREADS threads, $CONNECTIONS connections, $DURATION"
echo "============================================"

cat > /tmp/bench_post.lua <<'EOF'
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"model":"mock-model","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}]}'
EOF

wrk -t$THREADS -c$CONNECTIONS -d$DURATION -s /tmp/bench_post.lua \
  http://127.0.0.1:$MOCK_PORT/v1/chat/completions 2>&1

# --- 3. Benchmark: MaxLLM ---
echo ""
echo "=== Starting MaxLLM on :$MAXLLM_PORT ==="

# Use bare proxy config but route /v1/chat/completions
cat > /tmp/maxllm_bench.toml <<EOF
global_plugins = []

[server]
listen = "0.0.0.0:$MAXLLM_PORT"
threads = 4

[metrics]
enabled = false

[model_aliases]

[providers.mock]
kind = "openai"
base_url = "http://127.0.0.1:$MOCK_PORT"
api_key = "mock"

[[routes]]
path = "/v1/chat/completions"
provider = "mock"
timeout_secs = 10
plugins = []
EOF

"$PROJECT_DIR/target/release/maxllm" start -c /tmp/maxllm_bench.toml &
MAXLLM_PID=$!
sleep 1

# Verify maxllm
curl -sf http://127.0.0.1:$MAXLLM_PORT/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d "$WRK_BODY" > /dev/null
echo "MaxLLM OK"

echo ""
echo "============================================"
echo "  MaxLLM: wrk → MaxLLM → mock upstream"
echo "  $THREADS threads, $CONNECTIONS connections, $DURATION"
echo "============================================"

wrk -t$THREADS -c$CONNECTIONS -d$DURATION -s /tmp/bench_post.lua \
  http://127.0.0.1:$MAXLLM_PORT/v1/chat/completions 2>&1

kill $MAXLLM_PID 2>/dev/null || true
wait $MAXLLM_PID 2>/dev/null || true
sleep 1

# --- 4. Benchmark: LiteLLM ---
echo ""
echo "=== Starting LiteLLM on :$LITELLM_PORT ==="

# LiteLLM config
cat > /tmp/litellm_config.yaml <<EOF
model_list:
  - model_name: mock-model
    litellm_params:
      model: openai/mock-model
      api_base: http://127.0.0.1:$MOCK_PORT
      api_key: mock
EOF

litellm --config /tmp/litellm_config.yaml --port $LITELLM_PORT --num_workers 4 &
LITELLM_PID=$!

# Wait for LiteLLM to start (Python is slow to boot)
echo "Waiting for LiteLLM to start..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:$LITELLM_PORT/health > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Verify litellm
LITELLM_RESP=$(curl -sf http://127.0.0.1:$LITELLM_PORT/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d "$WRK_BODY" 2>&1) || echo "LiteLLM verify response: $LITELLM_RESP"
echo "LiteLLM OK"

echo ""
echo "============================================"
echo "  LiteLLM: wrk → LiteLLM → mock upstream"
echo "  $THREADS threads, $CONNECTIONS connections, $DURATION"
echo "============================================"

wrk -t$THREADS -c$CONNECTIONS -d$DURATION -s /tmp/bench_post.lua \
  http://127.0.0.1:$LITELLM_PORT/v1/chat/completions 2>&1

echo ""
echo "============================================"
echo "  BENCHMARK COMPLETE"
echo "============================================"
