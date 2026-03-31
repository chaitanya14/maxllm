#!/bin/bash
# Head-to-head: MaxLLM vs LiteLLM — clean output
set -euo pipefail

MOCK_PORT=9999
MAXLLM_PORT=8080
LITELLM_PORT=8090
DURATION=30s
THREADS=4
CONNECTIONS=50
WRK_BODY='{"model":"mock-model","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}]}'
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Kill stale processes
for p in $MOCK_PORT $MAXLLM_PORT $LITELLM_PORT; do kill $(lsof -ti :$p) 2>/dev/null || true; done
sleep 0.5

MOCK_PID=""
MAXLLM_PID=""
LITELLM_PID=""

cleanup() {
  [ -n "$MOCK_PID" ] && kill $MOCK_PID 2>/dev/null || true
  [ -n "$MAXLLM_PID" ] && kill $MAXLLM_PID 2>/dev/null || true
  [ -n "$LITELLM_PID" ] && kill $LITELLM_PID 2>/dev/null || true
  wait 2>/dev/null
}
trap cleanup EXIT

cat > /tmp/bench_post.lua <<'EOF'
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"model":"mock-model","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}]}'
EOF

# ---- Start mock upstream ----
echo "Starting mock upstream on :$MOCK_PORT..."
"$PROJECT_DIR/perf/mock_upstream" $MOCK_PORT > /dev/null 2>&1 &
MOCK_PID=$!
sleep 0.5
curl -sf http://127.0.0.1:$MOCK_PORT/v1/chat/completions -X POST -H "Content-Type: application/json" -d "$WRK_BODY" > /dev/null
echo "Mock upstream OK"

# ---- BASELINE ----
echo ""
echo "============================================"
echo "  BASELINE: wrk → mock upstream (no proxy)"
echo "  $THREADS threads, $CONNECTIONS connections, $DURATION"
echo "============================================"
wrk -t$THREADS -c$CONNECTIONS -d$DURATION -s /tmp/bench_post.lua http://127.0.0.1:$MOCK_PORT/v1/chat/completions

# ---- MaxLLM ----
echo ""
echo "Starting MaxLLM on :$MAXLLM_PORT..."

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

"$PROJECT_DIR/target/release/maxllm" start -c /tmp/maxllm_bench.toml > /dev/null 2>&1 &
MAXLLM_PID=$!
sleep 1
curl -sf http://127.0.0.1:$MAXLLM_PORT/v1/chat/completions -X POST -H "Content-Type: application/json" -d "$WRK_BODY" > /dev/null
echo "MaxLLM OK"

echo ""
echo "============================================"
echo "  MaxLLM: wrk → MaxLLM → mock upstream"
echo "  $THREADS threads, $CONNECTIONS connections, $DURATION"
echo "============================================"
wrk -t$THREADS -c$CONNECTIONS -d$DURATION -s /tmp/bench_post.lua http://127.0.0.1:$MAXLLM_PORT/v1/chat/completions

kill $MAXLLM_PID 2>/dev/null || true
wait $MAXLLM_PID 2>/dev/null || true
MAXLLM_PID=""
sleep 1

# ---- LiteLLM ----
echo ""
echo "Starting LiteLLM on :$LITELLM_PORT..."

cat > /tmp/litellm_config.yaml <<EOF
model_list:
  - model_name: mock-model
    litellm_params:
      model: openai/mock-model
      api_base: http://127.0.0.1:$MOCK_PORT
      api_key: mock
EOF

litellm --config /tmp/litellm_config.yaml --port $LITELLM_PORT --num_workers 4 > /dev/null 2>&1 &
LITELLM_PID=$!

echo "Waiting for LiteLLM to start..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:$LITELLM_PORT/health > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Verify with a real request
if ! curl -sf http://127.0.0.1:$LITELLM_PORT/v1/chat/completions -X POST \
  -H "Content-Type: application/json" -d "$WRK_BODY" > /dev/null 2>&1; then
  echo "ERROR: LiteLLM verify FAILED — cannot reach mock upstream"
  exit 1
fi
echo "LiteLLM OK"

echo ""
echo "============================================"
echo "  LiteLLM: wrk → LiteLLM → mock upstream"
echo "  $THREADS threads, $CONNECTIONS connections, $DURATION"
echo "============================================"
wrk -t$THREADS -c$CONNECTIONS -d$DURATION -s /tmp/bench_post.lua http://127.0.0.1:$LITELLM_PORT/v1/chat/completions

echo ""
echo "============================================"
echo "  BENCHMARK COMPLETE"
echo "============================================"
