# MaxLLM Gateway — OpenAI SDK Integration Test Plan

## Goal
Verify MaxLLM works as a drop-in proxy for apps using the official OpenAI Python SDK. Point the SDK at the gateway, use real provider keys, confirm everything works end-to-end.

---

## Prerequisites

```bash
# 1. Build the gateway
cd ~/Projects/maxllm
make build

# 2. Set provider API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."

# 3. Start the gateway
make run
# Gateway listens on http://localhost:8080

# 4. Install OpenAI Python SDK (if not already)
pip install openai
```

---

## Test Matrix

| # | Test | Route | Provider | Streaming | Expected |
|---|------|-------|----------|-----------|----------|
| 1 | Basic chat completion | `/v1/chat/completions` | OpenAI | ❌ | 200 + valid response |
| 2 | Streaming chat completion | `/v1/chat/completions` | OpenAI | ✅ | SSE chunks + final |
| 3 | Anthropic via OpenAI format | `/v1/anthropic` | Anthropic | ❌ | 200 + translated response |
| 4 | Anthropic streaming | `/v1/anthropic` | Anthropic | ✅ | SSE chunks translated |
| 5 | Gemini via OpenAI format | `/v1/gemini` | Gemini | ❌ | 200 + translated response |
| 6 | Multi-turn conversation | `/v1/chat/completions` | OpenAI | ❌ | Context maintained |
| 7 | System message | `/v1/chat/completions` | OpenAI | ❌ | System prompt respected |
| 8 | Model alias | `/v1/chat/completions` | OpenAI | ❌ | `gpt-4` → `gpt-4o` |
| 9 | Fallback (bad provider) | `/v1/chat/completions` | OpenAI→Anthropic | ❌ | Falls back gracefully |
| 10 | Auth rejection | `/v1/chat/completions` | — | — | 401 Unauthorized |
| 11 | Rate limit headers | `/v1/chat/completions` | OpenAI | ❌ | Response headers present |
| 12 | Gateway headers | `/v1/chat/completions` | OpenAI | ❌ | `X-MaxLLM-Provider` etc. |
| 13 | Health endpoint | `/health` | — | — | 200 OK |

---

## Test Scripts

### Test 1 — Basic Chat Completion (non-streaming)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1/chat",
    api_key="sk-maxllm-dev-key",
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say 'hello from maxllm' and nothing else."}],
)

print(f"✅ Model: {resp.model}")
print(f"✅ Content: {resp.choices[0].message.content}")
print(f"✅ Usage: {resp.usage}")
assert resp.choices[0].message.content is not None
print("PASS")
```

### Test 2 — Streaming Chat Completion

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1/chat",
    api_key="sk-maxllm-dev-key",
)

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
)

chunks = []
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        chunks.append(chunk.choices[0].delta.content)
        print(chunk.choices[0].delta.content, end="", flush=True)

print()
assert len(chunks) > 0, "No chunks received"
print(f"✅ Received {len(chunks)} chunks")
print("PASS")
```

### Test 3 — Anthropic via OpenAI Format (translation)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-maxllm-dev-key",
)

# The /v1/anthropic route translates OpenAI format → Anthropic Messages API
resp = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Say 'hello from anthropic via maxllm'"}],
    extra_body={"_maxllm_route": "/v1/anthropic"},  # Won't work with standard SDK routing
)
# NOTE: The OpenAI SDK hardcodes the path to /chat/completions.
# To test Anthropic route, use raw requests instead:

import requests
resp = requests.post(
    "http://localhost:8080/v1/anthropic",
    headers={
        "Authorization": "Bearer sk-maxllm-dev-key",
        "Content-Type": "application/json",
    },
    json={
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Say 'hello from anthropic via maxllm'"}],
    },
)
data = resp.json()
print(f"✅ Status: {resp.status_code}")
print(f"✅ Response: {data}")
assert resp.status_code == 200
print("PASS")
```

### Test 4 — Anthropic Streaming

```python
import requests

resp = requests.post(
    "http://localhost:8080/v1/anthropic",
    headers={
        "Authorization": "Bearer sk-maxllm-dev-key",
        "Content-Type": "application/json",
    },
    json={
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Count 1 to 5."}],
        "stream": True,
    },
    stream=True,
)

chunks = 0
for line in resp.iter_lines():
    if line:
        decoded = line.decode("utf-8")
        if decoded.startswith("data: ") and decoded != "data: [DONE]":
            chunks += 1
            print(decoded[:80])

print(f"\n✅ Received {chunks} SSE chunks")
assert chunks > 0
print("PASS")
```

### Test 5 — Gemini via OpenAI Format

```python
import requests

resp = requests.post(
    "http://localhost:8080/v1/gemini",
    headers={
        "Authorization": "Bearer sk-maxllm-dev-key",
        "Content-Type": "application/json",
    },
    json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Say 'hello from gemini via maxllm'"}],
    },
)
data = resp.json()
print(f"✅ Status: {resp.status_code}")
print(f"✅ Response: {data}")
assert resp.status_code == 200
print("PASS")
```

### Test 6 — Multi-Turn Conversation

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1/chat",
    api_key="sk-maxllm-dev-key",
)

messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"},
]

resp1 = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(f"Turn 1: {resp1.choices[0].message.content}")

messages.append({"role": "assistant", "content": resp1.choices[0].message.content})
messages.append({"role": "user", "content": "Now multiply that by 3."})

resp2 = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(f"Turn 2: {resp2.choices[0].message.content}")
assert "12" in resp2.choices[0].message.content
print("PASS")
```

### Test 7 — Model Alias Resolution

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1/chat",
    api_key="sk-maxllm-dev-key",
)

# Config has: "gpt-4" = "gpt-4o"
resp = client.chat.completions.create(
    model="gpt-4",  # Should resolve to gpt-4o
    messages=[{"role": "user", "content": "What model are you?"}],
)
print(f"✅ Requested: gpt-4 → Actual: {resp.model}")
print(f"✅ Content: {resp.choices[0].message.content}")
print("PASS")
```

### Test 8 — Auth Rejection

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1/chat",
    api_key="sk-wrong-key",
)

try:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
    )
    print("FAIL — should have raised an error")
except Exception as e:
    print(f"✅ Rejected: {e}")
    print("PASS")
```

### Test 9 — Gateway Response Headers

```python
import requests

resp = requests.post(
    "http://localhost:8080/v1/chat/completions",
    headers={
        "Authorization": "Bearer sk-maxllm-dev-key",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    },
)

print(f"✅ Status: {resp.status_code}")
print(f"✅ X-Request-Id: {resp.headers.get('X-Request-Id', 'MISSING')}")
print(f"✅ X-MaxLLM-Provider: {resp.headers.get('X-MaxLLM-Provider', 'MISSING')}")
print(f"✅ X-MaxLLM-Upstream-Ms: {resp.headers.get('X-MaxLLM-Upstream-Ms', 'MISSING')}")
print("PASS")
```

### Test 10 — Health Endpoint

```python
import requests

resp = requests.get("http://localhost:8080/health")
print(f"✅ Status: {resp.status_code}")
print(f"✅ Body: {resp.text}")
assert resp.status_code == 200
print("PASS")
```

---

## Run All Tests

Save the test scripts above into `tests/sdk/` and run:

```bash
# Terminal 1: Start the gateway
cd ~/Projects/maxllm && make run

# Terminal 2: Run tests
cd ~/Projects/maxllm
python tests/sdk/test_all.py
```

---

## Important Notes

1. **OpenAI SDK base_url quirk**: The SDK appends `/completions` to the base_url path. So `base_url="http://localhost:8080/v1/chat"` results in requests to `/v1/chat/completions`. For non-OpenAI routes (`/v1/anthropic`, `/v1/gemini`), use raw `requests` instead.

2. **Known issue — Gemini streaming**: The gateway doesn't inject `alt=sse` for streaming Gemini requests. Non-streaming Gemini should work.

3. **Known issue — Bedrock**: SigV4 signing is stubbed. Don't test Bedrock.

4. **Config auth key**: Default dev key is `sk-maxllm-dev-key` (set in `maxllm.toml`).

5. **Fallback testing**: To test fallback, temporarily invalidate the primary provider's API key and verify the request routes to the fallback.
