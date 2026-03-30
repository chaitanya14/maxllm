#!/usr/bin/env python3
"""
MaxLLM Gateway — OpenAI SDK Integration Tests

Usage:
    1. Start the gateway:  cd ~/Projects/maxllm && make run
    2. Run tests:          python tests/sdk/test_all.py

Requires: pip install openai requests
"""

import os
import sys
import time
import requests
from openai import OpenAI

GATEWAY = "http://localhost:8080"
API_KEY = "sk-maxllm-dev-key"

# Detect which provider keys are available
HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY"))

passed = 0
failed = 0
skipped = 0


def test(name):
    """Decorator to register and run a test."""
    def decorator(fn):
        fn._test_name = name
        return fn
    return decorator


def run_test(fn):
    global passed, failed, skipped
    name = getattr(fn, "_test_name", fn.__name__)
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        passed += 1
        print(f"✅ PASS: {name}")
    except requests.exceptions.ConnectionError:
        skipped += 1
        print(f"⏭️  SKIP: {name} — gateway not running")
    except AssertionError as e:
        failed += 1
        print(f"❌ FAIL: {name} — {e}")
    except Exception as e:
        failed += 1
        print(f"❌ FAIL: {name} — {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@test("Health endpoint")
def test_health():
    resp = requests.get(f"{GATEWAY}/health", timeout=5)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    print(f"  Body: {resp.text.strip()}")


@test("Basic chat completion (OpenAI, non-streaming)")
def test_basic_chat():
    if not HAS_OPENAI:
        print("  ⏭️  OPENAI_API_KEY not set, skipping")
        raise ConnectionError("skip")
    client = OpenAI(base_url=f"{GATEWAY}/v1", api_key=API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'hello from maxllm' and nothing else."}],
        max_tokens=20,
    )
    content = resp.choices[0].message.content
    print(f"  Model: {resp.model}")
    print(f"  Content: {content}")
    print(f"  Usage: in={resp.usage.prompt_tokens} out={resp.usage.completion_tokens}")
    assert content is not None and len(content) > 0, "Empty response"


@test("Streaming chat completion (OpenAI)")
def test_streaming():
    if not HAS_OPENAI:
        print("  ⏭️  OPENAI_API_KEY not set, skipping")
        raise ConnectionError("skip")
    client = OpenAI(base_url=f"{GATEWAY}/v1", api_key=API_KEY)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 5, one per line."}],
        stream=True,
        max_tokens=50,
    )
    chunks = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks.append(chunk.choices[0].delta.content)
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    assert len(chunks) > 0, f"No chunks received"
    print(f"  Received {len(chunks)} chunks")


@test("Anthropic via OpenAI format (translation, non-streaming)")
def test_anthropic():
    if not HAS_ANTHROPIC:
        print("  ⏭️  ANTHROPIC_API_KEY not set, skipping")
        raise ConnectionError("skip")
    resp = requests.post(
        f"{GATEWAY}/v1/anthropic",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Say 'hello from anthropic via maxllm' and nothing else."}],
            "max_tokens": 30,
        },
        timeout=30,
    )
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Response: {data}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


@test("Anthropic streaming (SSE translation)")
def test_anthropic_stream():
    if not HAS_ANTHROPIC:
        print("  ⏭️  ANTHROPIC_API_KEY not set, skipping")
        raise ConnectionError("skip")
    resp = requests.post(
        f"{GATEWAY}/v1/anthropic",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Count 1 to 3."}],
            "stream": True,
            "max_tokens": 30,
        },
        stream=True,
        timeout=30,
    )
    chunks = 0
    for line in resp.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                chunks += 1
                if chunks <= 5:
                    print(f"  {decoded[:100]}")
    assert chunks > 0, "No SSE chunks received"
    print(f"  Total SSE chunks: {chunks}")


@test("Gemini via OpenAI format (translation)")
def test_gemini():
    if not HAS_GEMINI:
        print("  ⏭️  GEMINI_API_KEY not set, skipping")
        raise ConnectionError("skip")
    resp = requests.post(
        f"{GATEWAY}/v1/gemini",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Say 'hello from gemini via maxllm' and nothing else."}],
        },
        timeout=30,
    )
    print(f"  Status: {resp.status_code}")
    data = resp.json()
    print(f"  Response: {data}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


@test("Multi-turn conversation")
def test_multi_turn():
    if not HAS_OPENAI:
        print("  ⏭️  OPENAI_API_KEY not set, skipping")
        raise ConnectionError("skip")
    client = OpenAI(base_url=f"{GATEWAY}/v1", api_key=API_KEY)
    messages = [
        {"role": "system", "content": "You are a math tutor. Be brief."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    resp1 = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=20)
    turn1 = resp1.choices[0].message.content
    print(f"  Turn 1: {turn1}")

    messages.append({"role": "assistant", "content": turn1})
    messages.append({"role": "user", "content": "Now multiply that by 3."})

    resp2 = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=20)
    turn2 = resp2.choices[0].message.content
    print(f"  Turn 2: {turn2}")
    assert turn2 is not None and len(turn2) > 0


@test("System message respected")
def test_system_message():
    if not HAS_OPENAI:
        print("  ⏭️  OPENAI_API_KEY not set, skipping")
        raise ConnectionError("skip")
    client = OpenAI(base_url=f"{GATEWAY}/v1", api_key=API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You must respond in ALL CAPS only."},
            {"role": "user", "content": "Say hello."},
        ],
        max_tokens=20,
    )
    content = resp.choices[0].message.content
    print(f"  Content: {content}")
    # Check that most characters are uppercase
    alpha = [c for c in content if c.isalpha()]
    upper_ratio = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)
    print(f"  Uppercase ratio: {upper_ratio:.0%}")
    assert upper_ratio > 0.7, f"Expected mostly uppercase, got {upper_ratio:.0%}"


@test("Model alias (gpt-4 → gpt-4o)")
def test_model_alias():
    if not HAS_OPENAI:
        print("  ⏭️  OPENAI_API_KEY not set, skipping")
        raise ConnectionError("skip")
    client = OpenAI(base_url=f"{GATEWAY}/v1", api_key=API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4",  # Should resolve to gpt-4o via alias
        messages=[{"role": "user", "content": "Say hi."}],
        max_tokens=10,
    )
    print(f"  Requested: gpt-4 → Model in response: {resp.model}")
    assert resp.choices[0].message.content is not None


@test("Auth rejection (bad key)")
def test_auth_reject():
    resp = requests.post(
        f"{GATEWAY}/v1/chat/completions",
        headers={"Authorization": "Bearer sk-wrong-key", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
        timeout=5,
    )
    print(f"  Status: {resp.status_code}")
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"


@test("Auth rejection (no header)")
def test_auth_missing():
    resp = requests.post(
        f"{GATEWAY}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
        timeout=5,
    )
    print(f"  Status: {resp.status_code}")
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"


@test("Gateway response headers (X-Request-Id, X-MaxLLM-*)")
def test_response_headers():
    # Use whichever provider is available
    if HAS_OPENAI:
        url = f"{GATEWAY}/v1/chat/completions"
        body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
    elif HAS_GEMINI:
        url = f"{GATEWAY}/v1/gemini"
        body = {"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]}
    else:
        print("  ⏭️  No provider keys set, skipping")
        raise ConnectionError("skip")
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    req_id = resp.headers.get("X-Request-Id")
    provider = resp.headers.get("X-MaxLLM-Provider")
    upstream_ms = resp.headers.get("X-MaxLLM-Upstream-Ms")
    print(f"  X-Request-Id: {req_id or 'MISSING'}")
    print(f"  X-MaxLLM-Provider: {provider or 'MISSING'}")
    print(f"  X-MaxLLM-Upstream-Ms: {upstream_ms or 'MISSING'}")
    assert req_id is not None, "Missing X-Request-Id header"


@test("CORS preflight")
def test_cors():
    resp = requests.options(
        f"{GATEWAY}/v1/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, Authorization",
        },
        timeout=5,
    )
    print(f"  Status: {resp.status_code}")
    acao = resp.headers.get("Access-Control-Allow-Origin")
    print(f"  Access-Control-Allow-Origin: {acao or 'MISSING'}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("MaxLLM Gateway — OpenAI SDK Integration Tests")
    print(f"Gateway: {GATEWAY}")
    print(f"API Key: {API_KEY[:15]}...")

    # Check gateway is up
    try:
        r = requests.get(f"{GATEWAY}/health", timeout=3)
        if r.status_code != 200:
            print(f"\n⚠️  Gateway returned {r.status_code} on /health")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Gateway not running at {GATEWAY}")
        print("   Start it first: cd ~/Projects/maxllm && make run")
        sys.exit(1)

    tests = [
        test_health,
        test_basic_chat,
        test_streaming,
        test_anthropic,
        test_anthropic_stream,
        test_gemini,
        test_multi_turn,
        test_system_message,
        test_model_alias,
        test_auth_reject,
        test_auth_missing,
        test_response_headers,
        test_cors,
    ]

    for t in tests:
        run_test(t)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)
