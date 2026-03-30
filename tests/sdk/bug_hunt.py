#!/usr/bin/env python3
"""
MaxLLM Gateway — Aggressive Bug Hunt

Throws every edge case, weird combo, and adversarial request at the gateway.
"""

import json
import time
import requests
from openai import OpenAI

GW = "http://localhost:8080"
KEY = "sk-maxllm-dev-key"
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

client = OpenAI(base_url=f"{GW}/v1", api_key=KEY)

results = []

def test(name):
    def decorator(fn):
        fn._name = name
        return fn
    return decorator

def run(fn):
    name = fn._name
    try:
        fn()
        results.append(("PASS", name, ""))
        print(f"  ✅ {name}")
    except Exception as e:
        msg = str(e)[:200]
        results.append(("FAIL", name, msg))
        print(f"  ❌ {name}")
        print(f"     → {msg}")

# ===================================================================
# CATEGORY 1: Request format edge cases
# ===================================================================
print("\n🔬 CATEGORY 1: Request Format Edge Cases")
print("-" * 50)

@test("Empty messages array")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "model": "gpt-4o-mini", "messages": []
    }, timeout=15)
    # Should return 400, not crash
    assert r.status_code in (400, 422, 200), f"Got {r.status_code}: {r.text[:200]}"
run(t)

@test("No model field")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "messages": [{"role": "user", "content": "hi"}]
    }, timeout=15)
    assert r.status_code in (400, 200), f"Got {r.status_code}: {r.text[:200]}"
run(t)

@test("Empty body")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={}, timeout=10)
    assert r.status_code in (400, 422), f"Got {r.status_code}: {r.text[:200]}"
run(t)

@test("Non-JSON body (plain text)")
def t():
    r = requests.post(f"{GW}/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "text/plain"},
        data="hello", timeout=10)
    assert r.status_code in (400, 415), f"Got {r.status_code}: {r.text[:200]}"
run(t)

@test("Malformed JSON body")
def t():
    r = requests.post(f"{GW}/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        data="{bad json", timeout=10)
    assert r.status_code in (400, 422), f"Got {r.status_code}: {r.text[:200]}"
run(t)

@test("Null content in message")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": None}]
    }, timeout=15)
    # Shouldn't crash the gateway
    assert r.status_code < 500, f"Server error {r.status_code}: {r.text[:200]}"
run(t)

@test("Missing content field in message")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user"}]
    }, timeout=15)
    assert r.status_code < 500, f"Server error {r.status_code}: {r.text[:200]}"
run(t)

@test("Unknown role in message")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "wizard", "content": "cast spell"}]
    }, timeout=15)
    assert r.status_code < 500, f"Server error {r.status_code}: {r.text[:200]}"
run(t)

@test("Very long prompt (~10K chars)")
def t():
    long_msg = "Tell me about AI. " * 500  # ~9500 chars
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": long_msg}],
        max_tokens=5,
    )
    assert resp.choices[0].message.content is not None
run(t)

@test("Unicode / emoji in content")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 🎉🔥🚀 back to me"}],
        max_tokens=10,
    )
    assert resp.choices[0].message.content is not None
run(t)

@test("Newlines and special chars in content")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Line1\nLine2\tTab\r\nCRLF\\backslash\"quotes\""}],
        max_tokens=10,
    )
    assert resp.choices[0].message.content is not None
run(t)

# ===================================================================
# CATEGORY 2: Parameter combinations
# ===================================================================
print("\n🔬 CATEGORY 2: Parameter Combinations")
print("-" * 50)

@test("temperature=0 (deterministic)")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'test'"}],
        temperature=0, max_tokens=5,
    )
    assert resp.choices[0].message.content is not None
run(t)

@test("temperature=2.0 (max)")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'test'"}],
        temperature=2.0, max_tokens=5,
    )
    assert resp.choices[0].message.content is not None
run(t)

@test("top_p=0.1")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'test'"}],
        top_p=0.1, max_tokens=5,
    )
    assert resp.choices[0].message.content is not None
run(t)

@test("max_tokens=1")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a poem"}],
        max_tokens=1,
    )
    content = resp.choices[0].message.content
    assert content is not None
    print(f"     (got: '{content}')")
run(t)

@test("n=2 (multiple completions)")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Pick a random number 1-100"}],
        n=2, max_tokens=10,
    )
    assert len(resp.choices) == 2, f"Expected 2 choices, got {len(resp.choices)}"
run(t)

@test("stop sequence")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 10 with commas"}],
        stop=[","], max_tokens=50,
    )
    content = resp.choices[0].message.content
    assert "," not in (content or ""), f"Stop sequence not honored: {content}"
run(t)

@test("presence_penalty + frequency_penalty")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        presence_penalty=1.5, frequency_penalty=1.5, max_tokens=10,
    )
    assert resp.choices[0].message.content is not None
run(t)

@test("seed parameter (reproducibility)")
def t():
    kwargs = dict(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Pick a color"}],
        seed=42, temperature=0, max_tokens=5,
    )
    r1 = client.chat.completions.create(**kwargs)
    r2 = client.chat.completions.create(**kwargs)
    print(f"     r1='{r1.choices[0].message.content}' r2='{r2.choices[0].message.content}'")
    # Both should work, even if not identical
run(t)

@test("response_format=json_object")
def t():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Respond in JSON."},
            {"role": "user", "content": "Give me a JSON object with keys 'name' and 'age'"},
        ],
        response_format={"type": "json_object"},
        max_tokens=50,
    )
    content = resp.choices[0].message.content
    parsed = json.loads(content)
    assert "name" in parsed, f"Missing 'name' in {parsed}"
run(t)

# ===================================================================
# CATEGORY 3: Streaming edge cases
# ===================================================================
print("\n🔬 CATEGORY 3: Streaming Edge Cases")
print("-" * 50)

@test("Streaming with max_tokens=1")
def t():
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True, max_tokens=1,
    )
    chunks = list(stream)
    assert len(chunks) > 0, "No chunks"
    print(f"     ({len(chunks)} chunks)")
run(t)

@test("Streaming with stop sequence")
def t():
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count 1 to 10 with commas"}],
        stream=True, stop=[","], max_tokens=50,
    )
    text = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    print(f"     (got: '{text}')")
run(t)

@test("Streaming with temperature=0")
def t():
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'deterministic'"}],
        stream=True, temperature=0, max_tokens=10,
    )
    text = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    assert len(text) > 0
run(t)

# ===================================================================
# CATEGORY 4: Routing & provider edge cases
# ===================================================================
print("\n🔬 CATEGORY 4: Routing & Provider Edge Cases")
print("-" * 50)

@test("Nonexistent model name")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "model": "gpt-999-turbo-ultra",
        "messages": [{"role": "user", "content": "hi"}],
    }, timeout=15)
    # Should get error from upstream, not gateway crash
    assert r.status_code < 500, f"Server error: {r.status_code}"
    print(f"     (status: {r.status_code})")
run(t)

@test("Nonexistent route")
def t():
    r = requests.post(f"{GW}/v1/fake/route", headers=HEADERS, json={
        "model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]
    }, timeout=10)
    assert r.status_code == 404
run(t)

@test("GET on POST-only route")
def t():
    r = requests.get(f"{GW}/v1/chat/completions", headers=HEADERS, timeout=10)
    assert r.status_code in (404, 405), f"Got {r.status_code}"
run(t)

@test("Gemini — non-streaming")
def t():
    r = requests.post(f"{GW}/v1/gemini", headers=HEADERS, json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Say 'pong'"}],
    }, timeout=30)
    assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"
    data = r.json()
    print(f"     Response format: {list(data.keys())}")
run(t)

@test("Gemini — with system message")
def t():
    r = requests.post(f"{GW}/v1/gemini", headers=HEADERS, json={
        "model": "gemini-2.5-flash",
        "messages": [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Say hi"},
        ],
    }, timeout=30)
    assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"
run(t)

@test("Gemini — multi-turn")
def t():
    r = requests.post(f"{GW}/v1/gemini", headers=HEADERS, json={
        "model": "gemini-2.5-flash",
        "messages": [
            {"role": "user", "content": "My name is Bob."},
            {"role": "assistant", "content": "Hi Bob!"},
            {"role": "user", "content": "What's my name?"},
        ],
    }, timeout=30)
    assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"
    text = str(r.json())
    assert "Bob" in text or "bob" in text, f"Context lost: {r.text[:200]}"
run(t)

@test("Gemini — streaming")
def t():
    r = requests.post(f"{GW}/v1/gemini", headers=HEADERS, json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Count 1 to 3"}],
        "stream": True,
    }, stream=True, timeout=30)
    chunks = 0
    for line in r.iter_lines():
        if line and line.decode().startswith("data: "):
            chunks += 1
    print(f"     ({chunks} SSE chunks, status {r.status_code})")
    # Known issue: Gemini streaming may not work (missing alt=sse)
    if chunks == 0:
        print(f"     ⚠️  BUG: No SSE chunks — streamGenerateContent not wired up")
        raise AssertionError("Gemini streaming returns no SSE chunks")
run(t)

@test("Ollama route (local, may not be running)")
def t():
    r = requests.post(f"{GW}/v1/ollama", headers=HEADERS, json={
        "model": "gemma3:1b",
        "messages": [{"role": "user", "content": "hi"}],
    }, timeout=15)
    if r.status_code == 502 or r.status_code == 504:
        print(f"     ⏭️  Ollama not running (status {r.status_code})")
    else:
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"
        print(f"     Ollama responded: {r.status_code}")
run(t)

# ===================================================================
# CATEGORY 5: Concurrency & timing
# ===================================================================
print("\n🔬 CATEGORY 5: Concurrency & Timing")
print("-" * 50)

@test("Rapid sequential requests (5x)")
def t():
    for i in range(5):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Say '{i}'"}],
            max_tokens=5,
        )
        assert resp.choices[0].message.content is not None
    print(f"     All 5 completed")
run(t)

@test("Concurrent requests (3 parallel via threads)")
def t():
    import concurrent.futures
    def do_req(i):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Say '{i}'"}],
            max_tokens=5,
        )
        return resp.choices[0].message.content

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(do_req, i) for i in range(3)]
        results_list = [f.result(timeout=30) for f in futures]

    assert all(r is not None for r in results_list)
    print(f"     All 3 parallel requests completed: {results_list}")
run(t)

@test("Request timeout (very short client timeout)")
def t():
    try:
        r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Write a 1000 word essay about philosophy"}],
            "max_tokens": 2000,
        }, timeout=0.001)  # unreasonably short
        print(f"     Surprisingly got response: {r.status_code}")
    except requests.exceptions.Timeout:
        print(f"     Client timed out as expected")
    except requests.exceptions.ConnectionError:
        print(f"     Connection error (also acceptable)")
    # Gateway shouldn't crash either way
    time.sleep(1)
    health = requests.get(f"{GW}/health", timeout=5)
    assert health.status_code == 200, "Gateway crashed after timeout!"
run(t)

# ===================================================================
# CATEGORY 6: Security edge cases
# ===================================================================
print("\n🔬 CATEGORY 6: Security Edge Cases")
print("-" * 50)

@test("Auth with empty Bearer token")
def t():
    r = requests.post(f"{GW}/v1/chat/completions",
        headers={"Authorization": "Bearer ", "Content-Type": "application/json"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10)
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"
run(t)

@test("Auth with just 'Bearer' (no space)")
def t():
    r = requests.post(f"{GW}/v1/chat/completions",
        headers={"Authorization": "Bearer", "Content-Type": "application/json"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10)
    assert r.status_code == 401, f"Expected 401, got {r.status_code}"
run(t)

@test("Huge body (1MB of text)")
def t():
    huge = "x" * (1024 * 1024)
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": huge}],
    }, timeout=30)
    # Should handle gracefully — 400 or 413
    assert r.status_code < 500, f"Server error: {r.status_code}"
    print(f"     (status: {r.status_code})")
run(t)

@test("SQL injection in model name")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "model": "'; DROP TABLE users; --",
        "messages": [{"role": "user", "content": "hi"}],
    }, timeout=15)
    assert r.status_code < 500, f"Server error: {r.status_code}"
    print(f"     (status: {r.status_code})")
run(t)

@test("Path traversal in route")
def t():
    r = requests.post(f"{GW}/v1/../../etc/passwd", headers=HEADERS, json={}, timeout=10)
    assert r.status_code in (400, 404), f"Got {r.status_code}"
run(t)

@test("Extra headers pass-through")
def t():
    r = requests.post(f"{GW}/v1/chat/completions",
        headers={**HEADERS, "X-Custom-Header": "test", "X-Evil": "<script>alert(1)</script>"},
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        timeout=15)
    assert r.status_code == 200, f"Got {r.status_code}"
run(t)

# ===================================================================
# CATEGORY 7: Response validation
# ===================================================================
print("\n🔬 CATEGORY 7: Response Validation")
print("-" * 50)

@test("Response has valid OpenAI format (id, object, choices, usage)")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    }, timeout=15)
    data = r.json()
    assert "id" in data, f"Missing 'id': {data.keys()}"
    assert "object" in data, f"Missing 'object': {data.keys()}"
    assert data["object"] == "chat.completion", f"Wrong object: {data['object']}"
    assert "choices" in data, f"Missing 'choices'"
    assert "usage" in data, f"Missing 'usage'"
    assert "model" in data, f"Missing 'model'"
    print(f"     Format OK: id={data['id'][:20]}... object={data['object']}")
run(t)

@test("Gemini response is NOT translated to OpenAI format (known bug)")
def t():
    r = requests.post(f"{GW}/v1/gemini", headers=HEADERS, json={
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Say 'test'"}],
    }, timeout=30)
    data = r.json()
    if "candidates" in data:
        print(f"     ⚠️  BUG CONFIRMED: Gemini returns native format, not OpenAI format")
        print(f"     Keys: {list(data.keys())}")
        raise AssertionError("Gemini response not translated to OpenAI format")
    else:
        assert "choices" in data, f"Unexpected format: {data.keys()}"
run(t)

@test("Streaming chunks have valid SSE format")
def t():
    r = requests.post(f"{GW}/v1/chat/completions", headers=HEADERS, json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True, "max_tokens": 10,
    }, stream=True, timeout=15)
    
    lines = []
    for line in r.iter_lines():
        if line:
            lines.append(line.decode())
    
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) > 0, "No data lines"
    
    # Check last line is [DONE]
    last_data = data_lines[-1]
    assert last_data == "data: [DONE]", f"Last chunk not [DONE]: {last_data}"
    
    # Check intermediate chunks parse as JSON
    for dl in data_lines[:-1]:
        payload = dl[6:]  # strip "data: "
        parsed = json.loads(payload)
        assert "choices" in parsed, f"Missing choices in chunk: {parsed}"
    print(f"     {len(data_lines)} data lines, [DONE] present")
run(t)

@test("finish_reason is set correctly")
def t():
    # max_tokens should give 'length'
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a very long poem about the ocean"}],
        max_tokens=5,
    )
    reason = resp.choices[0].finish_reason
    print(f"     finish_reason={reason}")
    assert reason in ("stop", "length"), f"Unexpected: {reason}"
run(t)

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 60)
print("BUG HUNT SUMMARY")
print("=" * 60)

passes = [r for r in results if r[0] == "PASS"]
fails = [r for r in results if r[0] == "FAIL"]

print(f"\n✅ PASSED: {len(passes)}/{len(results)}")
print(f"❌ FAILED: {len(fails)}/{len(results)}")

if fails:
    print(f"\n{'─'*60}")
    print("BUGS FOUND:")
    print(f"{'─'*60}")
    for _, name, msg in fails:
        print(f"  ❌ {name}")
        print(f"     {msg}")
