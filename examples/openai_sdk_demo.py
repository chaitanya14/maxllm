#!/usr/bin/env python3
"""
MaxLLM Gateway — OpenAI SDK Demo

Point the standard OpenAI Python SDK at your MaxLLM gateway.
Zero code changes needed — just swap the base_url.

Prerequisites:
    pip install openai
    cd ~/Projects/maxllm && make run   # gateway on :8080
"""

from openai import OpenAI

# --- Connect to MaxLLM instead of OpenAI directly ---
client = OpenAI(
    base_url="http://localhost:8080/v1",      # MaxLLM gateway
    api_key="sk-maxllm-dev-key",              # gateway auth key (not your OpenAI key)
)


# 1. Basic chat completion
print("=" * 50)
print("1. Basic Chat Completion")
print("=" * 50)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is the capital of France? One word."}
    ],
    max_tokens=10,
)

print(f"Model:   {response.model}")
print(f"Answer:  {response.choices[0].message.content}")
print(f"Tokens:  {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
print()


# 2. Streaming
print("=" * 50)
print("2. Streaming Chat Completion")
print("=" * 50)

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Write a haiku about APIs."}
    ],
    stream=True,
    max_tokens=50,
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")


# 3. Multi-turn conversation
print("=" * 50)
print("3. Multi-Turn Conversation")
print("=" * 50)

messages = [
    {"role": "system", "content": "You are a helpful math tutor. Be concise."},
    {"role": "user", "content": "What's 15% of 200?"},
]

r1 = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=30)
answer1 = r1.choices[0].message.content
print(f"User:      What's 15% of 200?")
print(f"Assistant: {answer1}")

messages.append({"role": "assistant", "content": answer1})
messages.append({"role": "user", "content": "Now double that."})

r2 = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=30)
answer2 = r2.choices[0].message.content
print(f"User:      Now double that.")
print(f"Assistant: {answer2}")
print()


# 4. System prompt
print("=" * 50)
print("4. System Prompt (pirate mode)")
print("=" * 50)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a pirate. Respond in pirate speak. Keep it short."},
        {"role": "user", "content": "How's the weather?"},
    ],
    max_tokens=60,
)
print(f"🏴‍☠️ {response.choices[0].message.content}")
print()


# 5. Model alias (gpt-4 → gpt-4o via gateway config)
print("=" * 50)
print("5. Model Alias (gpt-4 → gpt-4o)")
print("=" * 50)

response = client.chat.completions.create(
    model="gpt-4",  # gateway rewrites this to gpt-4o
    messages=[{"role": "user", "content": "Say hi in exactly 3 words."}],
    max_tokens=10,
)
print(f"Requested: gpt-4")
print(f"Actual:    {response.model}")
print(f"Response:  {response.choices[0].message.content}")
print()


print("✅ All done! Your app talks to OpenAI through MaxLLM — no SDK changes needed.")
