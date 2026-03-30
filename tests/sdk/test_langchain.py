#!/usr/bin/env python3
"""
MaxLLM Gateway — LangChain Integration Tests

Verifies MaxLLM works as a drop-in backend for LangChain applications.
Tests chains, streaming, structured output, tools, memory, and batch.

Prerequisites:
    pip install langchain langchain-openai
    cd ~/Projects/maxllm && make run
"""

import json
import os
import sys
import requests
from typing import Optional

# Point LangChain at MaxLLM
os.environ["OPENAI_API_KEY"] = "sk-maxllm-dev-key"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8080/v1"

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import tool
from pydantic import BaseModel, Field

GW = "http://localhost:8080"
passed = 0
failed = 0
skipped = 0


def test(name):
    def decorator(fn):
        fn._name = name
        return fn
    return decorator


def run(fn):
    global passed, failed, skipped
    name = fn._name
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        passed += 1
        print(f"✅ PASS: {name}")
    except Exception as e:
        msg = str(e)[:300]
        failed += 1
        print(f"❌ FAIL: {name}")
        print(f"   → {msg}")


# ===================================================================
# Setup
# ===================================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=f"{GW}/v1",
    api_key="sk-maxllm-dev-key",
    max_tokens=100,
)

llm_temp0 = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=f"{GW}/v1",
    api_key="sk-maxllm-dev-key",
    temperature=0,
    max_tokens=50,
)


# ===================================================================
# 1. Basic invoke
# ===================================================================

@test("Basic invoke")
def t():
    result = llm.invoke("Say 'hello from langchain' and nothing else.")
    print(f"  Content: {result.content}")
    assert result.content is not None and len(result.content) > 0
run(t)


# ===================================================================
# 2. Streaming
# ===================================================================

@test("Streaming")
def t():
    chunks = []
    for chunk in llm.stream("Count from 1 to 5, one number per line."):
        if chunk.content:
            chunks.append(chunk.content)
            print(chunk.content, end="", flush=True)
    print()
    assert len(chunks) > 0, f"No chunks received"
    print(f"  Received {len(chunks)} chunks")
run(t)


# ===================================================================
# 3. System + Human messages
# ===================================================================

@test("System + Human messages")
def t():
    messages = [
        SystemMessage(content="You are a pirate. Always respond in pirate speak. Be brief."),
        HumanMessage(content="What's 2+2?"),
    ]
    result = llm.invoke(messages)
    print(f"  🏴‍☠️ {result.content}")
    assert result.content is not None
run(t)


# ===================================================================
# 4. Multi-turn with message history
# ===================================================================

@test("Multi-turn conversation")
def t():
    messages = [
        SystemMessage(content="You are a math tutor. Be very concise."),
        HumanMessage(content="What is 7 * 8?"),
    ]
    r1 = llm_temp0.invoke(messages)
    print(f"  Turn 1: {r1.content}")

    messages.append(AIMessage(content=r1.content))
    messages.append(HumanMessage(content="Divide that by 2."))

    r2 = llm_temp0.invoke(messages)
    print(f"  Turn 2: {r2.content}")
    assert "28" in r2.content, f"Expected 28 in response: {r2.content}"
run(t)


# ===================================================================
# 5. Prompt template + chain (LCEL)
# ===================================================================

@test("LCEL chain (prompt | llm | parser)")
def t():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "input_language": "English",
        "output_language": "French",
        "text": "Hello, how are you?",
    })
    print(f"  Translation: {result}")
    assert len(result) > 0
run(t)


# ===================================================================
# 6. Streaming through LCEL chain
# ===================================================================

@test("LCEL chain streaming")
def t():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a poet. Write short poems."),
        ("human", "Write a haiku about {topic}."),
    ])
    chain = prompt | llm | StrOutputParser()
    chunks = []
    for chunk in chain.stream({"topic": "coding"}):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print()
    assert len(chunks) > 0
    print(f"  {len(chunks)} chunks")
run(t)


# ===================================================================
# 7. Structured output (Pydantic)
# ===================================================================

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating out of 10")
    summary: str = Field(description="One sentence summary")

@test("Structured output (with_structured_output)")
def t():
    structured_llm = llm.with_structured_output(MovieReview)
    result = structured_llm.invoke("Review the movie The Matrix")
    print(f"  Title: {result.title}")
    print(f"  Rating: {result.rating}/10")
    print(f"  Summary: {result.summary}")
    assert isinstance(result, MovieReview)
    assert result.rating >= 1 and result.rating <= 10
run(t)


# ===================================================================
# 8. JSON output parser
# ===================================================================

@test("JSON output parser")
def t():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You always respond with valid JSON. No markdown, no explanation."),
        ("human", "Give me a JSON object with keys 'name' (string) and 'age' (number) for a fictional person."),
    ])
    chain = prompt | llm_temp0.bind(response_format={"type": "json_object"}) | JsonOutputParser()
    result = chain.invoke({})
    print(f"  Result: {result}")
    assert "name" in result
    assert "age" in result
run(t)


# ===================================================================
# 9. Tool calling
# ===================================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 72°F in {city}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception:
        return "Error"

@test("Tool calling (bind_tools)")
def t():
    llm_with_tools = llm_temp0.bind_tools([get_weather, calculate])
    result = llm_with_tools.invoke("What's the weather in Tokyo?")
    print(f"  Tool calls: {result.tool_calls}")
    assert len(result.tool_calls) > 0, "No tool calls made"
    assert result.tool_calls[0]["name"] == "get_weather"
    assert "Tokyo" in str(result.tool_calls[0]["args"])
run(t)


# ===================================================================
# 10. Tool calling + execution
# ===================================================================

@test("Tool call → execute → respond")
def t():
    llm_with_tools = llm_temp0.bind_tools([get_weather])

    # Step 1: LLM decides to call a tool
    messages = [HumanMessage(content="What's the weather in Paris?")]
    ai_msg = llm_with_tools.invoke(messages)
    print(f"  Step 1 - Tool calls: {ai_msg.tool_calls}")
    assert len(ai_msg.tool_calls) > 0

    # Step 2: Execute the tool
    tool_call = ai_msg.tool_calls[0]
    tool_result = get_weather.invoke(tool_call["args"])
    print(f"  Step 2 - Tool result: {tool_result}")

    # Step 3: Send tool result back to LLM
    from langchain_core.messages import ToolMessage
    messages.append(ai_msg)
    messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
    final = llm.invoke(messages)
    print(f"  Step 3 - Final: {final.content}")
    assert "Paris" in final.content or "72" in final.content or "Sunny" in final.content
run(t)


# ===================================================================
# 11. Batch processing
# ===================================================================

@test("Batch invoke (3 parallel)")
def t():
    results = llm_temp0.batch([
        "What is the capital of France? One word.",
        "What is the capital of Japan? One word.",
        "What is the capital of Brazil? One word.",
    ])
    for i, r in enumerate(results):
        print(f"  [{i}] {r.content}")
    assert len(results) == 3
    assert all(r.content for r in results)
run(t)


# ===================================================================
# 12. Max tokens / stop
# ===================================================================

@test("Max tokens truncation")
def t():
    short_llm = ChatOpenAI(
        model="gpt-4o-mini",
        base_url=f"{GW}/v1",
        api_key="sk-maxllm-dev-key",
        max_tokens=5,
    )
    result = short_llm.invoke("Write a very long essay about the history of computing.")
    print(f"  Content: '{result.content}'")
    assert result.response_metadata.get("finish_reason") in ("length", "stop")
run(t)


# ===================================================================
# 13. Temperature control
# ===================================================================

@test("Temperature 0 (deterministic)")
def t():
    results = []
    for _ in range(2):
        r = llm_temp0.invoke("Pick exactly one color: red, blue, or green. Say only the color.")
        results.append(r.content.strip().lower())
    print(f"  Responses: {results}")
    # With temp=0, should be identical (or very similar)
    assert results[0] == results[1], f"Expected identical: {results}"
run(t)


# ===================================================================
# 14. Response metadata
# ===================================================================

@test("Response metadata (model, usage, finish_reason)")
def t():
    result = llm_temp0.invoke("Say hi")
    meta = result.response_metadata
    print(f"  Model: {meta.get('model_name', 'N/A')}")
    print(f"  Finish reason: {meta.get('finish_reason', 'N/A')}")
    usage = result.usage_metadata
    if usage:
        print(f"  Tokens: in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')}")
    assert meta.get("finish_reason") is not None
run(t)


# ===================================================================
# 15. Error handling (bad model)
# ===================================================================

@test("Error handling (nonexistent model)")
def t():
    bad_llm = ChatOpenAI(
        model="gpt-999-turbo",
        base_url=f"{GW}/v1",
        api_key="sk-maxllm-dev-key",
        max_tokens=10,
    )
    try:
        bad_llm.invoke("hi")
        assert False, "Should have raised an error"
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {str(e)[:100]}")
run(t)


# ===================================================================
# 16. Async invoke
# ===================================================================

@test("Async invoke (ainvoke)")
def t():
    import asyncio
    async def run_async():
        result = await llm.ainvoke("Say 'async works' and nothing else.")
        return result
    result = asyncio.run(run_async())
    print(f"  Content: {result.content}")
    assert result.content is not None
run(t)


# ===================================================================
# 17. Async streaming
# ===================================================================

@test("Async streaming (astream)")
def t():
    import asyncio
    async def run_async():
        chunks = []
        async for chunk in llm.astream("Count 1 to 3"):
            if chunk.content:
                chunks.append(chunk.content)
                print(chunk.content, end="", flush=True)
        print()
        return chunks
    chunks = asyncio.run(run_async())
    assert len(chunks) > 0
    print(f"  {len(chunks)} chunks")
run(t)


# ===================================================================
# Summary
# ===================================================================

print(f"\n{'='*60}")
print(f"LANGCHAIN TEST RESULTS")
print(f"{'='*60}")
print(f"✅ PASSED: {passed}/{passed+failed}")
print(f"❌ FAILED: {failed}/{passed+failed}")
print(f"{'='*60}")
sys.exit(1 if failed > 0 else 0)
