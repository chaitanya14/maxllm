#!/usr/bin/env python3
"""
Minimal async mock upstream that returns a canned OpenAI response instantly.
Used for perf-testing the gateway proxy path in isolation.

Usage: python3 perf/mock_upstream.py [port]  (default: 9999)
"""

import asyncio
import sys

RESPONSE_BODY = b'''{
  "id": "chatcmpl-mock-001",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "mock-model",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Four."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}
}'''

RESPONSE = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: application/json\r\n"
    b"Content-Length: " + str(len(RESPONSE_BODY)).encode() + b"\r\n"
    b"Connection: keep-alive\r\n"
    b"\r\n" + RESPONSE_BODY
)


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        while True:
            # Read until we see the end of HTTP headers
            header = await reader.readuntil(b"\r\n\r\n")
            # Read body if Content-Length present
            for line in header.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":")[1].strip())
                    await reader.readexactly(length)
                    break
            writer.write(RESPONSE)
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


async def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
    server = await asyncio.start_server(handle_client, "127.0.0.1", port)
    print(f"Mock upstream listening on 127.0.0.1:{port}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
