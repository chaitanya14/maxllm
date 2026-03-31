# MaxLLM vs LiteLLM — Head-to-Head Benchmark

> **TL;DR:** MaxLLM handles **33.8x more requests/sec** than LiteLLM with **36x lower latency** — on the same hardware, same mock upstream, same payload.

## Results

```
┌─────────────────┬────────────┬─────────────┬─────────────┬──────────────┐
│                 │  Req/sec   │ Avg Latency │ Max Latency │  Throughput  │
├─────────────────┼────────────┼─────────────┼─────────────┼──────────────┤
│ Baseline (mock) │   217,699  │    160 µs   │    5.3 ms   │  73.08 MB/s  │
│ MaxLLM          │    71,203  │    667 µs   │    3.9 ms   │  31.51 MB/s  │
│ LiteLLM         │     2,109  │  24.16 ms   │  252.9 ms   │   2.43 MB/s  │
└─────────────────┴────────────┴─────────────┴─────────────┴──────────────┘
```

### Key Numbers

| Metric | MaxLLM | LiteLLM | Difference |
|---|---|---|---|
| **Requests/sec** | 71,203 | 2,109 | **33.8x faster** |
| **Avg latency** | 667 µs | 24.16 ms | **36x lower** |
| **Max latency** | 3.9 ms | 252.9 ms | **65x lower** |
| **Proxy overhead** | ~507 µs | ~24 ms | **47x less overhead** |
| **Total requests (30s)** | 2,143,200 | 63,376 | 33.8x more |

### Proxy Overhead

How much latency does each proxy add on top of the baseline?

- **MaxLLM:** 507 µs (667 µs − 160 µs baseline) — **sub-millisecond**
- **LiteLLM:** ~24 ms (24,160 µs − 160 µs baseline) — **47x more overhead**

MaxLLM preserves **32.7%** of raw upstream throughput. LiteLLM preserves **0.97%**.

## Test Setup

### Hardware

- **Machine:** Mac mini (M4, 2024)
- **Chip:** Apple M4 — 10 cores (4P + 6E)
- **Memory:** 16 GB unified
- **OS:** macOS 15.3 (Darwin 24.3.0, arm64)

### Software Versions

| Component | Version |
|---|---|
| MaxLLM | built from source (Rust, release mode) |
| LiteLLM | 1.82.6 (pip, Python 3.14) |
| wrk | 4.2.0 [kqueue] |
| Rust | 1.94.0 (2026-03-02) |

### Configuration

**Load generator:** `wrk` with 4 threads, 50 concurrent connections, 30-second duration.

**Mock upstream:** A pre-compiled HTTP server returning a fixed OpenAI-format JSON response (~350 bytes) with no processing delay. This isolates pure proxy overhead — no network variance, no model inference time.

**Payload:**
```json
{
  "model": "mock-model",
  "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}]
}
```

**MaxLLM config:** 4 threads, no plugins, single route to mock upstream.

**LiteLLM config:** 4 workers (`--num_workers 4`), single model pointing to same mock upstream.

Both proxies were configured with equivalent minimal setups — no auth, no logging, no rate limiting — to measure pure proxy overhead.

### Methodology

1. Start mock upstream on `:9999`
2. Run baseline benchmark (wrk → mock directly)
3. Start MaxLLM on `:8080`, verify with curl, run benchmark
4. Stop MaxLLM, start LiteLLM on `:8090`, verify with curl, run benchmark
5. Each test runs for 30 seconds to capture steady-state performance

All tests ran sequentially on the same machine to ensure fair comparison. The mock upstream was verified alive before each proxy test.

## Why the Difference?

| | MaxLLM | LiteLLM |
|---|---|---|
| **Language** | Rust (compiled, zero-cost abstractions) | Python (interpreted, GIL-bound) |
| **Runtime** | Pingora (Cloudflare's proxy framework) | uvicorn + FastAPI + asyncio |
| **HTTP handling** | Zero-copy, kernel-level kqueue | Python async with multiple abstraction layers |
| **Memory model** | Stack-allocated, no GC | Heap-heavy, garbage collected |
| **Connection handling** | Epoll/kqueue native | Python event loop |

LiteLLM routes requests through ~15 layers of Python abstraction (router → fallback → retry → function wrapper → OpenAI client → httpx → aiohttp transport) before a single byte reaches the upstream. MaxLLM's Pingora pipeline does the same work in compiled Rust with near-zero overhead.

## Reproducing

```bash
# Build MaxLLM (release mode)
cargo build --release

# Run the benchmark
bash perf/benchmark_clean.sh
```

Requires: `wrk`, `litellm` (pip), and the mock upstream binary in `perf/`.

---

*Benchmark conducted March 31, 2026. Results may vary by hardware.*
