# MaxLLM Roadmap

A phased plan to build MaxLLM into a complete, production-grade AI gateway — leveraging our Rust/Pingora performance advantage.

## Current State (v0.1)

MaxLLM today is a working AI gateway with:

- **15 providers**: OpenAI, Anthropic, Gemini, Azure, Bedrock, Groq, Together, Fireworks, Mistral, xAI, DeepSeek, Ollama, Cohere, DeepInfra, and custom OpenAI-compatible endpoints
- **1 endpoint**: `/v1/chat/completions`
- **Streaming**: Full SSE translation (Anthropic SSE, Gemini SSE, Cohere SSE -> OpenAI SSE)
- **Smart routing**: Fallback, weighted, round-robin, least-connections with circuit breaker
- **Plugin system**: 13 built-in plugins across 5 lifecycle hooks
- **Guardrails**: 8 providers (prompt_guard, pii_filter, secret_scan, keyword_block, regex_guard, webhook, lakera, cel)
- **Virtual keys**: Key generation, per-key budgets, rate limits, model access control, teams
- **Cost tracking**: Automatic cost calculation with built-in model pricing for 10+ model families
- **Metrics**: Prometheus (latency, tokens, fallbacks, active requests)
- **Performance**: 178K req/s on `/health`, <1ms overhead on proxied requests

---

## Phase 1: Multi-Endpoint Support

**Goal**: Support embeddings, image generation, audio, and moderation endpoints.

| Endpoint | Format | Priority |
|----------|--------|----------|
| `/v1/embeddings` | Simple input/output, many providers | High |
| `/v1/images/generations` | OpenAI DALL-E format | Medium |
| `/v1/audio/transcriptions` | Multipart form data (Whisper) | Medium |
| `/v1/audio/speech` | Binary response body (TTS) | Medium |
| `/v1/moderations` | Classification response | Low |
| `/v1/completions` | Legacy text completions | Low |
| `/v1/rerank` | Cohere-style reranking | Low |

### Architecture Change

Routes need an `endpoint_type` field so the gateway selects the right translator:

```toml
[[routes]]
path = "/v1/embeddings"
endpoint_type = "embeddings"
provider = "openai"
fallback = ["cohere"]
```

Each endpoint type gets its own request/response translation in `maxllm-translate`:

```
crates/maxllm-translate/src/
  chat/          # existing chat completions translation
  embeddings/    # new
  images/        # new
  audio/         # new
```

---

## Phase 2: Advanced Routing & Load Balancing

**Goal**: Smart routing beyond current strategies.

### Additional Routing Strategies

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Latency-based** | Route to fastest provider | Track p50/p99 in atomic ring buffer, select lowest |
| **Cost-based** | Route to cheapest provider | Model cost map lookup, select minimum |

### Additional Features

| Feature | Description |
|---------|-------------|
| **Retries with backoff** | `num_retries` per route, exponential backoff before fallback |
| **Context window fallback** | Parse "context length exceeded" errors, auto-route to larger model |
| **Tag-based routing** | `x-maxllm-tag: fast` header routes to tagged provider group |
| **Wildcard routes** | `/v1/*` catch-all with provider selection |

### Configuration

```toml
[[routes]]
path = "/v1/chat/completions"
strategy = "latency"          # or "weighted", "least_connections", "cost", "round_robin"
num_retries = 2
retry_backoff_ms = 500

[[routes.targets]]
provider = "openai"
weight = 70
tags = ["fast", "us"]

[[routes.targets]]
provider = "anthropic"
weight = 30
tags = ["smart", "us"]
```

---

## Phase 3: Caching Layer

**Goal**: Reduce costs and latency for repeated queries.

### Cache Backends

| Backend | Use Case | Crate |
|---------|----------|-------|
| **In-memory LRU** | Single-node, low-latency | `moka` (async, bounded, TTL) |
| **Redis** | Multi-node shared cache | `redis` (optional feature flag) |
| **Disk** | Persistent, large cache | `sled` or filesystem |

### Cache Key

Hash of `(model, messages, temperature, top_p, max_tokens)` — deterministic params only. Exclude `stream`, `user`, metadata.

### Cache Control

```
# Per-request headers
x-maxllm-cache: true              # opt-in
x-maxllm-cache-ttl: 3600          # TTL override
Cache-Control: no-cache            # bypass cache

# Response headers
x-maxllm-cache-hit: true           # cache hit indicator
```

```toml
[plugins.cache]
category = "cache"
backend = "memory"       # or "redis"
max_entries = 10000
default_ttl = 3600
# redis_url = "redis://localhost:6379"
```

---

## Phase 4: Virtual Key Maturity

**Goal**: Production multi-tenant deployment with advanced key controls.

### Features

| Feature | Description |
|---------|-------------|
| **RBAC** | Role-based access control (admin, editor, viewer) |
| **API key rotation** | Rotate keys without downtime |
| **Usage quotas** | Daily/monthly token and request quotas per key |
| **Database backends** | SQLite (embedded) or Postgres (multi-pod) |
| **Key expiry** | Auto-expire keys after a configured duration |
| **Audit log** | Track key creation, modification, and revocation |

---

## Phase 5: Observability & Logging Integrations

**Goal**: Ship structured telemetry that works with any backend.

### Strategy: OpenTelemetry First

OpenTelemetry (OTLP) covers the majority of observability platforms. Most platforms (Datadog, Grafana, Honeycomb, Jaeger, etc.) accept OTLP natively.

### Standard Logging Payload

Every request produces a structured event:

```json
{
    "request_id": "...",
    "model": "gpt-4o",
    "provider": "openai",
    "status": 200,
    "tokens_in": 150,
    "tokens_out": 42,
    "cost_usd": 0.000795,
    "latency_ms": 1340,
    "cache_hit": false,
    "fallback_used": false,
    "key_name": "prod-service-a",
    "team_id": "engineering"
}
```

### Integrations

| Integration | Approach | Effort |
|-------------|----------|--------|
| **OpenTelemetry (OTLP)** | `opentelemetry` + `opentelemetry-otlp` crates, export traces + metrics | Medium |
| **Webhook callback** | POST payload to configurable URL on each request | Small |
| **Langfuse** | HTTP API (`/api/public/ingestion`) for trace + generation logging | Medium |
| **Custom logger plugin** | `on_logging` hook already exists, expose full payload | Small |

### Additional Features

| Feature | Description |
|---------|-------------|
| **W3C traceparent** | Parse incoming `traceparent`, propagate to upstream, include in logs |
| **Message redaction** | Strip message content from logs (GDPR compliance) |
| **Per-key log control** | Disable logging for specific keys |
| **`no-log` header** | `x-maxllm-no-log: true` to skip logging for a request |

---

## Phase 6: Admin UI & Management

**Goal**: Web dashboard for managing the gateway.

### Features

| Feature | Description |
|---------|-------------|
| **Key management** | Create, revoke, edit virtual keys |
| **Spend dashboard** | Charts for spend by key/team/model over time |
| **Health dashboard** | Per-provider health status, latency, error rates |
| **Config editor** | Edit routes, providers, plugins in the UI |
| **Live metrics** | Real-time request rate, latency percentiles |

### Implementation

- Static SPA (React or plain HTML/JS) served by Pingora
- Communicates with `/admin/*` REST API
- Protected by master key authentication
- Optional: config hot-reload without restart (watch TOML file, rebuild gateway state)

---

## Phase 7: Advanced Protocol Support

**Goal**: Support emerging AI protocols.

| Protocol | Description | Effort |
|----------|-------------|--------|
| **MCP gateway** | Accept MCP tool calls, route to appropriate LLM | Medium |
| **A2A** | Agent-to-agent message routing | Medium |
| **Batch API** | Accept batch of requests, process async, return results | Medium |
| **Responses API** | OpenAI's new `/v1/responses` format for reasoning models | Medium |

---

## Priority & Sequencing

```
Phase 1: Endpoints          ████████████████████  Embeddings alone is high-value
Phase 2: Routing            ██████████████████    Differentiator vs simple proxies
Phase 3: Caching            ████████████████      Big cost saver for users
Phase 4: Virtual Key Maturity ██████████████      Required for production multi-tenant
Phase 5: Observability      ████████████          OTEL covers most use cases
Phase 6: Admin UI           ██████████            Nice-to-have, big effort
Phase 7: Protocols          ████████              Bleeding edge, low priority
```

### Critical Path

```
Phase 1 (Endpoints) ──> Phase 4 (Virtual Keys) ──> Phase 5 (Observability)
                    \                            /
                     ──> Phase 2 (Routing) ──> Phase 3 (Caching)
```

Phases 6-7 can be built in parallel at any point.

---

## MaxLLM: Competitive Advantages

| Dimension | MaxLLM (Rust/Pingora) |
|-----------|----------------------|
| **Proxy latency** | <1ms p99 overhead |
| **Throughput** | 178K req/s (health), 14K req/s (with plugins) |
| **Memory usage** | ~10MB RSS |
| **Architecture** | True proxy — zero HTTP clients, Pingora handles all TCP/TLS/HTTP |
| **Connection pooling** | Pingora native (battle-tested at Cloudflare scale) |
| **Deployment** | Single static binary, ~15MB |
| **Startup time** | <100ms |
| **Security surface** | Minimal dependencies, memory-safe language, no arbitrary code execution via config |

### Go-to-Market Strategy

1. **Phase 1-2**: Target performance-sensitive teams hitting scale limits with existing proxies. "Drop-in replacement for your top providers, 100x faster."
2. **Phase 3-4**: Target platform teams building internal AI gateways. "Full multi-tenant gateway in a single binary."
3. **Phase 5-7**: Target enterprises. "Production-grade AI gateway with guardrails, compliance, and observability."
