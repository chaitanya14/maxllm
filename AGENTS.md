# AGENTS.md

Guidance for Codex/SWE agents working in this repository.

## Project context

MaxLLM is a Rust workspace for a high-performance AI gateway built on Cloudflare Pingora. Start with `CLAUDE.md` for the current architecture, crate layout, request flow, and development notes.

## Repository layout

- `Cargo.toml` — workspace root.
- `crates/maxllm-config` — TOML config parsing, environment expansion, and validation.
- `crates/maxllm-plugin` — plugin traits, chain execution, and built-in plugins.
- `crates/maxllm-translate` — provider request/response translation.
- `crates/maxllm-admin` — virtual keys, teams, cost tracking, budgets, and admin API.
- `crates/maxllm-gateway` — Pingora proxy implementation and routing.
- `maxllm.toml` — example gateway configuration.
- `tests/` — integration tests.

## Development workflow

Prefer the narrowest validation command that covers your change, then run broader checks when practical:

```bash
cargo fmt --all --check
cargo check --workspace --all-targets
cargo test --workspace
```

If the environment has cached dependencies but restricted network access, use the offline check noted in `CLAUDE.md`:

```bash
cargo check --offline
```

## Agent safety rules

- Do not commit API keys, provider tokens, admin keys, or generated secrets.
- Keep sample credentials as placeholders or environment-variable references.
- Do not hand-edit generated references; update the scripts or source data that generate them.
- Preserve the proxy design: do not add HTTP client libraries for upstream LLM calls unless the architecture docs are intentionally changed.
- Avoid reading the request body in Pingora `request_filter`; body translation belongs in body filter hooks.

## Pull request expectations

- Summarize the user-visible behavior change and the files touched.
- Include the validation commands run, or explain why validation could not run.
- Call out any changes to provider translation, plugin lifecycle hooks, routing, auth, or secret-handling behavior.
