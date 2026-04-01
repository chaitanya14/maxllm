// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use crate::factory::PluginError;
use crate::{Plugin, PluginCtx, RequestAction};
use async_trait::async_trait;
use pingora::proxy::Session;

/// Request header that clients can set to disable compaction for a single request.
pub const SKIP_COMPACTION_HEADER: &str = "x-maxllm-no-compact";

/// Extension key signaling that auto-compaction should be applied in request_body_filter.
pub const EXT_AUTO_COMPACT: &str = "auto_compact";
/// Extension key carrying the strategy string.
pub const EXT_COMPACT_STRATEGY: &str = "compact_strategy";
/// Extension key carrying the token threshold.
pub const EXT_COMPACT_THRESHOLD: &str = "compact_threshold";
/// Extension key carrying the sliding window size (for sliding_window strategy).
pub const EXT_COMPACT_WINDOW: &str = "compact_window_size";
/// Extension key carrying the LLM provider name (for llm strategy).
pub const EXT_COMPACT_LLM_PROVIDER: &str = "compact_llm_provider";
/// Extension key carrying the LLM model name (for llm strategy).
pub const EXT_COMPACT_LLM_MODEL: &str = "compact_llm_model";

/// Compaction strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum CompactionStrategy {
    /// Drop oldest non-system messages until estimated tokens are under threshold.
    Truncate,
    /// Keep system message(s) + last N messages regardless of token count.
    SlidingWindow,
    /// Summarize dropped messages via a configured LLM provider before forwarding.
    Llm,
}

impl CompactionStrategy {
    pub fn as_str(&self) -> &'static str {
        match self {
            CompactionStrategy::Truncate => "truncate",
            CompactionStrategy::SlidingWindow => "sliding_window",
            CompactionStrategy::Llm => "llm",
        }
    }
}

/// Auto-compaction plugin.
///
/// Intercepts the messages array in the request body and compacts it when the
/// estimated token count exceeds the configured threshold. Three strategies are
/// supported: truncate (drop oldest), sliding_window (keep last N), and llm
/// (summarize via a secondary LLM call).
///
/// Clients can skip compaction for a single request by sending:
///   X-MaxLLM-No-Compact: true
///
/// Config example:
/// ```toml
/// [plugins.compactor]
/// category = "auto_compaction"
/// threshold_tokens = 6000
/// strategy = "truncate"         # truncate | sliding_window | llm
/// preserve_system = true        # never drop system messages (default: true)
/// min_messages = 2              # always keep at least this many messages (default: 2)
///
/// # Required for strategy = "llm"
/// summarize_provider = "openai"
/// summarize_model = "gpt-4o-mini"
///
/// # Required for strategy = "sliding_window"
/// window_size = 20
/// ```
pub struct AutoCompactionPlugin {
    name: String,
    threshold_tokens: usize,
    strategy: CompactionStrategy,
    preserve_system: bool,
    min_messages: usize,
    window_size: usize,
    summarize_provider: Option<String>,
    summarize_model: Option<String>,
}

impl AutoCompactionPlugin {
    pub fn from_config(name: &str, config: &toml::Table) -> Result<Self, PluginError> {
        let threshold_tokens = config
            .get("threshold_tokens")
            .and_then(|v| v.as_integer())
            .unwrap_or(6000) as usize;

        let strategy = match config
            .get("strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("truncate")
        {
            "truncate" => CompactionStrategy::Truncate,
            "sliding_window" => CompactionStrategy::SlidingWindow,
            "llm" => CompactionStrategy::Llm,
            other => {
                return Err(PluginError::Config(format!(
                    "auto_compaction strategy must be 'truncate', 'sliding_window', or 'llm', got '{other}'"
                )));
            }
        };

        let preserve_system = config
            .get("preserve_system")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let min_messages = config
            .get("min_messages")
            .and_then(|v| v.as_integer())
            .unwrap_or(2) as usize;

        let window_size = config
            .get("window_size")
            .and_then(|v| v.as_integer())
            .unwrap_or(20) as usize;

        let summarize_provider = config
            .get("summarize_provider")
            .and_then(|v| v.as_str())
            .map(String::from);

        let summarize_model = config
            .get("summarize_model")
            .and_then(|v| v.as_str())
            .map(String::from);

        if strategy == CompactionStrategy::Llm {
            if summarize_provider.is_none() {
                return Err(PluginError::Config(
                    "auto_compaction with strategy 'llm' requires 'summarize_provider'".into(),
                ));
            }
            if summarize_model.is_none() {
                return Err(PluginError::Config(
                    "auto_compaction with strategy 'llm' requires 'summarize_model'".into(),
                ));
            }
        }

        Ok(Self {
            name: name.to_string(),
            threshold_tokens,
            strategy,
            preserve_system,
            min_messages,
            window_size,
            summarize_provider,
            summarize_model,
        })
    }

    /// Estimate token count for a string using the chars/4 heuristic.
    pub fn estimate_tokens(text: &str) -> usize {
        text.len().div_ceil(4)
    }

    /// Estimate total tokens across all messages in an OpenAI-format body.
    pub fn estimate_body_tokens(body: &serde_json::Value) -> usize {
        let messages = match body.get("messages").and_then(|m| m.as_array()) {
            Some(m) => m,
            None => return 0,
        };
        messages.iter().map(Self::estimate_message_tokens).sum()
    }

    /// Estimate tokens for a single message.
    pub fn estimate_message_tokens(msg: &serde_json::Value) -> usize {
        // Role overhead (~4 tokens per message for formatting)
        let mut tokens = 4;
        if let Some(content) = msg.get("content") {
            tokens += match content {
                serde_json::Value::String(s) => Self::estimate_tokens(s),
                serde_json::Value::Array(parts) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .map(Self::estimate_tokens)
                    .sum(),
                _ => 0,
            };
        }
        tokens
    }
}

#[async_trait]
impl Plugin for AutoCompactionPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_request(
        &self,
        session: &mut Session,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<RequestAction> {
        // Honour the per-request opt-out header.
        if let Some(val) = session.req_header().headers.get(SKIP_COMPACTION_HEADER) {
            let v = val.to_str().unwrap_or("").to_ascii_lowercase();
            if v == "true" || v == "1" {
                tracing::debug!(
                    plugin = self.name.as_str(),
                    "auto-compaction skipped via {} header",
                    SKIP_COMPACTION_HEADER
                );
                return Ok(RequestAction::Continue);
            }
        }

        // Signal the gateway to run compaction in request_body_filter.
        ctx.extensions.insert(EXT_AUTO_COMPACT.into(), "1".into());
        ctx.extensions
            .insert(EXT_COMPACT_STRATEGY.into(), self.strategy.as_str().into());
        ctx.extensions.insert(
            EXT_COMPACT_THRESHOLD.into(),
            self.threshold_tokens.to_string(),
        );
        ctx.extensions
            .insert(EXT_COMPACT_WINDOW.into(), self.window_size.to_string());

        // preserve_system and min_messages are booleans/numbers — encode as strings.
        ctx.extensions.insert(
            "compact_preserve_system".into(),
            if self.preserve_system { "1" } else { "0" }.into(),
        );
        ctx.extensions
            .insert("compact_min_messages".into(), self.min_messages.to_string());

        if let Some(ref provider) = self.summarize_provider {
            ctx.extensions
                .insert(EXT_COMPACT_LLM_PROVIDER.into(), provider.clone());
        }
        if let Some(ref model) = self.summarize_model {
            ctx.extensions
                .insert(EXT_COMPACT_LLM_MODEL.into(), model.clone());
        }

        Ok(RequestAction::Continue)
    }
}

/// Parameters for the auto-compaction body transformation.
pub struct CompactionParams<'a> {
    pub strategy: &'a str,
    pub threshold: usize,
    pub window_size: usize,
    pub preserve_system: bool,
    pub min_messages: usize,
    pub llm_provider: Option<&'a str>,
    pub llm_model: Option<&'a str>,
}

/// Data needed to perform an async LLM summarization call.
pub struct LlmCompactionData {
    pub provider: String,
    pub model: String,
    pub summary_text: String,
    pub system_messages: Vec<serde_json::Value>,
    pub to_keep: Vec<serde_json::Value>,
}

/// Result of a synchronous `compact_messages` call.
pub enum CompactionResult {
    /// No compaction was needed.
    NoOp,
    /// Body was compacted in-place (truncate or sliding_window strategy).
    Compacted,
    /// LLM strategy: an async summarization call is required.
    NeedsLlm(LlmCompactionData),
}

/// Separate messages into (system, non_system) depending on `preserve_system`.
fn separate_system_messages(
    messages: Vec<serde_json::Value>,
    preserve_system: bool,
) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    if preserve_system {
        let mut system = Vec::new();
        let non_system = messages
            .into_iter()
            .filter(|m| {
                if m.get("role").and_then(|r| r.as_str()) == Some("system") {
                    system.push(m.clone());
                    false
                } else {
                    true
                }
            })
            .collect();
        (system, non_system)
    } else {
        (Vec::new(), messages)
    }
}

/// Apply auto-compaction to an OpenAI-format request body.
///
/// Returns `CompactionResult::NoOp` when no action is needed,
/// `CompactionResult::Compacted` when the body was modified in-place, or
/// `CompactionResult::NeedsLlm` when the LLM strategy requires an async call.
pub fn compact_messages(
    body: &mut serde_json::Value,
    params: CompactionParams<'_>,
) -> CompactionResult {
    let CompactionParams {
        strategy,
        threshold,
        window_size,
        preserve_system,
        min_messages,
        llm_provider,
        llm_model,
    } = params;

    let messages = match body.get("messages").and_then(|m| m.as_array()) {
        Some(m) => m.clone(),
        None => return CompactionResult::NoOp,
    };

    match strategy {
        "truncate" => {
            let total_tokens = AutoCompactionPlugin::estimate_body_tokens(body);
            if total_tokens <= threshold {
                return CompactionResult::NoOp;
            }

            let (system_messages, non_system) = separate_system_messages(messages, preserve_system);

            let mut kept: Vec<serde_json::Value> = Vec::new();
            let mut tokens: usize = system_messages
                .iter()
                .map(AutoCompactionPlugin::estimate_message_tokens)
                .sum();

            for msg in non_system.iter().rev() {
                let msg_tokens = AutoCompactionPlugin::estimate_message_tokens(msg);
                if tokens + msg_tokens <= threshold || kept.len() < min_messages {
                    tokens += msg_tokens;
                    kept.push(msg.clone());
                }
            }
            kept.reverse();

            let dropped = non_system.len().saturating_sub(kept.len());
            if dropped == 0 {
                return CompactionResult::NoOp;
            }

            let mut final_messages = system_messages;
            final_messages.extend(kept);
            body["messages"] = serde_json::Value::Array(final_messages);

            tracing::info!(
                strategy = "truncate",
                dropped = dropped,
                estimated_tokens = tokens,
                threshold = threshold,
                "auto-compaction: truncated message history"
            );
            CompactionResult::Compacted
        }

        "sliding_window" => {
            if messages.len() <= window_size {
                return CompactionResult::NoOp;
            }

            let (system_messages, non_system) = separate_system_messages(messages, preserve_system);

            if non_system.len() <= window_size {
                return CompactionResult::NoOp;
            }

            let dropped = non_system.len() - window_size;
            let kept: Vec<serde_json::Value> = non_system.into_iter().skip(dropped).collect();

            let mut final_messages = system_messages;
            final_messages.extend(kept);
            body["messages"] = serde_json::Value::Array(final_messages);

            tracing::info!(
                strategy = "sliding_window",
                dropped = dropped,
                window_size = window_size,
                "auto-compaction: applied sliding window"
            );
            CompactionResult::Compacted
        }

        "llm" => {
            let total_tokens = AutoCompactionPlugin::estimate_body_tokens(body);
            if total_tokens <= threshold {
                return CompactionResult::NoOp;
            }

            let provider = match llm_provider {
                Some(p) => p,
                None => {
                    tracing::warn!(
                        "auto-compaction strategy=llm but no summarize_provider configured"
                    );
                    return CompactionResult::NoOp;
                }
            };
            let model = match llm_model {
                Some(m) => m,
                None => {
                    tracing::warn!(
                        "auto-compaction strategy=llm but no summarize_model configured"
                    );
                    return CompactionResult::NoOp;
                }
            };

            let (system_messages, non_system) = separate_system_messages(messages, preserve_system);

            if non_system.len() <= min_messages {
                return CompactionResult::NoOp;
            }

            let keep_count = (non_system.len() / 2).max(min_messages);
            let to_summarize = &non_system[..non_system.len() - keep_count];
            let to_keep = non_system[non_system.len() - keep_count..].to_vec();

            let summary_text: String = to_summarize
                .iter()
                .filter_map(|m| {
                    let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("unknown");
                    let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    if content.is_empty() {
                        None
                    } else {
                        Some(format!("{role}: {content}"))
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

            CompactionResult::NeedsLlm(LlmCompactionData {
                provider: provider.to_string(),
                model: model.to_string(),
                summary_text,
                system_messages,
                to_keep,
            })
        }

        _ => CompactionResult::NoOp,
    }
}

/// Complete the LLM compaction strategy by calling the summarization provider.
///
/// The `api_key` and `base_url` should come from the provider config; the caller
/// is responsible for resolving them (falling back to env vars if needed).
/// `client` should be a long-lived shared `reqwest::Client`.
///
/// Returns `true` if the body was modified.
pub async fn apply_llm_compaction(
    body: &mut serde_json::Value,
    data: LlmCompactionData,
    api_key: &str,
    base_url: &str,
    client: &reqwest::Client,
) -> bool {
    let LlmCompactionData {
        provider,
        model,
        summary_text,
        system_messages,
        to_keep,
    } = data;

    let request_body = serde_json::json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise conversation summarizer. Summarize the following conversation history into a brief paragraph capturing the key context and decisions. Be factual and concise."
            },
            {
                "role": "user",
                "content": format!("Summarize this conversation:\n\n{summary_text}")
            }
        ],
        "max_tokens": 512
    });

    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await;

    let summary = match resp {
        Ok(r) => match r.json::<serde_json::Value>().await {
            Ok(json) => json
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .map(String::from),
            Err(e) => {
                tracing::warn!(error = %e, "auto-compaction: failed to parse LLM response");
                None
            }
        },
        Err(e) => {
            tracing::warn!(error = %e, provider = provider.as_str(), model = model.as_str(), "auto-compaction: LLM summarization call failed");
            None
        }
    };

    let summary = match summary {
        Some(s) => s,
        None => return false,
    };

    let summary_message = serde_json::json!({
        "role": "assistant",
        "content": format!("[Summary of earlier conversation]\n{summary}")
    });

    let mut final_messages = system_messages;
    final_messages.push(summary_message);
    final_messages.extend(to_keep);
    body["messages"] = serde_json::Value::Array(final_messages);

    tracing::info!(
        strategy = "llm",
        provider = provider.as_str(),
        model = model.as_str(),
        "auto-compaction: summarized message history via LLM"
    );
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(extra: &[(&str, toml::Value)]) -> toml::Table {
        let mut config = toml::Table::new();
        config.insert("category".into(), "auto_compaction".into());
        for (k, v) in extra {
            config.insert((*k).into(), v.clone());
        }
        config
    }

    #[test]
    fn test_defaults() {
        let config = make_config(&[]);
        let plugin = AutoCompactionPlugin::from_config("compact", &config).unwrap();
        assert_eq!(plugin.threshold_tokens, 6000);
        assert_eq!(plugin.strategy, CompactionStrategy::Truncate);
        assert!(plugin.preserve_system);
        assert_eq!(plugin.min_messages, 2);
        assert_eq!(plugin.window_size, 20);
    }

    #[test]
    fn test_truncate_config() {
        let config = make_config(&[
            ("threshold_tokens", 4000i64.into()),
            ("strategy", "truncate".into()),
            ("preserve_system", true.into()),
            ("min_messages", 3i64.into()),
        ]);
        let plugin = AutoCompactionPlugin::from_config("compact", &config).unwrap();
        assert_eq!(plugin.threshold_tokens, 4000);
        assert_eq!(plugin.strategy, CompactionStrategy::Truncate);
        assert_eq!(plugin.min_messages, 3);
    }

    #[test]
    fn test_sliding_window_config() {
        let config = make_config(&[
            ("strategy", "sliding_window".into()),
            ("window_size", 10i64.into()),
        ]);
        let plugin = AutoCompactionPlugin::from_config("compact", &config).unwrap();
        assert_eq!(plugin.strategy, CompactionStrategy::SlidingWindow);
        assert_eq!(plugin.window_size, 10);
    }

    #[test]
    fn test_llm_config() {
        let config = make_config(&[
            ("strategy", "llm".into()),
            ("summarize_provider", "openai".into()),
            ("summarize_model", "gpt-4o-mini".into()),
        ]);
        let plugin = AutoCompactionPlugin::from_config("compact", &config).unwrap();
        assert_eq!(plugin.strategy, CompactionStrategy::Llm);
        assert_eq!(plugin.summarize_provider.as_deref(), Some("openai"));
        assert_eq!(plugin.summarize_model.as_deref(), Some("gpt-4o-mini"));
    }

    #[test]
    fn test_llm_missing_provider_errors() {
        let config = make_config(&[
            ("strategy", "llm".into()),
            ("summarize_model", "gpt-4o-mini".into()),
        ]);
        assert!(AutoCompactionPlugin::from_config("compact", &config).is_err());
    }

    #[test]
    fn test_llm_missing_model_errors() {
        let config = make_config(&[
            ("strategy", "llm".into()),
            ("summarize_provider", "openai".into()),
        ]);
        assert!(AutoCompactionPlugin::from_config("compact", &config).is_err());
    }

    #[test]
    fn test_invalid_strategy_errors() {
        let config = make_config(&[("strategy", "compress".into())]);
        assert!(AutoCompactionPlugin::from_config("compact", &config).is_err());
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(AutoCompactionPlugin::estimate_tokens("hello"), 2);
        assert_eq!(AutoCompactionPlugin::estimate_tokens("hello world"), 3);
        assert_eq!(AutoCompactionPlugin::estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_body_tokens() {
        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "Four."}
            ]
        });
        let tokens = AutoCompactionPlugin::estimate_body_tokens(&body);
        assert!(tokens > 0);
    }
}
