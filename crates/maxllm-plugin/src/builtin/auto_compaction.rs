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
