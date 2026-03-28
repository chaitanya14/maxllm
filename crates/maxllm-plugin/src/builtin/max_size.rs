// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use crate::factory::PluginError;
use crate::{HttpResponse, Plugin, PluginCtx, RequestAction};
use async_trait::async_trait;
use pingora::proxy::Session;

/// Extension key storing the max_tokens limit for gateway-level enforcement.
const EXT_MAX_TOKENS: &str = "max_tokens_limit";

/// Request size limit plugin.
///
/// Enforces maximum request body size based on the Content-Length header
/// and optionally stores a max_tokens limit in extensions for the gateway
/// to enforce after parsing the request body.
pub struct MaxSizePlugin {
    name: String,
    max_body_bytes: usize,
    max_tokens: Option<u64>,
}

impl MaxSizePlugin {
    pub fn from_config(name: &str, config: &toml::Table) -> Result<Self, PluginError> {
        let max_body_bytes = config
            .get("max_body_bytes")
            .and_then(|v| v.as_integer())
            .unwrap_or(1_048_576) as usize; // Default 1MB

        let max_tokens = config
            .get("max_tokens")
            .and_then(|v| v.as_integer())
            .map(|v| v as u64);

        Ok(Self {
            name: name.to_string(),
            max_body_bytes,
            max_tokens,
        })
    }

    /// Returns the configured max body size in bytes.
    pub fn max_body_bytes(&self) -> usize {
        self.max_body_bytes
    }

    /// Returns the configured max tokens limit, if any.
    pub fn max_tokens(&self) -> Option<u64> {
        self.max_tokens
    }
}

#[async_trait]
impl Plugin for MaxSizePlugin {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_request(
        &self,
        session: &mut Session,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<RequestAction> {
        // Check Content-Length header if present.
        if let Some(content_length) = session
            .req_header()
            .headers
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<usize>().ok())
        {
            if content_length > self.max_body_bytes {
                tracing::warn!(
                    plugin = self.name.as_str(),
                    content_length = content_length,
                    max_bytes = self.max_body_bytes,
                    "request body exceeds maximum size"
                );
                return Ok(RequestAction::Respond(HttpResponse::json_error(
                    413,
                    &format!(
                        "Request body too large: {} bytes exceeds limit of {} bytes",
                        content_length, self.max_body_bytes
                    ),
                )));
            }
        }

        // Store max_tokens in extensions for gateway-level enforcement.
        if let Some(max_tokens) = self.max_tokens {
            ctx.extensions
                .insert(EXT_MAX_TOKENS.into(), max_tokens.to_string());
        }

        Ok(RequestAction::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_config_defaults() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "max_size".into());

        let plugin = MaxSizePlugin::from_config("size", &config).unwrap();
        assert_eq!(plugin.max_body_bytes, 1_048_576);
        assert!(plugin.max_tokens.is_none());
    }

    #[test]
    fn test_from_config_custom() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "max_size".into());
        config.insert("max_body_bytes".into(), 524_288i64.into());
        config.insert("max_tokens".into(), 128_000i64.into());

        let plugin = MaxSizePlugin::from_config("size", &config).unwrap();
        assert_eq!(plugin.max_body_bytes, 524_288);
        assert_eq!(plugin.max_tokens, Some(128_000));
    }

    #[test]
    fn test_accessors() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "max_size".into());
        config.insert("max_body_bytes".into(), 2_000_000i64.into());
        config.insert("max_tokens".into(), 64_000i64.into());

        let plugin = MaxSizePlugin::from_config("size", &config).unwrap();
        assert_eq!(plugin.max_body_bytes(), 2_000_000);
        assert_eq!(plugin.max_tokens(), Some(64_000));
    }
}
