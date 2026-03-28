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
use std::time::Instant;

/// Extension key for recording the request start time (epoch nanos as string).
const EXT_REQUEST_START: &str = "_webhook_start_ns";
/// Extension key for upstream response status code.
const EXT_RESPONSE_STATUS: &str = "_webhook_status";

/// Webhook callback plugin.
///
/// Emits a structured JSON payload via `tracing::info!` at a configurable
/// target after each request completes. In production, a log shipper
/// (e.g., Fluent Bit, Vector) would forward these structured logs to an
/// HTTP endpoint, message queue, or analytics service.
pub struct WebhookPlugin {
    name: String,
    target: String,
    include_body: bool,
}

impl WebhookPlugin {
    pub fn from_config(name: &str, config: &toml::Table) -> Result<Self, PluginError> {
        let target = config
            .get("target")
            .and_then(|v| v.as_str())
            .unwrap_or("maxllm::webhook")
            .to_string();

        let include_body = config
            .get("include_body")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self {
            name: name.to_string(),
            target,
            include_body,
        })
    }
}

#[async_trait]
impl Plugin for WebhookPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_request(
        &self,
        _session: &mut Session,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<RequestAction> {
        // Record the request start time for latency calculation.
        let now = Instant::now();
        ctx.extensions.insert(
            EXT_REQUEST_START.into(),
            format!("{}", now.elapsed().as_nanos()),
        );
        // Store the raw instant as nanos since we need it in on_logging.
        // Since Instant cannot be serialized, we store a placeholder and
        // compute latency from the system time instead.
        ctx.extensions.insert(
            EXT_REQUEST_START.into(),
            format!(
                "{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            ),
        );

        Ok(RequestAction::Continue)
    }

    async fn on_response(
        &self,
        _session: &mut Session,
        upstream_response: &mut pingora::http::ResponseHeader,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<()> {
        // Capture the response status for the logging phase.
        ctx.extensions.insert(
            EXT_RESPONSE_STATUS.into(),
            upstream_response.status.as_u16().to_string(),
        );
        Ok(())
    }

    async fn on_logging(
        &self,
        _session: &mut Session,
        error: Option<&pingora::Error>,
        ctx: &mut PluginCtx,
    ) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        let start_ms: u128 = ctx
            .extensions
            .get(EXT_REQUEST_START)
            .and_then(|s| s.parse().ok())
            .unwrap_or(now_ms);

        let latency_ms = now_ms.saturating_sub(start_ms);

        let status: u16 = ctx
            .extensions
            .get(EXT_RESPONSE_STATUS)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let error_msg = error.map(|e| e.to_string());

        let payload = serde_json::json!({
            "event": "request_complete",
            "request_id": ctx.request_id,
            "model": ctx.model,
            "provider": ctx.provider_name,
            "latency_ms": latency_ms,
            "status": status,
            "client_id": ctx.client_id,
            "client_ip": ctx.client_ip,
            "route_path": ctx.route_path,
            "error": error_msg,
            "timestamp": now_ms,
        });

        // Emit the payload as a structured log line. The tracing target
        // allows log shippers to filter and route these events.
        // NOTE: We use a string target here; in production the tracing
        // subscriber would be configured to forward lines matching this
        // target to the webhook URL.
        tracing::info!(
            target: "maxllm::webhook",
            webhook_target = %self.target,
            include_body = %self.include_body,
            payload = %payload,
            "webhook event"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_config_defaults() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "webhook".into());

        let plugin = WebhookPlugin::from_config("hook", &config).unwrap();
        assert_eq!(plugin.target, "maxllm::webhook");
        assert!(!plugin.include_body);
    }

    #[test]
    fn test_from_config_custom() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "webhook".into());
        config.insert("target".into(), "custom::target".into());
        config.insert("include_body".into(), true.into());

        let plugin = WebhookPlugin::from_config("hook", &config).unwrap();
        assert_eq!(plugin.target, "custom::target");
        assert!(plugin.include_body);
    }
}
