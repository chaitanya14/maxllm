// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! External guardrail providers via HTTP.
//!
//! ## Generic Webhook
//! Calls any HTTP endpoint with a standard request/response format.
//! Works with custom guardrail services, Pillar Security, etc.
//!
//! ## Lakera
//! First-class integration with Lakera AI's prompt injection detection API.

use super::{Guardrail, GuardrailError, GuardrailInput, GuardrailOutput, GuardrailVerdict};
use async_trait::async_trait;
use maxllm_config::GuardrailConfig;
use std::time::Duration;

// ─── Generic Webhook ────────────────────────────────────────────────────

/// Generic HTTP webhook guardrail provider.
///
/// Sends a POST request to the configured endpoint with:
/// ```json
/// {
///   "content": "message text",
///   "model": "gpt-4",
///   "mode": "pre_call",
///   "metadata": { "client_id": "..." }
/// }
/// ```
///
/// Expects a response:
/// ```json
/// {
///   "action": "allow" | "block" | "modify",
///   "reason": "...",
///   "modified_content": "..."
/// }
/// ```
pub struct WebhookGuardrail {
    name: String,
    client: reqwest::Client,
    api_base: String,
    api_key: Option<String>,
    timeout: Duration,
    headers: Vec<(String, String)>,
}

impl WebhookGuardrail {
    pub fn from_config(config: &GuardrailConfig) -> Result<Self, GuardrailError> {
        let api_base = config
            .params
            .get("api_base")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GuardrailError::Config("webhook guardrail requires 'api_base'".into()))?
            .to_string();

        let api_key = config
            .params
            .get("api_key")
            .and_then(|v| v.as_str())
            .map(String::from);

        let timeout_secs = config
            .params
            .get("timeout_secs")
            .and_then(|v| v.as_integer())
            .unwrap_or(5) as u64;

        // Static headers from config
        let mut headers = Vec::new();
        if let Some(headers_table) = config.params.get("headers").and_then(|v| v.as_table()) {
            for (k, v) in headers_table {
                if let Some(val) = v.as_str() {
                    headers.push((k.clone(), val.to_string()));
                }
            }
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| GuardrailError::Config(format!("failed to create HTTP client: {e}")))?;

        Ok(Self {
            name: config.name.clone(),
            client,
            api_base,
            api_key,
            timeout: Duration::from_secs(timeout_secs),
            headers,
        })
    }

    async fn call_webhook(
        &self,
        content: &str,
        model: &str,
        mode: &str,
        client_id: Option<&str>,
    ) -> GuardrailVerdict {
        let payload = serde_json::json!({
            "content": content,
            "model": model,
            "mode": mode,
            "metadata": {
                "client_id": client_id.unwrap_or(""),
                "guardrail_name": &self.name,
            }
        });

        let mut req = self.client.post(&self.api_base).json(&payload);

        // Add auth header
        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }

        // Add static headers
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }

        match req.send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    tracing::warn!(
                        guardrail = self.name.as_str(),
                        status = resp.status().as_u16(),
                        "webhook guardrail returned non-200 status"
                    );
                    // Fail open — don't block on webhook errors
                    return GuardrailVerdict::Pass;
                }

                match resp.json::<serde_json::Value>().await {
                    Ok(body) => self.parse_response(&body),
                    Err(e) => {
                        tracing::warn!(
                            guardrail = self.name.as_str(),
                            error = %e,
                            "failed to parse webhook response"
                        );
                        GuardrailVerdict::Pass
                    }
                }
            }
            Err(e) => {
                if e.is_timeout() {
                    tracing::warn!(
                        guardrail = self.name.as_str(),
                        timeout_ms = self.timeout.as_millis() as u64,
                        "webhook guardrail timed out"
                    );
                } else {
                    tracing::warn!(
                        guardrail = self.name.as_str(),
                        error = %e,
                        "webhook guardrail request failed"
                    );
                }
                // Fail open
                GuardrailVerdict::Pass
            }
        }
    }

    fn parse_response(&self, body: &serde_json::Value) -> GuardrailVerdict {
        let action = body
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("allow");

        let reason = body
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("guardrail policy violation")
            .to_string();

        match action {
            "block" => GuardrailVerdict::Block {
                guardrail: self.name.clone(),
                reason,
            },
            "modify" => {
                let content = body
                    .get("modified_content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                GuardrailVerdict::Modify {
                    guardrail: self.name.clone(),
                    content,
                    reason,
                }
            }
            "log" => {
                let findings = body
                    .get("findings")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_else(|| vec![reason]);
                GuardrailVerdict::Log {
                    guardrail: self.name.clone(),
                    findings,
                }
            }
            _ => GuardrailVerdict::Pass, // "allow" or unknown
        }
    }
}

#[async_trait]
impl Guardrail for WebhookGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check_input(&self, input: &GuardrailInput<'_>) -> GuardrailVerdict {
        self.call_webhook(input.content, input.model, "pre_call", input.client_id)
            .await
    }

    async fn check_output(&self, output: &GuardrailOutput<'_>) -> GuardrailVerdict {
        self.call_webhook(output.content, output.model, "post_call", None)
            .await
    }
}

// ─── Lakera AI ──────────────────────────────────────────────────────────

/// Lakera AI prompt injection detection guardrail.
///
/// Calls the Lakera Guard API to detect prompt injection, jailbreaks,
/// and other LLM security threats.
///
/// API docs: https://platform.lakera.ai/docs
pub struct LakeraGuardrail {
    name: String,
    client: reqwest::Client,
    api_base: String,
    api_key: String,
}

impl LakeraGuardrail {
    pub fn from_config(config: &GuardrailConfig) -> Result<Self, GuardrailError> {
        let api_key = config
            .params
            .get("api_key")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GuardrailError::Config("lakera guardrail requires 'api_key'".into()))?
            .to_string();

        let api_base = config
            .params
            .get("api_base")
            .and_then(|v| v.as_str())
            .unwrap_or("https://api.lakera.ai/v2/guard")
            .to_string();

        let timeout_secs = config
            .params
            .get("timeout_secs")
            .and_then(|v| v.as_integer())
            .unwrap_or(5) as u64;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| GuardrailError::Config(format!("failed to create HTTP client: {e}")))?;

        Ok(Self {
            name: config.name.clone(),
            client,
            api_base,
            api_key,
        })
    }
}

#[async_trait]
impl Guardrail for LakeraGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check_input(&self, input: &GuardrailInput<'_>) -> GuardrailVerdict {
        // Lakera Guard API format
        let payload = serde_json::json!({
            "messages": [
                {"role": "user", "content": input.content}
            ]
        });

        let result = self
            .client
            .post(&self.api_base)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await;

        match result {
            Ok(resp) => {
                if !resp.status().is_success() {
                    tracing::warn!(
                        guardrail = self.name.as_str(),
                        status = resp.status().as_u16(),
                        "Lakera API returned non-200"
                    );
                    return GuardrailVerdict::Pass;
                }

                match resp.json::<serde_json::Value>().await {
                    Ok(body) => {
                        // Lakera v2 response format:
                        // { "results": [{ "categories": { "prompt_injection": true, ... }, "flagged": true }] }
                        let flagged = body
                            .get("results")
                            .and_then(|r| r.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|r| r.get("flagged"))
                            .and_then(|f| f.as_bool())
                            .unwrap_or(false);

                        if flagged {
                            // Collect which categories triggered
                            let categories: Vec<String> = body
                                .get("results")
                                .and_then(|r| r.as_array())
                                .and_then(|arr| arr.first())
                                .and_then(|r| r.get("categories"))
                                .and_then(|c| c.as_object())
                                .map(|obj| {
                                    obj.iter()
                                        .filter(|(_, v)| v.as_bool().unwrap_or(false))
                                        .map(|(k, _)| k.clone())
                                        .collect()
                                })
                                .unwrap_or_default();

                            GuardrailVerdict::Block {
                                guardrail: self.name.clone(),
                                reason: format!(
                                    "Lakera flagged content: {}",
                                    if categories.is_empty() {
                                        "policy violation".to_string()
                                    } else {
                                        categories.join(", ")
                                    }
                                ),
                            }
                        } else {
                            GuardrailVerdict::Pass
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            guardrail = self.name.as_str(),
                            error = %e,
                            "failed to parse Lakera response"
                        );
                        GuardrailVerdict::Pass
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    guardrail = self.name.as_str(),
                    error = %e,
                    "Lakera API request failed"
                );
                GuardrailVerdict::Pass // fail open
            }
        }
    }

    async fn check_output(&self, output: &GuardrailOutput<'_>) -> GuardrailVerdict {
        // Lakera can also check outputs for harmful content
        let payload = serde_json::json!({
            "messages": [
                {"role": "assistant", "content": output.content}
            ]
        });

        let result = self
            .client
            .post(&self.api_base)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(body) => {
                        let flagged = body
                            .get("results")
                            .and_then(|r| r.as_array())
                            .and_then(|arr| arr.first())
                            .and_then(|r| r.get("flagged"))
                            .and_then(|f| f.as_bool())
                            .unwrap_or(false);

                        if flagged {
                            GuardrailVerdict::Block {
                                guardrail: self.name.clone(),
                                reason: "Lakera flagged response content".into(),
                            }
                        } else {
                            GuardrailVerdict::Pass
                        }
                    }
                    _ => GuardrailVerdict::Pass,
                }
            }
            _ => GuardrailVerdict::Pass,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_config_requires_api_base() {
        let config = GuardrailConfig {
            name: "test".into(),
            provider: "webhook".into(),
            mode: "pre_call".into(),
            default_on: false,
            action: "block".into(),
            params: toml::Table::new(),
        };
        assert!(WebhookGuardrail::from_config(&config).is_err());
    }

    #[test]
    fn test_webhook_config_valid() {
        let mut params = toml::Table::new();
        params.insert("api_base".into(), "https://safety.example.com/check".into());
        params.insert("api_key".into(), "test-key".into());
        params.insert("timeout_secs".into(), toml::Value::Integer(10));

        let config = GuardrailConfig {
            name: "test".into(),
            provider: "webhook".into(),
            mode: "pre_call".into(),
            default_on: false,
            action: "block".into(),
            params,
        };
        assert!(WebhookGuardrail::from_config(&config).is_ok());
    }

    #[test]
    fn test_webhook_parse_response_block() {
        let mut params = toml::Table::new();
        params.insert("api_base".into(), "https://example.com".into());
        let config = GuardrailConfig {
            name: "test".into(),
            provider: "webhook".into(),
            mode: "pre_call".into(),
            default_on: false,
            action: "block".into(),
            params,
        };
        let webhook = WebhookGuardrail::from_config(&config).unwrap();

        let body = serde_json::json!({
            "action": "block",
            "reason": "content violation"
        });
        let verdict = webhook.parse_response(&body);
        assert!(matches!(verdict, GuardrailVerdict::Block { .. }));
    }

    #[test]
    fn test_webhook_parse_response_allow() {
        let mut params = toml::Table::new();
        params.insert("api_base".into(), "https://example.com".into());
        let config = GuardrailConfig {
            name: "test".into(),
            provider: "webhook".into(),
            mode: "pre_call".into(),
            default_on: false,
            action: "block".into(),
            params,
        };
        let webhook = WebhookGuardrail::from_config(&config).unwrap();

        let body = serde_json::json!({"action": "allow"});
        let verdict = webhook.parse_response(&body);
        assert!(matches!(verdict, GuardrailVerdict::Pass));
    }

    #[test]
    fn test_webhook_parse_response_modify() {
        let mut params = toml::Table::new();
        params.insert("api_base".into(), "https://example.com".into());
        let config = GuardrailConfig {
            name: "test".into(),
            provider: "webhook".into(),
            mode: "pre_call".into(),
            default_on: false,
            action: "block".into(),
            params,
        };
        let webhook = WebhookGuardrail::from_config(&config).unwrap();

        let body = serde_json::json!({
            "action": "modify",
            "reason": "PII redacted",
            "modified_content": "My email is [REDACTED]"
        });
        let verdict = webhook.parse_response(&body);
        match verdict {
            GuardrailVerdict::Modify { content, .. } => {
                assert_eq!(content, "My email is [REDACTED]");
            }
            _ => panic!("expected Modify"),
        }
    }

    #[test]
    fn test_lakera_config_requires_api_key() {
        let config = GuardrailConfig {
            name: "test".into(),
            provider: "lakera".into(),
            mode: "pre_call".into(),
            default_on: false,
            action: "block".into(),
            params: toml::Table::new(),
        };
        assert!(LakeraGuardrail::from_config(&config).is_err());
    }

    #[test]
    fn test_lakera_config_valid() {
        let mut params = toml::Table::new();
        params.insert("api_key".into(), "lk-test-key".into());

        let config = GuardrailConfig {
            name: "test".into(),
            provider: "lakera".into(),
            mode: "pre_call".into(),
            default_on: false,
            action: "block".into(),
            params,
        };
        assert!(LakeraGuardrail::from_config(&config).is_ok());
    }
}
