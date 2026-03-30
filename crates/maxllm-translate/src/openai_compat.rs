// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! Configurable passthrough translator for OpenAI-compatible providers.
//!
//! Many providers (Groq, Together AI, Fireworks, DeepInfra, Mistral, xAI,
//! DeepSeek, Ollama) expose an OpenAI-compatible chat completions API. This
//! module provides a single translator that can be configured with the
//! provider name, upstream path, and authentication style.

use crate::formats::OpenAIChatRequest;
use crate::{ProviderTranslator, StreamTranslator, TranslateError, TranslatedRequest};

/// Authentication style for an OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub enum AuthStyle {
    /// `Authorization: Bearer {key}`
    Bearer,
    /// Custom header: `{header_name}: {key}`
    ApiKey(String),
    /// Query parameter: `?{param}={key}` (appended by the gateway)
    QueryParam(String),
    /// No authentication (e.g., local Ollama).
    None,
}

/// A configurable passthrough translator for OpenAI-compatible providers.
///
/// Request and response bodies are passed through as-is (they already use
/// OpenAI format), with optional model override support. Only the upstream
/// path and authentication headers differ.
pub struct OpenAICompatTranslator {
    provider_name: String,
    upstream_path: String,
    auth_style: AuthStyle,
}

impl OpenAICompatTranslator {
    pub fn new(
        provider_name: impl Into<String>,
        upstream_path: impl Into<String>,
        auth_style: AuthStyle,
    ) -> Self {
        Self {
            provider_name: provider_name.into(),
            upstream_path: upstream_path.into(),
            auth_style,
        }
    }

    /// Pre-configured translator for Groq.
    pub fn groq() -> Self {
        Self::new("groq", "/openai/v1/chat/completions", AuthStyle::Bearer)
    }

    /// Pre-configured translator for Together AI.
    pub fn together() -> Self {
        Self::new("together", "/v1/chat/completions", AuthStyle::Bearer)
    }

    /// Pre-configured translator for Fireworks AI.
    pub fn fireworks() -> Self {
        Self::new(
            "fireworks",
            "/inference/v1/chat/completions",
            AuthStyle::Bearer,
        )
    }

    /// Pre-configured translator for DeepInfra.
    pub fn deepinfra() -> Self {
        Self::new(
            "deepinfra",
            "/v1/openai/chat/completions",
            AuthStyle::Bearer,
        )
    }

    /// Pre-configured translator for Mistral.
    pub fn mistral() -> Self {
        Self::new("mistral", "/v1/chat/completions", AuthStyle::Bearer)
    }

    /// Pre-configured translator for xAI (Grok).
    pub fn xai() -> Self {
        Self::new("xai", "/v1/chat/completions", AuthStyle::Bearer)
    }

    /// Pre-configured translator for DeepSeek.
    pub fn deepseek() -> Self {
        Self::new("deepseek", "/chat/completions", AuthStyle::Bearer)
    }

    /// Pre-configured translator for local Ollama.
    pub fn ollama() -> Self {
        Self::new("ollama", "/v1/chat/completions", AuthStyle::None)
    }
}

impl ProviderTranslator for OpenAICompatTranslator {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn translate_request(
        &self,
        body: &[u8],
        model_override: Option<&str>,
    ) -> Result<TranslatedRequest, TranslateError> {
        if let Some(model) = model_override {
            let mut req: OpenAIChatRequest = serde_json::from_slice(body)?;
            req.model = model.to_string();
            let is_streaming = req.stream.unwrap_or(false);
            Ok(TranslatedRequest {
                body: serde_json::to_vec(&req)?,
                is_streaming,
            })
        } else {
            let req: OpenAIChatRequest = serde_json::from_slice(body)?;
            let is_streaming = req.stream.unwrap_or(false);
            Ok(TranslatedRequest {
                body: body.to_vec(),
                is_streaming,
            })
        }
    }

    fn translate_response(&self, body: &[u8]) -> Result<Vec<u8>, TranslateError> {
        // Already in OpenAI format.
        Ok(body.to_vec())
    }

    fn streaming_translator(&self) -> Box<dyn StreamTranslator> {
        Box::new(OpenAICompatPassthroughStream)
    }

    fn upstream_path(&self) -> &str {
        &self.upstream_path
    }

    fn upstream_headers(&self, api_key: &str) -> Vec<(String, String)> {
        let mut headers = vec![("Content-Type".to_string(), "application/json".to_string())];
        match &self.auth_style {
            AuthStyle::Bearer => {
                headers.push(("Authorization".to_string(), format!("Bearer {api_key}")));
            }
            AuthStyle::ApiKey(header_name) => {
                headers.push((header_name.clone(), api_key.to_string()));
            }
            AuthStyle::QueryParam(_) => {
                // Query params are appended by the gateway to the URL, not as headers.
            }
            AuthStyle::None => {}
        }
        headers
    }
}

/// Passthrough SSE stream (already in OpenAI format).
struct OpenAICompatPassthroughStream;

impl StreamTranslator for OpenAICompatPassthroughStream {
    fn process_chunk(&mut self, data: &[u8], _end_of_stream: bool) -> Vec<u8> {
        data.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_request() {
        let body = br#"{"model":"llama-3.1-70b","messages":[{"role":"user","content":"Hi"}]}"#;
        let t = OpenAICompatTranslator::groq();
        let result = t.translate_request(body, None).unwrap();
        assert_eq!(result.body, body.to_vec());
        assert!(!result.is_streaming);
    }

    #[test]
    fn test_model_override() {
        let body = br#"{"model":"llama-3.1-70b","messages":[{"role":"user","content":"Hi"}]}"#;
        let t = OpenAICompatTranslator::together();
        let result = t.translate_request(body, Some("llama-3.1-8b")).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result.body).unwrap();
        assert_eq!(parsed["model"], "llama-3.1-8b");
    }

    #[test]
    fn test_bearer_headers() {
        let t = OpenAICompatTranslator::groq();
        let headers = t.upstream_headers("gsk_test");
        assert!(headers
            .iter()
            .any(|(k, v)| k == "Authorization" && v == "Bearer gsk_test"));
    }

    #[test]
    fn test_api_key_header() {
        let t = OpenAICompatTranslator::new(
            "custom",
            "/v1/chat",
            AuthStyle::ApiKey("X-Api-Key".to_string()),
        );
        let headers = t.upstream_headers("my-key");
        assert!(headers
            .iter()
            .any(|(k, v)| k == "X-Api-Key" && v == "my-key"));
        assert!(!headers.iter().any(|(k, _)| k == "Authorization"));
    }

    #[test]
    fn test_no_auth_headers() {
        let t = OpenAICompatTranslator::ollama();
        let headers = t.upstream_headers("");
        // Should only have Content-Type
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "Content-Type");
    }

    #[test]
    fn test_provider_names() {
        assert_eq!(OpenAICompatTranslator::groq().name(), "groq");
        assert_eq!(OpenAICompatTranslator::together().name(), "together");
        assert_eq!(OpenAICompatTranslator::fireworks().name(), "fireworks");
        assert_eq!(OpenAICompatTranslator::deepinfra().name(), "deepinfra");
        assert_eq!(OpenAICompatTranslator::mistral().name(), "mistral");
        assert_eq!(OpenAICompatTranslator::xai().name(), "xai");
        assert_eq!(OpenAICompatTranslator::deepseek().name(), "deepseek");
        assert_eq!(OpenAICompatTranslator::ollama().name(), "ollama");
    }

    #[test]
    fn test_upstream_paths() {
        assert_eq!(
            OpenAICompatTranslator::groq().upstream_path(),
            "/openai/v1/chat/completions"
        );
        assert_eq!(
            OpenAICompatTranslator::deepseek().upstream_path(),
            "/chat/completions"
        );
    }

    #[test]
    fn test_streaming_detection() {
        let body =
            br#"{"model":"mixtral","messages":[{"role":"user","content":"Hi"}],"stream":true}"#;
        let t = OpenAICompatTranslator::mistral();
        let result = t.translate_request(body, None).unwrap();
        assert!(result.is_streaming);
    }
}
