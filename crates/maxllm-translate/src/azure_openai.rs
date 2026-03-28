// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! Translator for Azure OpenAI Service.
//!
//! Azure OpenAI uses the same request/response format as OpenAI, but with:
//! - A different endpoint pattern: `/openai/deployments/{deployment}/chat/completions?api-version={version}`
//! - `api-key` header instead of `Authorization: Bearer`
//! - The model field in the request body is ignored (deployment determines the model)

use crate::formats::OpenAIChatRequest;
use crate::{ProviderTranslator, StreamTranslator, TranslateError, TranslatedRequest};

const DEFAULT_API_VERSION: &str = "2024-02-01";

/// Translator for Azure OpenAI Service.
///
/// The deployment name and API version are embedded in the upstream path.
/// Request/response bodies are identical to OpenAI format.
pub struct AzureOpenAITranslator {
    deployment: String,
    api_version: String,
    /// Pre-computed path string for the trait's `&str` return.
    path: String,
}

impl AzureOpenAITranslator {
    pub fn new(deployment: impl Into<String>, api_version: impl Into<String>) -> Self {
        let deployment = deployment.into();
        let api_version = api_version.into();
        let path = format!(
            "/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        );
        Self {
            deployment,
            api_version,
            path,
        }
    }

    /// Create with the default API version (2024-02-01).
    pub fn with_deployment(deployment: impl Into<String>) -> Self {
        Self::new(deployment, DEFAULT_API_VERSION)
    }

    pub fn deployment(&self) -> &str {
        &self.deployment
    }

    pub fn api_version(&self) -> &str {
        &self.api_version
    }
}

impl ProviderTranslator for AzureOpenAITranslator {
    fn name(&self) -> &str {
        "azure-openai"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn translate_request(
        &self,
        body: &[u8],
        model_override: Option<&str>,
    ) -> Result<TranslatedRequest, TranslateError> {
        // Azure ignores the model field (deployment determines model), but we
        // still parse to detect streaming and optionally override model.
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
        // Azure returns OpenAI-compatible responses.
        Ok(body.to_vec())
    }

    fn streaming_translator(&self) -> Box<dyn StreamTranslator> {
        Box::new(AzureOpenAIPassthroughStream)
    }

    fn upstream_path(&self) -> &str {
        &self.path
    }

    fn upstream_headers(&self, api_key: &str) -> Vec<(String, String)> {
        vec![
            ("api-key".to_string(), api_key.to_string()),
            ("Content-Type".to_string(), "application/json".to_string()),
        ]
    }
}

/// Passthrough SSE stream (Azure uses OpenAI SSE format).
struct AzureOpenAIPassthroughStream;

impl StreamTranslator for AzureOpenAIPassthroughStream {
    fn process_chunk(&mut self, data: &[u8], _end_of_stream: bool) -> Vec<u8> {
        data.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upstream_path() {
        let t = AzureOpenAITranslator::new("gpt-4o-deploy", "2024-06-01");
        assert_eq!(
            t.upstream_path(),
            "/openai/deployments/gpt-4o-deploy/chat/completions?api-version=2024-06-01"
        );
    }

    #[test]
    fn test_default_api_version() {
        let t = AzureOpenAITranslator::with_deployment("my-deploy");
        assert!(t.upstream_path().contains("api-version=2024-02-01"));
        assert_eq!(t.deployment(), "my-deploy");
        assert_eq!(t.api_version(), "2024-02-01");
    }

    #[test]
    fn test_headers_use_api_key() {
        let t = AzureOpenAITranslator::with_deployment("deploy");
        let headers = t.upstream_headers("my-azure-key");
        assert!(headers
            .iter()
            .any(|(k, v)| k == "api-key" && v == "my-azure-key"));
        // Should NOT have Authorization header
        assert!(!headers.iter().any(|(k, _)| k == "Authorization"));
    }

    #[test]
    fn test_passthrough_request() {
        let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}"#;
        let t = AzureOpenAITranslator::with_deployment("gpt4o");
        let result = t.translate_request(body, None).unwrap();
        assert_eq!(result.body, body.to_vec());
        assert!(!result.is_streaming);
    }

    #[test]
    fn test_model_override() {
        let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}"#;
        let t = AzureOpenAITranslator::with_deployment("gpt4o");
        let result = t.translate_request(body, Some("gpt-4o-mini")).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result.body).unwrap();
        assert_eq!(parsed["model"], "gpt-4o-mini");
    }

    #[test]
    fn test_streaming_detection() {
        let body =
            br#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}],"stream":true}"#;
        let t = AzureOpenAITranslator::with_deployment("gpt4o");
        let result = t.translate_request(body, None).unwrap();
        assert!(result.is_streaming);
    }

    #[test]
    fn test_name() {
        let t = AzureOpenAITranslator::with_deployment("x");
        assert_eq!(t.name(), "azure-openai");
    }
}
