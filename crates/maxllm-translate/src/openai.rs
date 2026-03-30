// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use crate::formats::OpenAIChatRequest;
use crate::{ProviderTranslator, StreamTranslator, TranslateError, TranslatedRequest};

/// Pass-through translator for OpenAI (canonical format).
pub struct OpenAITranslator;

impl ProviderTranslator for OpenAITranslator {
    fn name(&self) -> &str {
        "openai"
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
        // Already in OpenAI format — pass through
        Ok(body.to_vec())
    }

    fn streaming_translator(&self) -> Box<dyn StreamTranslator> {
        Box::new(OpenAIPassthroughStream)
    }

    fn upstream_path(&self) -> &str {
        "/v1/chat/completions"
    }

    fn upstream_headers(&self, api_key: &str) -> Vec<(String, String)> {
        vec![
            ("Authorization".to_string(), format!("Bearer {api_key}")),
            ("Content-Type".to_string(), "application/json".to_string()),
        ]
    }
}

/// Pass-through stream — SSE is already in OpenAI format.
struct OpenAIPassthroughStream;

impl StreamTranslator for OpenAIPassthroughStream {
    fn process_chunk(&mut self, data: &[u8], _end_of_stream: bool) -> Vec<u8> {
        data.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_request() {
        let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}"#;
        let t = OpenAITranslator;
        let result = t.translate_request(body, None).unwrap();
        assert_eq!(result.body, body.to_vec());
        assert!(!result.is_streaming);
    }

    #[test]
    fn test_model_override() {
        let body = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}"#;
        let t = OpenAITranslator;
        let result = t.translate_request(body, Some("gpt-4o-mini")).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&result.body).unwrap();
        assert_eq!(parsed["model"], "gpt-4o-mini");
    }

    #[test]
    fn test_streaming_detection() {
        let body =
            br#"{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}],"stream":true}"#;
        let t = OpenAITranslator;
        let result = t.translate_request(body, None).unwrap();
        assert!(result.is_streaming);
    }

    #[test]
    fn test_headers() {
        let t = OpenAITranslator;
        let headers = t.upstream_headers("sk-test");
        assert_eq!(
            headers[0],
            ("Authorization".to_string(), "Bearer sk-test".to_string())
        );
    }
}
