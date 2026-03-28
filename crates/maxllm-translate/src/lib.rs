// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

pub mod formats;
pub mod openai;
pub mod anthropic;
pub mod anthropic_stream;
pub mod gemini;
pub mod gemini_stream;
pub mod cohere;
pub mod cohere_stream;
pub mod openai_compat;
pub mod azure_openai;
pub mod bedrock;

#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("translation error: {0}")]
    Translation(String),
}

pub struct TranslatedRequest {
    pub body: Vec<u8>,
    pub is_streaming: bool,
}

/// Trait for translating requests/responses between OpenAI canonical format
/// and a provider's native format.
pub trait ProviderTranslator: Send + Sync {
    fn name(&self) -> &str;

    /// Downcast support for provider-specific operations.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Translate OpenAI-format request body → provider-native request bytes.
    fn translate_request(
        &self,
        body: &[u8],
        model_override: Option<&str>,
    ) -> Result<TranslatedRequest, TranslateError>;

    /// Translate provider-native response bytes → OpenAI-format response bytes.
    fn translate_response(&self, body: &[u8]) -> Result<Vec<u8>, TranslateError>;

    /// Create a streaming translator for SSE chunks (provider → OpenAI SSE).
    fn streaming_translator(&self) -> Box<dyn StreamTranslator>;

    /// Provider-specific upstream path (e.g. "/v1/messages" for Anthropic).
    fn upstream_path(&self) -> &str;

    /// Provider-specific headers to set on the upstream request.
    fn upstream_headers(&self, api_key: &str) -> Vec<(String, String)>;
}

/// Incremental SSE stream translator.
pub trait StreamTranslator: Send {
    fn process_chunk(&mut self, data: &[u8], end_of_stream: bool) -> Vec<u8>;
}
