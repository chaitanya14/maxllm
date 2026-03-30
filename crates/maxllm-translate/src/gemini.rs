// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! Translator between OpenAI canonical format and Google Gemini API.

use crate::formats::*;
use crate::gemini_stream::GeminiToOpenAIStream;
use crate::{ProviderTranslator, StreamTranslator, TranslateError, TranslatedRequest};
use serde_json::Value;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_MODEL: &str = "gemini-2.0-flash";

/// Translator for the Google Gemini (Generative Language) API.
///
/// Gemini embeds the model name into the URL path, so the translator stores
/// the most recently seen model in a mutex. Because `upstream_path` returns
/// `&str`, it returns a sensible default; the gateway is expected to construct
/// the final path using `last_model()` when needed.
pub struct GeminiTranslator {
    /// Buffers the last model seen in `translate_request` so the gateway can
    /// construct the correct path (`/v1beta/models/{model}:generateContent`).
    last_model: Mutex<String>,
    /// Pre-formatted default path returned by the trait method.
    default_path: String,
}

impl GeminiTranslator {
    pub fn new() -> Self {
        let default_path = format!("/v1beta/models/{DEFAULT_MODEL}:generateContent");
        Self {
            last_model: Mutex::new(DEFAULT_MODEL.to_string()),
            default_path,
        }
    }

    /// Returns the last model name seen during request translation. The gateway
    /// can use this to build the correct upstream path.
    pub fn last_model(&self) -> String {
        self.last_model.lock().unwrap().clone()
    }
}

impl Default for GeminiTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderTranslator for GeminiTranslator {
    fn name(&self) -> &str {
        "gemini"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn translate_request(
        &self,
        body: &[u8],
        model_override: Option<&str>,
    ) -> Result<TranslatedRequest, TranslateError> {
        let req: OpenAIChatRequest = serde_json::from_slice(body)?;
        let model = model_override.unwrap_or(&req.model);

        // Store the model for path construction.
        if let Ok(mut m) = self.last_model.lock() {
            *m = normalize_gemini_model(model);
        }

        let gemini_req = openai_to_gemini(&req, model)?;
        let is_streaming = req.stream.unwrap_or(false);

        Ok(TranslatedRequest {
            body: serde_json::to_vec(&gemini_req)?,
            is_streaming,
        })
    }

    fn translate_response(&self, body: &[u8]) -> Result<Vec<u8>, TranslateError> {
        let resp: GeminiResponse = serde_json::from_slice(body)?;
        let openai_resp = gemini_to_openai_response(&resp);
        Ok(serde_json::to_vec(&openai_resp)?)
    }

    fn streaming_translator(&self) -> Box<dyn StreamTranslator> {
        Box::new(GeminiToOpenAIStream::new())
    }

    fn upstream_path(&self) -> &str {
        &self.default_path
    }

    fn upstream_headers(&self, _api_key: &str) -> Vec<(String, String)> {
        // Gemini uses query-param auth (?key=), handled via upstream_path.
        // No auth headers needed here.
        vec![("Content-Type".to_string(), "application/json".to_string())]
    }
}

/// Strip `@provider/` prefix if present.
fn normalize_gemini_model(model: &str) -> String {
    if let Some(pos) = model.find('/') {
        if model.starts_with('@') {
            return model[pos + 1..].to_string();
        }
    }
    model.to_string()
}

fn openai_to_gemini(req: &OpenAIChatRequest, model: &str) -> Result<GeminiRequest, TranslateError> {
    let mut system_instruction: Option<GeminiContent> = None;
    let mut contents: Vec<GeminiContent> = Vec::new();

    for msg in &req.messages {
        if msg.role == "system" {
            let text = content_to_text(&msg.content);
            system_instruction = Some(GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart {
                    text: Some(text),
                    function_call: None,
                    function_response: None,
                }],
            });
            continue;
        }

        if msg.role == "tool" {
            // Tool result message -> functionResponse part
            let name = msg
                .extra
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let response_text = content_to_text(&msg.content);
            let response_val: Value =
                serde_json::from_str(&response_text).unwrap_or(Value::String(response_text));
            contents.push(GeminiContent {
                role: "user".to_string(),
                parts: vec![GeminiPart {
                    text: None,
                    function_call: None,
                    function_response: Some(GeminiFunctionResponse {
                        name,
                        response: response_val,
                    }),
                }],
            });
            continue;
        }

        let gemini_role = if msg.role == "assistant" {
            "model".to_string()
        } else {
            msg.role.clone()
        };

        if let Some(tool_calls) = &msg.tool_calls {
            // Assistant message with tool calls -> functionCall parts
            let mut parts: Vec<GeminiPart> = Vec::new();
            if let Some(text) = msg.content.as_ref().and_then(|c| c.as_str()) {
                if !text.is_empty() {
                    parts.push(GeminiPart {
                        text: Some(text.to_string()),
                        function_call: None,
                        function_response: None,
                    });
                }
            }
            for tc in tool_calls {
                let args: Value = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(Value::Object(serde_json::Map::new()));
                parts.push(GeminiPart {
                    text: None,
                    function_call: Some(GeminiFunctionCall {
                        name: tc.function.name.clone(),
                        args,
                    }),
                    function_response: None,
                });
            }
            contents.push(GeminiContent {
                role: gemini_role,
                parts,
            });
            continue;
        }

        let text = content_to_text(&msg.content);
        contents.push(GeminiContent {
            role: gemini_role,
            parts: vec![GeminiPart {
                text: Some(text),
                function_call: None,
                function_response: None,
            }],
        });
    }

    // Build generation config
    let generation_config = GeminiGenerationConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        max_output_tokens: req.max_tokens,
        stop_sequences: req.stop.clone(),
    };
    // Only include config if at least one field is set.
    let has_config = generation_config.temperature.is_some()
        || generation_config.top_p.is_some()
        || generation_config.max_output_tokens.is_some()
        || generation_config.stop_sequences.is_some();
    let generation_config_opt = if has_config {
        Some(generation_config)
    } else {
        drop(generation_config);
        None
    };

    // Translate tools
    let tools = req.tools.as_ref().map(|openai_tools| {
        let declarations: Vec<GeminiFunctionDeclaration> = openai_tools
            .iter()
            .map(|t| GeminiFunctionDeclaration {
                name: t.function.name.clone(),
                description: t.function.description.clone().unwrap_or_default(),
                parameters: t.function.parameters.clone(),
            })
            .collect();
        vec![GeminiToolConfig {
            function_declarations: declarations,
        }]
    });

    let _ = model; // model is stored via Mutex, not in the body for Gemini

    Ok(GeminiRequest {
        contents,
        system_instruction,
        generation_config: generation_config_opt,
        tools,
    })
}

fn content_to_text(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

fn gemini_to_openai_response(resp: &GeminiResponse) -> OpenAIChatResponse {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();
    let mut finish_reason = None;
    let mut tc_index: u32 = 0;

    if let Some(candidates) = &resp.candidates {
        for candidate in candidates {
            if let Some(content) = &candidate.content {
                for part in &content.parts {
                    if let Some(text) = &part.text {
                        text_parts.push(text.clone());
                    }
                    if let Some(fc) = &part.function_call {
                        tool_calls.push(OpenAIToolCall {
                            id: format!("call_{tc_index}"),
                            call_type: "function".to_string(),
                            function: OpenAIFunctionCall {
                                name: fc.name.clone(),
                                arguments: fc.args.to_string(),
                            },
                        });
                        tc_index += 1;
                    }
                }
            }
            finish_reason = candidate.finish_reason.as_deref().map(|fr| {
                match fr {
                    "STOP" => "stop",
                    "MAX_TOKENS" => "length",
                    "SAFETY" => "content_filter",
                    "RECITATION" => "content_filter",
                    _ => fr,
                }
                .to_string()
            });
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(Value::String(text_parts.join("")))
    };

    let message = OpenAIMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
        extra: serde_json::Map::new(),
    };

    let usage = resp.usage_metadata.as_ref().map(|u| OpenAIUsage {
        prompt_tokens: u.prompt_token_count.unwrap_or(0),
        completion_tokens: u.candidates_token_count.unwrap_or(0),
        total_tokens: u.prompt_token_count.unwrap_or(0) + u.candidates_token_count.unwrap_or(0),
    });

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    OpenAIChatResponse {
        id: format!("chatcmpl-gemini-{created}"),
        object: "chat.completion".to_string(),
        created,
        model: "gemini".to_string(),
        choices: vec![OpenAIChoice {
            index: 0,
            message,
            finish_reason,
        }],
        usage,
        extra: serde_json::Map::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_request_translation() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }))
        .unwrap();

        let t = GeminiTranslator::new();
        let result = t.translate_request(&body, None).unwrap();
        let parsed: GeminiRequest = serde_json::from_slice(&result.body).unwrap();

        assert!(parsed.system_instruction.is_some());
        let sys = parsed.system_instruction.unwrap();
        assert_eq!(sys.parts[0].text.as_deref(), Some("You are helpful."));
        assert_eq!(parsed.contents.len(), 1);
        assert_eq!(parsed.contents[0].role, "user");
        assert_eq!(parsed.contents[0].parts[0].text.as_deref(), Some("Hello!"));

        let config = parsed.generation_config.unwrap();
        assert_eq!(config.max_output_tokens, Some(1024));
        assert_eq!(config.temperature, Some(0.7));
    }

    #[test]
    fn test_response_translation() {
        let resp = serde_json::to_vec(&serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello there!"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        }))
        .unwrap();

        let t = GeminiTranslator::new();
        let result = t.translate_response(&resp).unwrap();
        let parsed: OpenAIChatResponse = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed.object, "chat.completion");
        assert_eq!(parsed.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(
            parsed.choices[0].message.content,
            Some(Value::String("Hello there!".to_string()))
        );
        let usage = parsed.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn test_tool_call_translation() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }
            }]
        }))
        .unwrap();

        let t = GeminiTranslator::new();
        let result = t.translate_request(&body, None).unwrap();
        let parsed: GeminiRequest = serde_json::from_slice(&result.body).unwrap();

        let tools = parsed.tools.unwrap();
        assert_eq!(tools[0].function_declarations[0].name, "get_weather");
    }

    #[test]
    fn test_role_mapping() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"}
            ]
        }))
        .unwrap();

        let t = GeminiTranslator::new();
        let result = t.translate_request(&body, None).unwrap();
        let parsed: GeminiRequest = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed.contents[0].role, "user");
        assert_eq!(parsed.contents[1].role, "model");
        assert_eq!(parsed.contents[2].role, "user");
    }

    #[test]
    fn test_model_stored() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "gemini-1.5-pro",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .unwrap();

        let t = GeminiTranslator::new();
        t.translate_request(&body, None).unwrap();
        assert_eq!(t.last_model(), "gemini-1.5-pro");
    }

    #[test]
    fn test_headers() {
        let t = GeminiTranslator::new();
        let headers = t.upstream_headers("my-api-key");
        // Gemini uses query-param auth, no Authorization header
        assert!(!headers.iter().any(|(k, _)| k == "Authorization"));
        assert!(headers
            .iter()
            .any(|(k, v)| k == "Content-Type" && v == "application/json"));
    }
}
