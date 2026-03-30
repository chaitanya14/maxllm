// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! Translator between OpenAI canonical format and Cohere v2 Chat API.

use crate::cohere_stream::CohereToOpenAIStream;
use crate::formats::*;
use crate::{ProviderTranslator, StreamTranslator, TranslateError, TranslatedRequest};
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

/// Translator for the Cohere v2 Chat API.
pub struct CohereTranslator;

impl ProviderTranslator for CohereTranslator {
    fn name(&self) -> &str {
        "cohere"
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
        let cohere_req = openai_to_cohere(&req, model_override)?;
        let is_streaming = req.stream.unwrap_or(false);

        Ok(TranslatedRequest {
            body: serde_json::to_vec(&cohere_req)?,
            is_streaming,
        })
    }

    fn translate_response(&self, body: &[u8]) -> Result<Vec<u8>, TranslateError> {
        let resp: CohereResponse = serde_json::from_slice(body)?;
        let openai_resp = cohere_to_openai_response(&resp);
        Ok(serde_json::to_vec(&openai_resp)?)
    }

    fn streaming_translator(&self) -> Box<dyn StreamTranslator> {
        Box::new(CohereToOpenAIStream::new())
    }

    fn upstream_path(&self) -> &str {
        "/v2/chat"
    }

    fn upstream_headers(&self, api_key: &str) -> Vec<(String, String)> {
        vec![
            ("Authorization".to_string(), format!("Bearer {api_key}")),
            ("Content-Type".to_string(), "application/json".to_string()),
        ]
    }
}

/// Strip `@provider/` prefix if present.
fn normalize_cohere_model(model: &str) -> String {
    if let Some(pos) = model.find('/') {
        if model.starts_with('@') {
            return model[pos + 1..].to_string();
        }
    }
    model.to_string()
}

fn openai_to_cohere(
    req: &OpenAIChatRequest,
    model_override: Option<&str>,
) -> Result<CohereRequest, TranslateError> {
    let model = normalize_cohere_model(model_override.unwrap_or(&req.model));

    let mut messages: Vec<CohereMessage> = Vec::new();

    for msg in &req.messages {
        if msg.role == "system" {
            let text = match &msg.content {
                Some(Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => String::new(),
            };
            messages.push(CohereMessage {
                role: "system".to_string(),
                content: Some(Value::String(text)),
                tool_calls: None,
                tool_call_id: None,
            });
            continue;
        }

        if msg.role == "tool" {
            messages.push(CohereMessage {
                role: "tool".to_string(),
                content: msg.content.clone(),
                tool_calls: None,
                tool_call_id: msg.tool_call_id.clone(),
            });
            continue;
        }

        if msg.role == "assistant" {
            if let Some(tool_calls) = &msg.tool_calls {
                let cohere_tcs: Vec<CohereToolCall> = tool_calls
                    .iter()
                    .map(|tc| CohereToolCall {
                        id: tc.id.clone(),
                        call_type: "function".to_string(),
                        function: CohereFunctionCall {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    })
                    .collect();
                messages.push(CohereMessage {
                    role: "assistant".to_string(),
                    content: msg.content.clone(),
                    tool_calls: Some(cohere_tcs),
                    tool_call_id: None,
                });
                continue;
            }
        }

        messages.push(CohereMessage {
            role: msg.role.clone(),
            content: msg.content.clone(),
            tool_calls: None,
            tool_call_id: msg.tool_call_id.clone(),
        });
    }

    // Translate tools
    let tools = req.tools.as_ref().map(|openai_tools| {
        openai_tools
            .iter()
            .map(|t| CohereTool {
                tool_type: t.tool_type.clone(),
                function: CohereFunction {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: t.function.parameters.clone(),
                },
            })
            .collect()
    });

    let is_streaming = req.stream.unwrap_or(false);

    Ok(CohereRequest {
        model,
        messages,
        stream: if is_streaming { Some(true) } else { None },
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop.clone(),
        tools,
    })
}

fn cohere_to_openai_response(resp: &CohereResponse) -> OpenAIChatResponse {
    // Extract text content
    let content = resp.message.content.as_ref().and_then(|blocks| {
        let texts: Vec<&str> = blocks
            .iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text.as_deref())
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(Value::String(texts.join("")))
        }
    });

    // Extract tool calls
    let tool_calls = resp.message.tool_calls.as_ref().map(|tcs| {
        tcs.iter()
            .map(|tc| OpenAIToolCall {
                id: tc.id.clone(),
                call_type: "function".to_string(),
                function: OpenAIFunctionCall {
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                },
            })
            .collect::<Vec<_>>()
    });

    let finish_reason = resp.finish_reason.as_deref().map(|fr| {
        match fr {
            "COMPLETE" => "stop",
            "MAX_TOKENS" => "length",
            "TOOL_CALL" => "tool_calls",
            "ERROR" => "stop",
            other => other,
        }
        .to_string()
    });

    let message = OpenAIMessage {
        role: "assistant".to_string(),
        content,
        tool_calls: tool_calls.filter(|v| !v.is_empty()),
        tool_call_id: None,
        extra: serde_json::Map::new(),
    };

    let usage = resp.usage.as_ref().map(|u| {
        let input = u.tokens.as_ref().and_then(|t| t.input_tokens).unwrap_or(0);
        let output = u.tokens.as_ref().and_then(|t| t.output_tokens).unwrap_or(0);
        OpenAIUsage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: input + output,
        }
    });

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    OpenAIChatResponse {
        id: resp
            .id
            .clone()
            .unwrap_or_else(|| format!("chatcmpl-cohere-{created}")),
        object: "chat.completion".to_string(),
        created,
        model: resp
            .model
            .clone()
            .unwrap_or_else(|| "command-r-plus".to_string()),
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
            "model": "command-r-plus",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 512
        }))
        .unwrap();

        let t = CohereTranslator;
        let result = t.translate_request(&body, None).unwrap();
        let parsed: CohereRequest = serde_json::from_slice(&result.body).unwrap();

        assert_eq!(parsed.model, "command-r-plus");
        assert_eq!(parsed.messages.len(), 2);
        assert_eq!(parsed.messages[0].role, "system");
        assert_eq!(parsed.messages[1].role, "user");
        assert_eq!(parsed.max_tokens, Some(512));
    }

    #[test]
    fn test_response_translation() {
        let resp = serde_json::to_vec(&serde_json::json!({
            "id": "cohere-123",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
            },
            "finish_reason": "COMPLETE",
            "usage": {
                "tokens": {
                    "input_tokens": 10,
                    "output_tokens": 3
                }
            }
        }))
        .unwrap();

        let t = CohereTranslator;
        let result = t.translate_response(&resp).unwrap();
        let parsed: OpenAIChatResponse = serde_json::from_slice(&result).unwrap();

        assert_eq!(parsed.id, "cohere-123");
        assert_eq!(parsed.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(
            parsed.choices[0].message.content,
            Some(Value::String("Hello!".to_string()))
        );
        let usage = parsed.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 3);
    }

    #[test]
    fn test_model_override() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "command-r-plus",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .unwrap();

        let t = CohereTranslator;
        let result = t.translate_request(&body, Some("command-r")).unwrap();
        let parsed: CohereRequest = serde_json::from_slice(&result.body).unwrap();
        assert_eq!(parsed.model, "command-r");
    }

    #[test]
    fn test_streaming_flag() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "command-r-plus",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .unwrap();

        let t = CohereTranslator;
        let result = t.translate_request(&body, None).unwrap();
        assert!(result.is_streaming);
        let parsed: CohereRequest = serde_json::from_slice(&result.body).unwrap();
        assert_eq!(parsed.stream, Some(true));
    }

    #[test]
    fn test_headers() {
        let t = CohereTranslator;
        let headers = t.upstream_headers("co-test-key");
        assert!(headers
            .iter()
            .any(|(k, v)| k == "Authorization" && v == "Bearer co-test-key"));
    }

    #[test]
    fn test_tool_call_response() {
        let resp = serde_json::to_vec(&serde_json::json!({
            "id": "cohere-456",
            "message": {
                "role": "assistant",
                "content": [],
                "tool_calls": [{
                    "id": "tc_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"London\"}"
                    }
                }]
            },
            "finish_reason": "TOOL_CALL"
        }))
        .unwrap();

        let t = CohereTranslator;
        let result = t.translate_response(&resp).unwrap();
        let parsed: OpenAIChatResponse = serde_json::from_slice(&result).unwrap();

        assert_eq!(
            parsed.choices[0].finish_reason,
            Some("tool_calls".to_string())
        );
        let tool_calls = parsed.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }
}
