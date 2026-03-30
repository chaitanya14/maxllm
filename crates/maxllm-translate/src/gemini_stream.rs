// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! SSE streaming translation from Gemini to OpenAI format.
//!
//! Gemini streaming (with `alt=sse`) sends SSE events where each `data:` line
//! contains a JSON object with the same structure as the non-streaming response
//! (i.e., `{"candidates": [...], "usageMetadata": {...}}`).

use crate::StreamTranslator;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

/// Translates Gemini SSE events into OpenAI-compatible SSE chunks.
pub struct GeminiToOpenAIStream {
    buffer: Vec<u8>,
    created: u64,
    sent_role: bool,
    chunk_id: String,
    tool_call_index: u64,
}

impl GeminiToOpenAIStream {
    pub fn new() -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            buffer: Vec::new(),
            created,
            sent_role: false,
            chunk_id: format!("chatcmpl-gemini-{created}"),
            tool_call_index: 0,
        }
    }

    fn translate_data(&mut self, data: &str) -> Option<String> {
        let parsed: Value = serde_json::from_str(data).ok()?;
        let candidates = parsed.get("candidates")?.as_array()?;

        let mut output = String::new();

        for candidate in candidates {
            let content = candidate.get("content")?;
            let parts = content.get("parts")?.as_array()?;
            let finish_reason = candidate
                .get("finishReason")
                .and_then(|v| v.as_str())
                .map(|fr| match fr {
                    "STOP" => "stop",
                    "MAX_TOKENS" => "length",
                    "SAFETY" | "RECITATION" => "content_filter",
                    other => other,
                });

            for part in parts {
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    let mut delta = serde_json::Map::new();
                    if !self.sent_role {
                        self.sent_role = true;
                        delta.insert("role".to_string(), Value::String("assistant".to_string()));
                    }
                    delta.insert("content".to_string(), Value::String(text.to_string()));

                    let chunk = serde_json::json!({
                        "id": self.chunk_id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": "gemini",
                        "choices": [{
                            "index": 0,
                            "delta": delta,
                            "finish_reason": Value::Null
                        }]
                    });
                    output.push_str(&format!("data: {chunk}\n\n"));
                }

                if let Some(fc) = part.get("functionCall") {
                    let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args = fc
                        .get("args")
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string());

                    if !self.sent_role {
                        self.sent_role = true;
                    }

                    let chunk = serde_json::json!({
                        "id": self.chunk_id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": "gemini",
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": self.tool_call_index,
                                    "id": format!("call_{}", self.tool_call_index),
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": args
                                    }
                                }]
                            },
                            "finish_reason": Value::Null
                        }]
                    });
                    self.tool_call_index += 1;
                    output.push_str(&format!("data: {chunk}\n\n"));
                }
            }

            // Emit finish reason if present
            if let Some(fr) = finish_reason {
                let chunk = serde_json::json!({
                    "id": self.chunk_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": "gemini",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": fr
                    }]
                });
                output.push_str(&format!("data: {chunk}\n\n"));
            }
        }

        if output.is_empty() {
            None
        } else {
            Some(output)
        }
    }
}

impl StreamTranslator for GeminiToOpenAIStream {
    fn process_chunk(&mut self, data: &[u8], end_of_stream: bool) -> Vec<u8> {
        self.buffer.extend_from_slice(data);

        let mut output = Vec::new();

        loop {
            let buf_str = String::from_utf8_lossy(&self.buffer);
            // Gemini SSE uses standard `data: {...}\n\n` framing.
            if let Some(data_start) = buf_str.find("data: ") {
                let after_prefix = &buf_str[data_start + 6..];
                if let Some(end_pos) = after_prefix.find("\n\n") {
                    let json_str = after_prefix[..end_pos].trim().to_string();
                    let consume_to = data_start + 6 + end_pos + 2;
                    self.buffer = self.buffer[consume_to..].to_vec();

                    if let Some(translated) = self.translate_data(&json_str) {
                        output.extend(translated.as_bytes());
                    }
                } else {
                    break;
                }
            } else {
                // No more data: lines, discard non-data content
                if let Some(last_newline) = buf_str.rfind('\n') {
                    // Keep partial line
                    self.buffer = self.buffer[last_newline + 1..].to_vec();
                }
                break;
            }
        }

        if end_of_stream {
            // Try to process any remaining data
            if !self.buffer.is_empty() {
                let remaining = String::from_utf8_lossy(&self.buffer).to_string();
                if let Some(stripped) = remaining.strip_prefix("data: ") {
                    let json_str = stripped.trim();
                    if let Some(translated) = self.translate_data(json_str) {
                        output.extend(translated.as_bytes());
                    }
                }
                self.buffer.clear();
            }
            output.extend(b"data: [DONE]\n\n");
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_stream() {
        let mut stream = GeminiToOpenAIStream::new();
        let event = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":1}}\n\n";

        let output = stream.process_chunk(event.as_bytes(), true);
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("\"role\":\"assistant\""));
        assert!(output_str.contains("\"content\":\"Hello\""));
        assert!(output_str.contains("\"finish_reason\":\"stop\""));
        assert!(output_str.contains("data: [DONE]"));
    }

    #[test]
    fn test_chunked_stream() {
        let mut stream = GeminiToOpenAIStream::new();

        let part1 = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}],\"role\":\"model\"}}]}\n\n";
        let out1 = stream.process_chunk(part1.as_bytes(), false);
        let out1_str = String::from_utf8(out1).unwrap();
        assert!(out1_str.contains("\"content\":\"Hi\""));

        let part2 = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" there\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n";
        let out2 = stream.process_chunk(part2.as_bytes(), true);
        let out2_str = String::from_utf8(out2).unwrap();
        assert!(out2_str.contains("\"content\":\" there\""));
        assert!(out2_str.contains("data: [DONE]"));
    }

    #[test]
    fn test_function_call_stream() {
        let mut stream = GeminiToOpenAIStream::new();
        let event = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"London\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n";

        let output = stream.process_chunk(event.as_bytes(), true);
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("get_weather"));
        assert!(output_str.contains("tool_calls"));
        assert!(output_str.contains("data: [DONE]"));
    }

    #[test]
    fn test_partial_event_buffering() {
        let mut stream = GeminiToOpenAIStream::new();

        let part1 = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"";
        let out1 = stream.process_chunk(part1.as_bytes(), false);
        // Partial event should not produce output
        assert!(out1.is_empty());

        let part2 = "Hello\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n";
        let out2 = stream.process_chunk(part2.as_bytes(), true);
        let out2_str = String::from_utf8(out2).unwrap();
        assert!(out2_str.contains("Hello"));
    }
}
