// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! SSE streaming translation from Cohere v2 to OpenAI format.
//!
//! Cohere v2 streaming sends SSE events with different event types:
//! - `message-start`: Contains message metadata
//! - `content-start`: Marks the start of a content block
//! - `content-delta`: Contains incremental content
//! - `content-end`: Marks the end of a content block
//! - `tool-call-start`: Start of a tool call
//! - `tool-call-delta`: Incremental tool call arguments
//! - `tool-call-end`: End of a tool call
//! - `message-end`: Contains finish reason and usage

use crate::StreamTranslator;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

/// Translates Cohere v2 SSE events into OpenAI-compatible SSE chunks.
pub struct CohereToOpenAIStream {
    buffer: Vec<u8>,
    created: u64,
    sent_role: bool,
    chunk_id: String,
    model: String,
}

impl Default for CohereToOpenAIStream {
    fn default() -> Self {
        Self::new()
    }
}

impl CohereToOpenAIStream {
    pub fn new() -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            buffer: Vec::new(),
            created,
            sent_role: false,
            chunk_id: format!("chatcmpl-cohere-{created}"),
            model: String::new(),
        }
    }

    fn translate_event(&mut self, raw_event: &str) -> Option<String> {
        let mut event_type = String::new();
        let mut data_line = String::new();

        for line in raw_event.lines() {
            if let Some(et) = line.strip_prefix("event: ") {
                event_type = et.trim().to_string();
            } else if let Some(d) = line.strip_prefix("data: ") {
                data_line = d.to_string();
            } else if let Some(d) = line.strip_prefix("data:") {
                data_line = d.to_string();
            }
        }

        match event_type.as_str() {
            "message-start" => {
                if !data_line.is_empty() {
                    let parsed: Value = serde_json::from_str(&data_line).ok()?;
                    if let Some(id) = parsed.get("id").and_then(|v| v.as_str()) {
                        self.chunk_id = id.to_string();
                    }
                    if let Some(model) = parsed
                        .get("delta")
                        .and_then(|d| d.get("message"))
                        .and_then(|m| m.get("model"))
                        .and_then(|v| v.as_str())
                    {
                        self.model = model.to_string();
                    }
                }
                if !self.sent_role {
                    self.sent_role = true;
                    let chunk = serde_json::json!({
                        "id": self.chunk_id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": Value::Null
                        }]
                    });
                    return Some(format!("data: {chunk}\n\n"));
                }
                None
            }
            "content-delta" => {
                if data_line.is_empty() {
                    return None;
                }
                let parsed: Value = serde_json::from_str(&data_line).ok()?;
                let text = parsed
                    .get("delta")
                    .and_then(|d| d.get("message"))
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.get("text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let chunk = serde_json::json!({
                    "id": self.chunk_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": Value::Null
                    }]
                });
                Some(format!("data: {chunk}\n\n"))
            }
            "tool-call-start" => {
                if data_line.is_empty() {
                    return None;
                }
                let parsed: Value = serde_json::from_str(&data_line).ok()?;
                let delta = parsed.get("delta")?;
                let tool_call = delta
                    .get("message")
                    .and_then(|m| m.get("tool_calls"))
                    .and_then(|tc| tc.get(0))?;
                let id = tool_call.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = tool_call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0);

                let chunk = serde_json::json!({
                    "id": self.chunk_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""}
                            }]
                        },
                        "finish_reason": Value::Null
                    }]
                });
                Some(format!("data: {chunk}\n\n"))
            }
            "tool-call-delta" => {
                if data_line.is_empty() {
                    return None;
                }
                let parsed: Value = serde_json::from_str(&data_line).ok()?;
                let args = parsed
                    .get("delta")
                    .and_then(|d| d.get("message"))
                    .and_then(|m| m.get("tool_calls"))
                    .and_then(|tc| tc.get(0))
                    .and_then(|tc| tc.get("function"))
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0);

                let chunk = serde_json::json!({
                    "id": self.chunk_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "function": {"arguments": args}
                            }]
                        },
                        "finish_reason": Value::Null
                    }]
                });
                Some(format!("data: {chunk}\n\n"))
            }
            "message-end" => {
                let finish_reason: String = if !data_line.is_empty() {
                    let parsed: Value = serde_json::from_str(&data_line).ok()?;
                    parsed
                        .get("delta")
                        .and_then(|d| d.get("finish_reason"))
                        .and_then(|v| v.as_str())
                        .map(|fr| match fr {
                            "COMPLETE" => "stop",
                            "MAX_TOKENS" => "length",
                            "TOOL_CALL" => "tool_calls",
                            other => other,
                        })
                        .unwrap_or("stop")
                        .to_string()
                } else {
                    "stop".to_string()
                };

                let chunk = serde_json::json!({
                    "id": self.chunk_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }]
                });
                let mut out = format!("data: {chunk}\n\n");
                out.push_str("data: [DONE]\n\n");
                Some(out)
            }
            "content-start" | "content-end" | "tool-call-end" => None,
            _ => None,
        }
    }
}

impl StreamTranslator for CohereToOpenAIStream {
    fn process_chunk(&mut self, data: &[u8], end_of_stream: bool) -> Vec<u8> {
        self.buffer.extend_from_slice(data);

        let mut output = Vec::new();
        loop {
            let buf_str = String::from_utf8_lossy(&self.buffer);
            match buf_str.find("\n\n") {
                Some(pos) => {
                    let event_data: String = buf_str[..pos].to_string();
                    self.buffer = self.buffer[pos + 2..].to_vec();
                    if let Some(translated) = self.translate_event(&event_data) {
                        output.extend(translated.as_bytes());
                    }
                }
                None => break,
            }
        }

        if end_of_stream && !self.buffer.is_empty() {
            let remaining = String::from_utf8_lossy(&self.buffer).to_string();
            if let Some(translated) = self.translate_event(&remaining) {
                output.extend(translated.as_bytes());
            }
            self.buffer.clear();
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_stream() {
        let mut stream = CohereToOpenAIStream::new();

        let events = concat!(
            "event: message-start\n",
            "data: {\"id\":\"cohere-1\",\"delta\":{\"message\":{\"model\":\"command-r-plus\",\"role\":\"assistant\"}}}\n\n",
            "event: content-delta\n",
            "data: {\"delta\":{\"message\":{\"content\":{\"text\":\"Hello\"}}}}\n\n",
            "event: message-end\n",
            "data: {\"delta\":{\"finish_reason\":\"COMPLETE\"}}\n\n",
        );

        let output = stream.process_chunk(events.as_bytes(), true);
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("\"role\":\"assistant\""));
        assert!(output_str.contains("\"content\":\"Hello\""));
        assert!(output_str.contains("\"finish_reason\":\"stop\""));
        assert!(output_str.contains("data: [DONE]"));
    }

    #[test]
    fn test_chunked_delivery() {
        let mut stream = CohereToOpenAIStream::new();

        let part1 = "event: message-start\ndata: {\"id\":\"c2\",\"delta\":{\"message\":{\"model\":\"command-r\",\"role\":\"assistant\"}}}\n\nevent: content-del";
        let out1 = stream.process_chunk(part1.as_bytes(), false);
        let out1_str = String::from_utf8(out1).unwrap();
        assert!(out1_str.contains("\"role\":\"assistant\""));

        let part2 = "ta\ndata: {\"delta\":{\"message\":{\"content\":{\"text\":\"Hi\"}}}}\n\nevent: message-end\ndata: {\"delta\":{\"finish_reason\":\"COMPLETE\"}}\n\n";
        let out2 = stream.process_chunk(part2.as_bytes(), true);
        let out2_str = String::from_utf8(out2).unwrap();
        assert!(out2_str.contains("\"content\":\"Hi\""));
        assert!(out2_str.contains("data: [DONE]"));
    }

    #[test]
    fn test_tool_call_stream() {
        let mut stream = CohereToOpenAIStream::new();

        let events = concat!(
            "event: message-start\n",
            "data: {\"id\":\"c3\",\"delta\":{\"message\":{\"model\":\"command-r-plus\",\"role\":\"assistant\"}}}\n\n",
            "event: tool-call-start\n",
            "data: {\"index\":0,\"delta\":{\"message\":{\"tool_calls\":[{\"id\":\"tc1\",\"function\":{\"name\":\"search\"}}]}}}\n\n",
            "event: tool-call-delta\n",
            "data: {\"index\":0,\"delta\":{\"message\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{\\\"q\\\":\\\"hi\\\"}\"}}]}}}\n\n",
            "event: message-end\n",
            "data: {\"delta\":{\"finish_reason\":\"TOOL_CALL\"}}\n\n",
        );

        let output = stream.process_chunk(events.as_bytes(), true);
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("\"name\":\"search\""));
        assert!(output_str.contains("tool_calls"));
        assert!(output_str.contains("\"finish_reason\":\"tool_calls\""));
    }
}
