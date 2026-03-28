// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! SSE streaming translation from Anthropic to OpenAI format.

use crate::StreamTranslator;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

/// Translates Anthropic SSE events → OpenAI SSE format (chunk by chunk).
pub struct AnthropicToOpenAIStream {
    buffer: Vec<u8>,
    msg_id: String,
    model: String,
    sent_role: bool,
    created: u64,
}

impl AnthropicToOpenAIStream {
    pub fn new() -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            buffer: Vec::new(),
            msg_id: String::new(),
            model: String::new(),
            sent_role: false,
            created,
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
            } else if line.starts_with("data:") {
                data_line = line[5..].to_string();
            }
        }

        if data_line.is_empty() && event_type != "message_stop" {
            return None;
        }

        match event_type.as_str() {
            "message_start" => {
                let parsed: Value = serde_json::from_str(&data_line).ok()?;
                let msg = parsed.get("message")?;
                self.msg_id = msg.get("id").and_then(|v| v.as_str()).unwrap_or("chatcmpl-stream").to_string();
                self.model = msg.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();

                if !self.sent_role {
                    self.sent_role = true;
                    let chunk = serde_json::json!({
                        "id": self.msg_id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
                    });
                    return Some(format!("data: {}\n\n", chunk));
                }
                None
            }
            "content_block_start" => {
                let parsed: Value = serde_json::from_str(&data_line).ok()?;
                let block = parsed.get("content_block")?;
                if block.get("type")?.as_str()? == "tool_use" {
                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                    let chunk = serde_json::json!({
                        "id": self.msg_id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}, "finish_reason": null}]
                    });
                    return Some(format!("data: {}\n\n", chunk));
                }
                None
            }
            "content_block_delta" => {
                let parsed: Value = serde_json::from_str(&data_line).ok()?;
                let delta = parsed.get("delta")?;
                let delta_type = delta.get("type")?.as_str()?;

                match delta_type {
                    "text_delta" => {
                        let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                        let chunk = serde_json::json!({
                            "id": self.msg_id,
                            "object": "chat.completion.chunk",
                            "created": self.created,
                            "model": self.model,
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
                        });
                        Some(format!("data: {}\n\n", chunk))
                    }
                    "input_json_delta" => {
                        let partial = delta.get("partial_json").and_then(|v| v.as_str()).unwrap_or("");
                        let index = parsed.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                        let chunk = serde_json::json!({
                            "id": self.msg_id,
                            "object": "chat.completion.chunk",
                            "created": self.created,
                            "model": self.model,
                            "choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, "function": {"arguments": partial}}]}, "finish_reason": null}]
                        });
                        Some(format!("data: {}\n\n", chunk))
                    }
                    _ => None,
                }
            }
            "content_block_stop" => None,
            "message_delta" => {
                let parsed: Value = serde_json::from_str(&data_line).ok()?;
                let delta = parsed.get("delta")?;
                let stop_reason = delta.get("stop_reason").and_then(|v| v.as_str()).map(|sr| match sr {
                    "end_turn" => "stop",
                    "max_tokens" => "length",
                    "pause_turn" | "tool_use" => "tool_calls",
                    "stop_sequence" => "stop",
                    other => other,
                });

                let chunk = serde_json::json!({
                    "id": self.msg_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": stop_reason}]
                });
                Some(format!("data: {}\n\n", chunk))
            }
            "message_stop" => Some("data: [DONE]\n\n".to_string()),
            "ping" => None,
            _ => None,
        }
    }
}

impl StreamTranslator for AnthropicToOpenAIStream {
    fn process_chunk(&mut self, data: &[u8], end_of_stream: bool) -> Vec<u8> {
        self.buffer.extend_from_slice(data);

        let mut output = Vec::new();
        loop {
            let buf_str = String::from_utf8_lossy(&self.buffer);
            let boundary = buf_str.find("\n\n");
            match boundary {
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
    fn test_stream_text() {
        let mut stream = AnthropicToOpenAIStream::new();

        let events = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-haiku-20240307\",\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let output = stream.process_chunk(events.as_bytes(), true);
        let output_str = String::from_utf8(output).unwrap();

        assert!(output_str.contains("\"role\":\"assistant\""));
        assert!(output_str.contains("\"content\":\"Hello\""));
        assert!(output_str.contains("\"finish_reason\":\"stop\""));
        assert!(output_str.contains("data: [DONE]"));
    }

    #[test]
    fn test_stream_chunked() {
        let mut stream = AnthropicToOpenAIStream::new();

        let part1 = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_02\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-haiku-20240307\",\"stop_reason\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\nevent: content_block_del";
        let part2 = "ta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

        let out1 = stream.process_chunk(part1.as_bytes(), false);
        assert!(String::from_utf8(out1).unwrap().contains("\"role\":\"assistant\""));

        let out2 = stream.process_chunk(part2.as_bytes(), true);
        let out2_str = String::from_utf8(out2).unwrap();
        assert!(out2_str.contains("\"content\":\"Hi\""));
        assert!(out2_str.contains("data: [DONE]"));
    }
}
