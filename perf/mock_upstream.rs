//! Fast HTTP mock upstream that returns a canned OpenAI response.
//! Handles both Content-Length and Transfer-Encoding: chunked requests.
//!
//! Compile: rustc -O perf/mock_upstream.rs -o perf/mock_upstream
//! Run:     ./perf/mock_upstream [port]   (default: 9999)

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;

const BODY: &[u8] = br#"{"id":"chatcmpl-mock","object":"chat.completion","created":1700000000,"model":"mock-model","choices":[{"index":0,"message":{"role":"assistant","content":"Four."},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}"#;

fn drain_chunked(reader: &mut BufReader<std::net::TcpStream>) -> bool {
    // Read chunked body: each chunk is "size\r\n data \r\n", terminated by "0\r\n\r\n"
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return false;
        }
        let size = usize::from_str_radix(line.trim(), 16).unwrap_or(0);
        if size == 0 {
            // Read trailing \r\n after "0\r\n"
            line.clear();
            let _ = reader.read_line(&mut line);
            return true;
        }
        let mut body = vec![0u8; size + 2]; // +2 for trailing \r\n
        if reader.read_exact(&mut body).is_err() {
            return false;
        }
    }
}

fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(9999);

    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
        BODY.len()
    );
    let mut resp_bytes = response.into_bytes();
    resp_bytes.extend_from_slice(BODY);

    let listener = TcpListener::bind(format!("127.0.0.1:{port}")).unwrap();
    eprintln!("Mock upstream listening on 127.0.0.1:{port}");

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let resp = resp_bytes.clone();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stream);
            loop {
                // Read HTTP headers
                let mut content_length: Option<usize> = None;
                let mut chunked = false;
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line) {
                        Ok(0) | Err(_) => return,
                        Ok(_) => {}
                    }
                    let lower = line.to_ascii_lowercase();
                    if lower.starts_with("content-length:") {
                        if let Some(val) = lower.split(':').nth(1) {
                            content_length = val.trim().parse().ok();
                        }
                    }
                    if lower.starts_with("transfer-encoding:") && lower.contains("chunked") {
                        chunked = true;
                    }
                    if line == "\r\n" {
                        break;
                    }
                }

                // Drain request body
                if chunked {
                    if !drain_chunked(&mut reader) {
                        return;
                    }
                } else if let Some(len) = content_length {
                    if len > 0 {
                        let mut body = vec![0u8; len];
                        if reader.read_exact(&mut body).is_err() {
                            return;
                        }
                    }
                }

                // Send response
                if reader.get_mut().write_all(&resp).is_err() {
                    return;
                }
            }
        });
    }
}
