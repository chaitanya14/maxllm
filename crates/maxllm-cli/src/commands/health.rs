// Copyright 2025 MaxLLM Contributors.
// SPDX-License-Identifier: Apache-2.0

/// Hit the /health endpoint and report the result.
pub async fn run(url: &str) -> i32 {
    let health_url = format!("{}/health", url.trim_end_matches('/'));

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: failed to build HTTP client: {e}");
            return 1;
        }
    };

    match client.get(&health_url).send().await {
        Ok(resp) => {
            let status = resp.status();
            match resp.text().await {
                Ok(body) => {
                    if status.is_success() {
                        println!("healthy: {body}");
                        0
                    } else {
                        println!("unhealthy (HTTP {status}): {body}");
                        1
                    }
                }
                Err(e) => {
                    eprintln!("Error reading response: {e}");
                    1
                }
            }
        }
        Err(e) => {
            eprintln!("Error: gateway unreachable at {health_url}: {e}");
            1
        }
    }
}
