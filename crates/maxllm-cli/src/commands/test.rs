// Copyright 2025 MaxLLM Contributors.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

/// Test connectivity to each configured provider.
pub async fn run(config_path: PathBuf) -> i32 {
    let config = match maxllm_config::Config::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading config: {e}");
            return 1;
        }
    };

    if config.providers.is_empty() {
        println!("No providers configured.");
        return 0;
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();

    let mut all_ok = true;

    println!("Testing {} provider(s)...\n", config.providers.len());

    for (name, provider) in &config.providers {
        let url = &provider.base_url;
        print!("  {name} ({url})... ");

        match client.get(url).send().await {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() || status.as_u16() == 404 || status.as_u16() == 405 {
                    // 404/405 means the server is up but the path doesn't exist, which is fine
                    println!("OK (HTTP {status})");
                } else if status.as_u16() == 401 || status.as_u16() == 403 {
                    println!("OK (auth required — server is reachable)");
                } else {
                    println!("WARNING (HTTP {status})");
                }
            }
            Err(e) => {
                println!("FAILED: {e}");
                all_ok = false;
            }
        }
    }

    println!();
    if all_ok {
        println!("All providers reachable.");
        0
    } else {
        println!("Some providers are unreachable.");
        1
    }
}
