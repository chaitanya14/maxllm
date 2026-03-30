// Copyright 2025 MaxLLM Contributors.
// SPDX-License-Identifier: Apache-2.0

/// List virtual keys via the admin API.
pub async fn list(url: &str, admin_key: &str) -> i32 {
    let api_url = format!("{}/admin/keys", url.trim_end_matches('/'));

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: failed to build HTTP client: {e}");
            return 1;
        }
    };

    match client
        .get(&api_url)
        .header("Authorization", format!("Bearer {admin_key}"))
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            match resp.text().await {
                Ok(body) => {
                    if status.is_success() {
                        // Pretty-print JSON
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&body) {
                            if let Some(keys) = parsed.as_array() {
                                if keys.is_empty() {
                                    println!("No virtual keys found.");
                                } else {
                                    println!("{:<38} {:<20} {:<15} {}", "ID", "NAME", "PREFIX", "ACTIVE");
                                    println!("{}", "-".repeat(85));
                                    for key in keys {
                                        println!(
                                            "{:<38} {:<20} {:<15} {}",
                                            key["id"].as_str().unwrap_or("-"),
                                            key["name"].as_str().unwrap_or("-"),
                                            key["key_prefix"].as_str().unwrap_or("-"),
                                            key["is_active"].as_bool().unwrap_or(false),
                                        );
                                    }
                                }
                                return 0;
                            }
                        }
                        println!("{body}");
                        0
                    } else {
                        eprintln!("Error (HTTP {status}): {body}");
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
            eprintln!("Error: {e}");
            1
        }
    }
}

/// Create a new virtual key via the admin API.
pub async fn create(url: &str, admin_key: &str, name: &str) -> i32 {
    let api_url = format!("{}/admin/keys", url.trim_end_matches('/'));

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: failed to build HTTP client: {e}");
            return 1;
        }
    };

    let body = serde_json::json!({ "name": name });

    match client
        .post(&api_url)
        .header("Authorization", format!("Bearer {admin_key}"))
        .json(&body)
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            match resp.text().await {
                Ok(body) => {
                    if status.is_success() {
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&body) {
                            println!("Key created successfully!");
                            println!();
                            println!("  Key:    {}", parsed["key"].as_str().unwrap_or("-"));
                            println!("  ID:     {}", parsed["key_id"].as_str().unwrap_or("-"));
                            println!("  Name:   {}", parsed["name"].as_str().unwrap_or("-"));
                            println!("  Prefix: {}", parsed["key_prefix"].as_str().unwrap_or("-"));
                            println!();
                            println!("Save this key — it will not be shown again.");
                            return 0;
                        }
                        println!("{body}");
                        0
                    } else {
                        eprintln!("Error (HTTP {status}): {body}");
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
            eprintln!("Error: {e}");
            1
        }
    }
}
