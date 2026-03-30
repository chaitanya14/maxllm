// Copyright 2025 MaxLLM Contributors.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

/// Show the status of a running gateway.
pub async fn run(config: Option<PathBuf>) -> i32 {
    let pid_file = super::pid_file_path();

    let pid_str = match std::fs::read_to_string(&pid_file) {
        Ok(s) => s.trim().to_string(),
        Err(_) => {
            println!("Status: not running (no PID file)");
            return 1;
        }
    };

    let pid: i32 = match pid_str.parse() {
        Ok(p) => p,
        Err(_) => {
            println!("Status: unknown (invalid PID file)");
            return 1;
        }
    };

    // Check if the process is alive
    let alive = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok();

    if !alive {
        println!("Status: not running (stale PID file, PID: {pid})");
        let _ = std::fs::remove_file(&pid_file);
        return 1;
    }

    println!("Status: running");
    println!("  PID: {pid}");

    // Determine the port from config file
    let port = resolve_port(config.as_deref());
    let url = format!("http://127.0.0.1:{port}/health");

    // Try to hit the health endpoint
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            println!("  Health: unknown (failed to build HTTP client: {e})");
            return 0;
        }
    };

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("  Health: ok");
        }
        Ok(resp) => {
            println!("  Health: unhealthy (status {})", resp.status());
        }
        Err(_) => {
            println!("  Health: unreachable (port {port})");
        }
    }

    0
}

/// Try to read the listen port from the config file. Falls back to 8080.
fn resolve_port(config: Option<&std::path::Path>) -> u16 {
    let path = config.unwrap_or_else(|| std::path::Path::new("maxllm.toml"));
    if let Ok(cfg) = maxllm_config::Config::from_file(path) {
        cfg.server.listen.port()
    } else {
        8080
    }
}
