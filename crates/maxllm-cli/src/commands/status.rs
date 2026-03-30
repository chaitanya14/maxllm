// Copyright 2025 MaxLLM Contributors.
// SPDX-License-Identifier: Apache-2.0

const PID_FILE: &str = "/tmp/maxllm.pid";

/// Show the status of a running gateway.
pub async fn run() -> i32 {
    let pid_str = match std::fs::read_to_string(PID_FILE) {
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
        let _ = std::fs::remove_file(PID_FILE);
        return 1;
    }

    println!("Status: running");
    println!("  PID: {pid}");

    // Try to hit the health endpoint
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap();

    match client.get("http://127.0.0.1:8080/health").send().await {
        Ok(resp) if resp.status().is_success() => {
            println!("  Health: ok");
        }
        Ok(resp) => {
            println!("  Health: unhealthy (status {})", resp.status());
        }
        Err(_) => {
            println!("  Health: unreachable (port 8080)");
        }
    }

    0
}
