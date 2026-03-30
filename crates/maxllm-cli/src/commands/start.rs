// Copyright 2025 MaxLLM Contributors.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

const PID_FILE: &str = "/tmp/maxllm.pid";

/// Start the gateway server by spawning the `maxllm-server` binary.
pub fn run(config: PathBuf, port: Option<u16>, daemon: bool) -> i32 {
    // Validate config exists
    if !config.exists() {
        eprintln!("Error: config file not found: {}", config.display());
        eprintln!("Run `maxllm init` to generate a starter configuration.");
        return 1;
    }

    // Validate the config can be parsed
    if let Err(e) = maxllm_config::Config::from_file(&config) {
        eprintln!("Error: invalid configuration: {e}");
        return 1;
    }

    // Find the server binary (same directory as this CLI binary)
    let server_bin = find_server_binary();

    let mut cmd = std::process::Command::new(&server_bin);
    cmd.arg("--config").arg(&config);

    if daemon {
        cmd.arg("--daemon");
    }

    // If port override specified, set an env var that could be used
    // (the config itself must reference the port, but we print a hint)
    if let Some(p) = port {
        eprintln!("Note: --port {p} flag noted. Ensure your config's [server] listen address uses this port.");
    }

    eprintln!("Starting MaxLLM gateway...");
    eprintln!("  Config: {}", config.display());
    eprintln!("  Binary: {}", server_bin.display());

    match cmd.spawn() {
        Ok(child) => {
            let pid = child.id();
            // Write PID file
            if let Err(e) = std::fs::write(PID_FILE, pid.to_string()) {
                eprintln!("Warning: failed to write PID file: {e}");
            }
            if daemon {
                eprintln!("Gateway started in background (PID: {pid})");
                eprintln!("PID file: {PID_FILE}");
                0
            } else {
                eprintln!("Gateway started (PID: {pid})");
                // Wait for the child process
                let mut child = child;
                match child.wait() {
                    Ok(status) => {
                        let _ = std::fs::remove_file(PID_FILE);
                        status.code().unwrap_or(1)
                    }
                    Err(e) => {
                        eprintln!("Error waiting for gateway: {e}");
                        let _ = std::fs::remove_file(PID_FILE);
                        1
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error: failed to start gateway: {e}");
            eprintln!("Looked for server binary at: {}", server_bin.display());
            1
        }
    }
}

fn find_server_binary() -> PathBuf {
    // Try same directory as current executable
    if let Ok(exe) = std::env::current_exe() {
        let dir = exe.parent().unwrap_or(std::path::Path::new("."));
        let candidate = dir.join("maxllm-server");
        if candidate.exists() {
            return candidate;
        }
    }
    // Fall back to PATH
    PathBuf::from("maxllm-server")
}
