// Copyright 2025 MaxLLM Contributors.
// SPDX-License-Identifier: Apache-2.0

use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;

const PID_FILE: &str = "/tmp/maxllm.pid";

/// Stop a running gateway by sending SIGTERM.
pub fn run() -> i32 {
    let pid_str = match std::fs::read_to_string(PID_FILE) {
        Ok(s) => s.trim().to_string(),
        Err(_) => {
            eprintln!("No running gateway found (PID file not found: {PID_FILE})");
            return 1;
        }
    };

    let pid: i32 = match pid_str.parse() {
        Ok(p) => p,
        Err(_) => {
            eprintln!("Invalid PID in {PID_FILE}: {pid_str}");
            return 1;
        }
    };

    eprintln!("Sending SIGTERM to gateway (PID: {pid})...");
    match signal::kill(Pid::from_raw(pid), Signal::SIGTERM) {
        Ok(()) => {
            let _ = std::fs::remove_file(PID_FILE);
            eprintln!("Gateway stopped.");
            0
        }
        Err(e) => {
            eprintln!("Failed to stop gateway (PID {pid}): {e}");
            eprintln!("The process may have already exited.");
            let _ = std::fs::remove_file(PID_FILE);
            1
        }
    }
}
