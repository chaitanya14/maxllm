// Copyright 2025 MaxLLM Contributors.
// SPDX-License-Identifier: Apache-2.0

use nix::sys::signal::{self, Signal};
use nix::unistd::Pid;

/// Stop a running gateway by sending SIGTERM.
pub fn run() -> i32 {
    let pid_file = super::pid_file_path();

    let pid_str = match std::fs::read_to_string(&pid_file) {
        Ok(s) => s.trim().to_string(),
        Err(_) => {
            eprintln!(
                "No running gateway found (PID file not found: {})",
                pid_file.display()
            );
            return 1;
        }
    };

    let pid: i32 = match pid_str.parse() {
        Ok(p) => p,
        Err(_) => {
            eprintln!("Invalid PID in {}: {pid_str}", pid_file.display());
            return 1;
        }
    };

    // Validate the PID belongs to a maxllm process before sending signal.
    // First check if the process exists at all (kill -0).
    if signal::kill(Pid::from_raw(pid), None).is_err() {
        eprintln!("Process {pid} is not running (stale PID file).");
        let _ = std::fs::remove_file(&pid_file);
        return 1;
    }

    // On macOS, check the process name via `ps`.
    // On Linux, check /proc/{pid}/cmdline.
    if !is_maxllm_process(pid) {
        eprintln!("PID {pid} does not belong to a maxllm process. Refusing to send signal.");
        eprintln!("Removing stale PID file.");
        let _ = std::fs::remove_file(&pid_file);
        return 1;
    }

    eprintln!("Sending SIGTERM to gateway (PID: {pid})...");
    match signal::kill(Pid::from_raw(pid), Signal::SIGTERM) {
        Ok(()) => {
            let _ = std::fs::remove_file(&pid_file);
            eprintln!("Gateway stopped.");
            0
        }
        Err(e) => {
            eprintln!("Failed to stop gateway (PID {pid}): {e}");
            eprintln!("The process may have already exited.");
            let _ = std::fs::remove_file(&pid_file);
            1
        }
    }
}

/// Check whether the given PID belongs to a maxllm process.
fn is_maxllm_process(pid: i32) -> bool {
    #[cfg(target_os = "linux")]
    {
        // Read /proc/{pid}/cmdline — null-separated args
        if let Ok(cmdline) = std::fs::read_to_string(format!("/proc/{pid}/cmdline")) {
            let lower = cmdline.to_lowercase();
            return lower.contains("maxllm");
        }
        false
    }
    #[cfg(not(target_os = "linux"))]
    {
        // macOS / other Unix: use `ps -p <pid> -o comm=`
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output()
        {
            let name = String::from_utf8_lossy(&output.stdout).to_lowercase();
            return name.contains("maxllm");
        }
        false
    }
}
