// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Lock-free circuit breaker using atomic counters.
///
/// State transitions:
/// - Closed (healthy): failures < max_fails
/// - Open (unhealthy): failures >= max_fails AND within fail_timeout
/// - Half-open: failures >= max_fails BUT fail_timeout has elapsed → resets on next success
pub struct CircuitBreaker {
    failures: AtomicU32,
    last_failure_at: AtomicU64,
    max_fails: u32,
    fail_timeout_ms: u64,
}

impl CircuitBreaker {
    pub fn new(max_fails: u32, fail_timeout_secs: u64) -> Self {
        Self {
            failures: AtomicU32::new(0),
            last_failure_at: AtomicU64::new(0),
            max_fails,
            fail_timeout_ms: fail_timeout_secs * 1000,
        }
    }

    /// Returns true if the circuit is open (provider should be skipped).
    pub fn is_open(&self) -> bool {
        let failures = self.failures.load(Ordering::Relaxed);
        if failures < self.max_fails {
            return false;
        }
        let last = self.last_failure_at.load(Ordering::Relaxed);
        let now = now_ms();
        // If timeout has elapsed, the circuit is half-open (allow a retry)
        if now.saturating_sub(last) >= self.fail_timeout_ms {
            return false;
        }
        true
    }

    /// Record a failure. Increments failure count and updates timestamp.
    pub fn record_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
        self.last_failure_at.store(now_ms(), Ordering::Relaxed);
    }

    /// Record a success. Resets failure count.
    pub fn record_success(&self) {
        self.failures.store(0, Ordering::Relaxed);
    }

    pub fn failure_count(&self) -> u32 {
        self.failures.load(Ordering::Relaxed)
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starts_closed() {
        let cb = CircuitBreaker::new(3, 60);
        assert!(!cb.is_open());
    }

    #[test]
    fn test_opens_after_max_fails() {
        let cb = CircuitBreaker::new(3, 60);
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());
        cb.record_failure();
        assert!(cb.is_open());
    }

    #[test]
    fn test_success_resets() {
        let cb = CircuitBreaker::new(2, 60);
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_open());
        cb.record_success();
        assert!(!cb.is_open());
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn test_half_open_after_timeout() {
        let cb = CircuitBreaker::new(1, 0); // 0 second timeout
        cb.record_failure();
        // With 0s timeout, it should immediately be half-open
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(!cb.is_open());
    }
}
