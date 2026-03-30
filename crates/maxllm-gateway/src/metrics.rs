// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use once_cell::sync::Lazy;
use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts, Registry, TextEncoder,
};

pub static METRICS: Lazy<GatewayMetrics> = Lazy::new(GatewayMetrics::new);

pub struct GatewayMetrics {
    pub registry: Registry,
    pub requests_total: IntCounterVec,
    pub tokens_in_total: IntCounterVec,
    pub tokens_out_total: IntCounterVec,
    pub request_duration_seconds: HistogramVec,
    pub active_requests: IntGauge,
    pub fallbacks_total: IntCounterVec,
}

impl GatewayMetrics {
    fn new() -> Self {
        let registry = Registry::new();

        let requests_total = IntCounterVec::new(
            Opts::new("maxllm_requests_total", "Total number of requests"),
            &["provider", "model", "status"],
        )
        .expect("metric");
        registry
            .register(Box::new(requests_total.clone()))
            .expect("register");

        let tokens_in_total = IntCounterVec::new(
            Opts::new("maxllm_tokens_in_total", "Total input tokens"),
            &["provider", "model"],
        )
        .expect("metric");
        registry
            .register(Box::new(tokens_in_total.clone()))
            .expect("register");

        let tokens_out_total = IntCounterVec::new(
            Opts::new("maxllm_tokens_out_total", "Total output tokens"),
            &["provider", "model"],
        )
        .expect("metric");
        registry
            .register(Box::new(tokens_out_total.clone()))
            .expect("register");

        let request_duration_seconds = HistogramVec::new(
            HistogramOpts::new("maxllm_request_duration_seconds", "Request latency")
                .buckets(vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]),
            &["provider"],
        )
        .expect("metric");
        registry
            .register(Box::new(request_duration_seconds.clone()))
            .expect("register");

        let active_requests =
            IntGauge::new("maxllm_active_requests", "Currently active requests").expect("metric");
        registry
            .register(Box::new(active_requests.clone()))
            .expect("register");

        let fallbacks_total = IntCounterVec::new(
            Opts::new("maxllm_fallbacks_total", "Total fallback invocations"),
            &["from_provider", "to_provider"],
        )
        .expect("metric");
        registry
            .register(Box::new(fallbacks_total.clone()))
            .expect("register");

        Self {
            registry,
            requests_total,
            tokens_in_total,
            tokens_out_total,
            request_duration_seconds,
            active_requests,
            fallbacks_total,
        }
    }

    pub fn encode(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .expect("encode");
        String::from_utf8(buffer).expect("utf8")
    }
}
