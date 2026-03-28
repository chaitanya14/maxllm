// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use crate::factory::PluginError;
use crate::{HttpResponse, Plugin, PluginCtx, RequestAction};
use async_trait::async_trait;
use pingora::proxy::Session;
use pingora_limits::rate::Rate;
use std::sync::Arc;
use std::time::Duration;

/// What to use as the rate limit key.
#[derive(Debug, Clone)]
enum RateLimitKey {
    ClientIp,
    ClientId,
    Header(String),
}

/// Sliding-window rate limiter using pingora-limits.
pub struct RateLimitPlugin {
    name: String,
    rate: Arc<Rate>,
    max_requests: isize,
    window: Duration,
    key_type: RateLimitKey,
}

impl RateLimitPlugin {
    pub fn from_config(name: &str, config: &toml::Table) -> Result<Self, PluginError> {
        let rpm = config
            .get("requests_per_minute")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| {
                PluginError::Config("rate_limit requires 'requests_per_minute'".into())
            })?;

        let key_str = config
            .get("key")
            .and_then(|v| v.as_str())
            .unwrap_or("client_ip");

        let key_type = match key_str {
            "client_ip" => RateLimitKey::ClientIp,
            "client_id" => RateLimitKey::ClientId,
            other => RateLimitKey::Header(other.to_string()),
        };

        Ok(Self {
            name: name.to_string(),
            rate: Arc::new(Rate::new(Duration::from_secs(60))),
            max_requests: rpm as isize,
            window: Duration::from_secs(60),
            key_type,
        })
    }

    fn extract_key(&self, session: &Session, ctx: &PluginCtx) -> Option<String> {
        match &self.key_type {
            RateLimitKey::ClientIp => ctx.client_ip.clone(),
            RateLimitKey::ClientId => ctx.client_id.clone(),
            RateLimitKey::Header(h) => session
                .req_header()
                .headers
                .get(h)
                .and_then(|v| v.to_str().ok())
                .map(String::from),
        }
    }
}

#[async_trait]
impl Plugin for RateLimitPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_request(
        &self,
        session: &mut Session,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<RequestAction> {
        let key = match self.extract_key(session, ctx) {
            Some(k) => k,
            None => return Ok(RequestAction::Continue),
        };

        let current = self.rate.observe(&key, 1);
        if current > self.max_requests {
            let remaining_secs = self.window.as_secs();
            let mut resp = HttpResponse::json_error(429, "Rate limit exceeded");
            resp.headers
                .push(("Retry-After".into(), remaining_secs.to_string()));
            resp.headers
                .push(("X-RateLimit-Limit".into(), self.max_requests.to_string()));
            resp.headers.push(("X-RateLimit-Remaining".into(), "0".into()));
            return Ok(RequestAction::Respond(resp));
        }

        Ok(RequestAction::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_config() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "rate_limit".into());
        config.insert("requests_per_minute".into(), 600.into());
        config.insert("key".into(), "client_ip".into());

        let plugin = RateLimitPlugin::from_config("limiter", &config).unwrap();
        assert_eq!(plugin.max_requests, 600);
    }

    #[test]
    fn test_missing_rpm_fails() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "rate_limit".into());

        assert!(RateLimitPlugin::from_config("limiter", &config).is_err());
    }
}
