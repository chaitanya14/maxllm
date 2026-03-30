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
use pingora::http::ResponseHeader;
use pingora::proxy::Session;

/// CORS plugin. Handles OPTIONS preflight and adds CORS headers to responses.
pub struct CorsPlugin {
    name: String,
    allow_origin: String,
    allow_methods: String,
    allow_headers: String,
    max_age: String,
    allow_credentials: bool,
}

impl CorsPlugin {
    pub fn from_config(name: &str, config: &toml::Table) -> Result<Self, PluginError> {
        Ok(Self {
            name: name.to_string(),
            allow_origin: config
                .get("allow_origin")
                .and_then(|v| v.as_str())
                .unwrap_or("*")
                .to_string(),
            allow_methods: config
                .get("allow_methods")
                .and_then(|v| v.as_str())
                .unwrap_or("GET, POST, OPTIONS")
                .to_string(),
            allow_headers: config
                .get("allow_headers")
                .and_then(|v| v.as_str())
                .unwrap_or("Content-Type, Authorization")
                .to_string(),
            max_age: config
                .get("max_age")
                .and_then(|v| v.as_str())
                .unwrap_or("86400")
                .to_string(),
            allow_credentials: config
                .get("allow_credentials")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        })
    }

    fn cors_headers(&self) -> Vec<(String, String)> {
        let mut headers = vec![
            (
                "Access-Control-Allow-Origin".into(),
                self.allow_origin.clone(),
            ),
            (
                "Access-Control-Allow-Methods".into(),
                self.allow_methods.clone(),
            ),
            (
                "Access-Control-Allow-Headers".into(),
                self.allow_headers.clone(),
            ),
            ("Access-Control-Max-Age".into(), self.max_age.clone()),
        ];
        if self.allow_credentials {
            headers.push(("Access-Control-Allow-Credentials".into(), "true".into()));
        }
        headers
    }
}

#[async_trait]
impl Plugin for CorsPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_request(
        &self,
        session: &mut Session,
        _ctx: &mut PluginCtx,
    ) -> pingora::Result<RequestAction> {
        let method = session.req_header().method.as_str();
        if method == "OPTIONS" {
            let mut headers = self.cors_headers();
            headers.push(("Content-Length".into(), "0".into()));
            return Ok(RequestAction::Respond(HttpResponse {
                status: 204,
                headers,
                body: Vec::new(),
            }));
        }
        Ok(RequestAction::Continue)
    }

    async fn on_response(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        _ctx: &mut PluginCtx,
    ) -> pingora::Result<()> {
        for (k, v) in self.cors_headers() {
            upstream_response.insert_header(k, &v)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_config_defaults() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "cors".into());

        let plugin = CorsPlugin::from_config("cors", &config).unwrap();
        assert_eq!(plugin.allow_origin, "*");
        assert_eq!(plugin.allow_methods, "GET, POST, OPTIONS");
        assert!(!plugin.allow_credentials);
    }

    #[test]
    fn test_cors_headers() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "cors".into());
        config.insert("allow_credentials".into(), true.into());

        let plugin = CorsPlugin::from_config("cors", &config).unwrap();
        let headers = plugin.cors_headers();
        assert!(headers
            .iter()
            .any(|(k, v)| k == "Access-Control-Allow-Credentials" && v == "true"));
    }
}
