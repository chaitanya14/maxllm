// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use crate::factory::PluginError;
use crate::{Plugin, PluginCtx, RequestAction};
use async_trait::async_trait;
use pingora::http::ResponseHeader;
use pingora::proxy::Session;

/// Adds a unique request ID to each request.
/// If the incoming request already has the header, it is preserved.
pub struct RequestIdPlugin {
    name: String,
    header_name: String,
}

impl RequestIdPlugin {
    pub fn from_config(name: &str, config: &toml::Table) -> Result<Self, PluginError> {
        let header_name = config
            .get("header_name")
            .and_then(|v| v.as_str())
            .unwrap_or("X-Request-Id")
            .to_string();

        Ok(Self {
            name: name.to_string(),
            header_name,
        })
    }

    fn generate_id() -> String {
        uuid::Uuid::now_v7().to_string()
    }
}

#[async_trait]
impl Plugin for RequestIdPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_request(
        &self,
        session: &mut Session,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<RequestAction> {
        let existing = session
            .req_header()
            .headers
            .get(&self.header_name)
            .and_then(|v| v.to_str().ok())
            .map(String::from);

        let request_id = existing.unwrap_or_else(Self::generate_id);
        ctx.request_id = Some(request_id);

        Ok(RequestAction::Continue)
    }

    async fn on_response(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<()> {
        if let Some(ref id) = ctx.request_id {
            upstream_response.insert_header(self.header_name.clone(), id)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_id_is_uuid() {
        let id = RequestIdPlugin::generate_id();
        assert!(uuid::Uuid::parse_str(&id).is_ok());
    }

    #[test]
    fn test_from_config_defaults() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "request_id".into());

        let plugin = RequestIdPlugin::from_config("reqid", &config).unwrap();
        assert_eq!(plugin.header_name, "X-Request-Id");
    }
}
