// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use crate::{HttpResponse, Plugin, PluginCtx, RequestAction};
use bytes::Bytes;
use pingora::http::ResponseHeader;
use pingora::prelude::*;
use pingora::proxy::Session;
use std::sync::Arc;

/// An ordered chain of plugins to execute at each lifecycle hook.
pub struct PluginChain {
    plugins: Vec<Arc<dyn Plugin>>,
}

impl PluginChain {
    pub fn new(plugins: Vec<Arc<dyn Plugin>>) -> Self {
        Self { plugins }
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Run on_request for each plugin. Returns Some(HttpResponse) on first short-circuit.
    pub async fn run_request(
        &self,
        session: &mut Session,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<Option<HttpResponse>> {
        for plugin in &self.plugins {
            match plugin.on_request(session, ctx).await? {
                RequestAction::Continue => {}
                RequestAction::Respond(resp) => {
                    tracing::debug!(plugin = plugin.name(), "plugin short-circuited request");
                    return Ok(Some(resp));
                }
            }
        }
        Ok(None)
    }

    /// Run on_upstream_request for each plugin.
    pub async fn run_upstream_request(
        &self,
        session: &mut Session,
        upstream_request: &mut RequestHeader,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<()> {
        for plugin in &self.plugins {
            plugin
                .on_upstream_request(session, upstream_request, ctx)
                .await?;
        }
        Ok(())
    }

    /// Run on_response for each plugin.
    pub async fn run_response(
        &self,
        session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<()> {
        for plugin in &self.plugins {
            plugin.on_response(session, upstream_response, ctx).await?;
        }
        Ok(())
    }

    /// Run on_response_body for each plugin (sync).
    pub fn run_response_body(
        &self,
        session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<()> {
        for plugin in &self.plugins {
            plugin.on_response_body(session, body, end_of_stream, ctx)?;
        }
        Ok(())
    }

    /// Run on_logging for each plugin.
    pub async fn run_logging(
        &self,
        session: &mut Session,
        error: Option<&pingora::Error>,
        ctx: &mut PluginCtx,
    ) {
        for plugin in &self.plugins {
            plugin.on_logging(session, error, ctx).await;
        }
    }
}
