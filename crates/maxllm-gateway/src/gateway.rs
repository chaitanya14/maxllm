// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use crate::circuit_breaker::CircuitBreaker;
use crate::ctx::RequestCtx;
use crate::metrics::METRICS;
use crate::routing::{ProviderSelector, ProviderTarget};

use ahash::AHashMap;
use async_trait::async_trait;
use bytes::Bytes;
use maxllm_config::{Config, ProviderKind, RouteConfig};
use maxllm_plugin::guardrail::{
    self, GuardrailEngine, GuardrailInput, GuardrailOutput, GuardrailVerdict,
};
use maxllm_plugin::{Plugin, PluginChain};
use maxllm_translate::ProviderTranslator;
use pingora::http::ResponseHeader;
use pingora::prelude::*;
use pingora::proxy::{ProxyHttp, Session};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

/// Runtime state for a configured provider.
pub struct ProviderState {
    pub kind: ProviderKind,
    pub translator: Box<dyn ProviderTranslator>,
    pub circuit_breaker: CircuitBreaker,
    pub api_key: String,
    pub host: String,
    pub port: u16,
    pub tls: bool,
    pub sni: String,
    pub weight: u32,
    pub tags: Vec<String>,
    pub default_model: Option<String>,
}

/// The main AI gateway — implements Pingora's ProxyHttp trait.
pub struct AiGateway {
    pub providers: AHashMap<String, Arc<ProviderState>>,
    pub routes: Vec<RouteConfig>,
    pub metrics_enabled: bool,
    /// Global plugin chain — runs on every request before route-specific plugins.
    pub global_chain: PluginChain,
    /// Per-route plugin chains, indexed by route index.
    pub route_chains: Vec<PluginChain>,
    /// Model aliases: map requested model to actual model.
    pub model_aliases: HashMap<String, String>,
    /// Cost calculator for per-request cost tracking.
    pub cost_calculator: Arc<maxllm_admin::CostCalculator>,
    /// Budget enforcer for per-key budget tracking.
    pub budget_enforcer: Option<Arc<maxllm_admin::BudgetEnforcer>>,
    /// Guardrail engine for content-level inspection.
    pub guardrail_engine: Option<Arc<GuardrailEngine>>,
}

impl AiGateway {
    pub fn new(config: &Config) -> Result<Self, Box<dyn std::error::Error>> {
        let mut providers = AHashMap::new();

        for (name, conf) in &config.providers {
            let translator: Box<dyn ProviderTranslator> = match conf.kind {
                ProviderKind::OpenAI => {
                    Box::new(maxllm_translate::openai::OpenAITranslator)
                }
                ProviderKind::Anthropic => {
                    Box::new(maxllm_translate::anthropic::AnthropicTranslator)
                }
                ProviderKind::Gemini => {
                    Box::new(maxllm_translate::gemini::GeminiTranslator::new())
                }
                ProviderKind::AzureOpenai => {
                    let deployment = conf.deployment.as_deref().unwrap_or("gpt-4o");
                    let api_version = conf.api_version.as_deref().unwrap_or("2024-02-01");
                    Box::new(maxllm_translate::azure_openai::AzureOpenAITranslator::new(
                        deployment, api_version,
                    ))
                }
                ProviderKind::Bedrock => {
                    Box::new(maxllm_translate::bedrock::BedrockTranslator::new())
                }
                ProviderKind::Groq => {
                    Box::new(maxllm_translate::openai_compat::OpenAICompatTranslator::groq())
                }
                ProviderKind::Together => {
                    Box::new(maxllm_translate::openai_compat::OpenAICompatTranslator::together())
                }
                ProviderKind::Fireworks => {
                    Box::new(maxllm_translate::openai_compat::OpenAICompatTranslator::fireworks())
                }
                ProviderKind::DeepInfra => {
                    Box::new(maxllm_translate::openai_compat::OpenAICompatTranslator::deepinfra())
                }
                ProviderKind::Mistral => {
                    Box::new(maxllm_translate::openai_compat::OpenAICompatTranslator::mistral())
                }
                ProviderKind::XAI => {
                    Box::new(maxllm_translate::openai_compat::OpenAICompatTranslator::xai())
                }
                ProviderKind::DeepSeek => {
                    Box::new(maxllm_translate::openai_compat::OpenAICompatTranslator::deepseek())
                }
                ProviderKind::Ollama => {
                    Box::new(maxllm_translate::openai_compat::OpenAICompatTranslator::ollama())
                }
                ProviderKind::Cohere => {
                    Box::new(maxllm_translate::cohere::CohereTranslator)
                }
                ProviderKind::OpenaiCompat => {
                    let path = conf.upstream_path.as_deref().unwrap_or("/v1/chat/completions");
                    Box::new(maxllm_translate::openai_compat::OpenAICompatTranslator::new(
                        name.clone(),
                        path.to_string(),
                        maxllm_translate::openai_compat::AuthStyle::Bearer,
                    ))
                }
            };

            let url: url::Url = conf.base_url.parse().map_err(|e: url::ParseError| {
                format!("invalid base_url for provider '{}': {}", name, e)
            })?;

            let tls = url.scheme() == "https";
            let host = url.host_str().unwrap_or("localhost").to_string();
            let port = url.port().unwrap_or(if tls { 443 } else { 80 });

            providers.insert(
                name.clone(),
                Arc::new(ProviderState {
                    kind: conf.kind,
                    translator,
                    circuit_breaker: CircuitBreaker::new(conf.max_fails, conf.fail_timeout_secs),
                    api_key: conf.api_key.clone(),
                    host: host.clone(),
                    port,
                    tls,
                    sni: host,
                    weight: conf.weight,
                    tags: conf.tags.clone(),
                    default_model: conf.default_model.clone(),
                }),
            );
        }

        // Build plugin instances from config
        let mut plugin_instances: AHashMap<String, Arc<dyn Plugin>> = AHashMap::new();
        for (name, plugin_conf) in &config.plugins {
            let mut full_config = plugin_conf.params.clone();
            full_config.insert(
                "category".into(),
                toml::Value::String(plugin_conf.category.clone()),
            );
            let plugin = maxllm_plugin::create_plugin(name, &full_config)
                .map_err(|e| format!("failed to create plugin '{name}': {e}"))?;
            plugin_instances.insert(name.clone(), plugin);
        }

        // Build global plugin chain
        let global_plugins: Vec<Arc<dyn Plugin>> = config
            .global_plugins
            .iter()
            .filter_map(|name| plugin_instances.get(name).cloned())
            .collect();
        let global_chain = PluginChain::new(global_plugins);

        // Build per-route plugin chains
        let route_chains: Vec<PluginChain> = config
            .routes
            .iter()
            .map(|route| {
                let route_plugins: Vec<Arc<dyn Plugin>> = route
                    .plugins
                    .iter()
                    .filter_map(|name| plugin_instances.get(name).cloned())
                    .collect();
                PluginChain::new(route_plugins)
            })
            .collect();

        // Initialize cost calculator with config overrides
        let mut cost_calculator = maxllm_admin::CostCalculator::new();
        for (model, cost_conf) in &config.model_costs {
            cost_calculator.add_cost(maxllm_admin::ModelCost {
                model_pattern: model.clone(),
                input_cost_per_1m: cost_conf.input_per_1m,
                output_cost_per_1m: cost_conf.output_per_1m,
            });
        }
        let cost_calculator = Arc::new(cost_calculator);

        // Build guardrail engine from config
        let guardrail_engine = if config.guardrails.is_empty() {
            None
        } else {
            let engine = guardrail::build_engine(&config.guardrails)
                .map_err(|e| format!("failed to build guardrail engine: {e}"))?;
            info!(
                guardrails = config.guardrails.len(),
                "Guardrail engine initialized"
            );
            Some(Arc::new(engine))
        };

        info!(
            global_plugins = config.global_plugins.len(),
            total_plugins = plugin_instances.len(),
            providers = providers.len(),
            routes = config.routes.len(),
            model_aliases = config.model_aliases.len(),
            guardrails = config.guardrails.len(),
            "Gateway initialized"
        );

        Ok(Self {
            providers,
            routes: config.routes.clone(),
            metrics_enabled: config.metrics.enabled,
            global_chain,
            route_chains,
            model_aliases: config.model_aliases.clone(),
            cost_calculator,
            budget_enforcer: None,
            guardrail_engine,
        })
    }

    /// Find matching route by path prefix.
    fn match_route(&self, path: &str) -> Option<usize> {
        self.routes
            .iter()
            .position(|r| path.starts_with(&r.path))
    }

    /// Select provider using the route's configured strategy.
    fn select_provider(
        &self,
        route: &RouteConfig,
        ctx: &mut RequestCtx,
    ) -> Option<Arc<ProviderState>> {
        // Build list of all candidate providers for this route
        let mut candidates: Vec<ProviderTarget> = Vec::new();

        if let Some(provider) = self.providers.get(&route.provider) {
            candidates.push(ProviderTarget {
                name: route.provider.clone(),
                state: Arc::clone(provider),
                is_primary: true,
            });
        }

        for fb_name in &route.fallback {
            if let Some(provider) = self.providers.get(fb_name) {
                candidates.push(ProviderTarget {
                    name: fb_name.clone(),
                    state: Arc::clone(provider),
                    is_primary: false,
                });
            }
        }

        let selector = ProviderSelector::new(route.strategy);
        match selector.select(&candidates, ctx) {
            Some(target) => {
                if !target.is_primary {
                    ctx.fallback_used = true;
                    ctx.original_provider = Some(route.provider.clone());
                    METRICS
                        .fallbacks_total
                        .with_label_values(&[&route.provider, &target.name])
                        .inc();
                }
                ctx.provider_name = target.name.clone();
                Some(Arc::clone(&target.state))
            }
            None => None,
        }
    }

    /// Resolve model aliases.
    fn resolve_model_alias(&self, model: &str) -> String {
        self.model_aliases
            .get(model)
            .cloned()
            .unwrap_or_else(|| model.to_string())
    }

    async fn send_error_response(
        session: &mut Session,
        status: u16,
        message: &str,
    ) -> pingora::Result<bool> {
        let body = serde_json::json!({
            "error": {
                "message": message,
                "type": "gateway_error",
                "code": status
            }
        });
        let body_bytes = body.to_string().into_bytes();
        let content_len = body_bytes.len().to_string();

        let mut resp = ResponseHeader::build(status, Some(4))?;
        resp.insert_header("Content-Type", "application/json")?;
        resp.insert_header("Content-Length", &content_len)?;

        session.set_keepalive(None);
        session
            .write_response_header(Box::new(resp), false)
            .await?;
        session
            .write_response_body(Some(Bytes::from(body_bytes)), true)
            .await?;

        Ok(true)
    }
}

#[async_trait]
impl ProxyHttp for AiGateway {
    type CTX = RequestCtx;

    fn new_ctx(&self) -> Self::CTX {
        RequestCtx::new()
    }

    async fn request_filter(
        &self,
        session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> pingora::Result<bool> {
        let path = session.req_header().uri.path();

        // Health check endpoint — fast path, no allocations, no metrics overhead
        if path == "/health" || path == "/healthz" {
            static HEALTH_BODY: &[u8] = br#"{"status":"ok"}"#;
            let mut resp = ResponseHeader::build(200, Some(2))?;
            resp.insert_header("Content-Type", "application/json")?;
            resp.insert_header("Content-Length", "15")?;
            session
                .write_response_header(Box::new(resp), false)
                .await?;
            session
                .write_response_body(Some(Bytes::from_static(HEALTH_BODY)), true)
                .await?;
            ctx.skip_logging = true;
            return Ok(true);
        }

        METRICS.active_requests.inc();

        // Metrics endpoint (bypasses plugins)
        if path == "/metrics" && self.metrics_enabled {
            let body = METRICS.encode();
            let content_len = body.len().to_string();
            let mut resp = ResponseHeader::build(200, Some(2))?;
            resp.insert_header("Content-Type", "text/plain; version=0.0.4")?;
            resp.insert_header("Content-Length", &content_len)?;
            session
                .write_response_header(Box::new(resp), false)
                .await?;
            session
                .write_response_body(Some(Bytes::from(body)), true)
                .await?;
            METRICS.active_requests.dec();
            ctx.skip_logging = true;
            return Ok(true);
        }

        // Own the path for later use (after the borrow above is done)
        let path = path.to_string();

        // Admin API endpoints
        if path.starts_with("/admin/") {
            return self.handle_admin_request(session, &path).await;
        }

        // Extract client IP for plugins
        if let Some(addr) = session.client_addr() {
            if let Some(sock) = addr.as_inet() {
                ctx.plugin_ctx.client_ip = Some(sock.ip().to_string());
            }
        }

        // Run global plugin chain (auth, request_id, ip_restriction, etc.)
        if let Some(resp) = self.global_chain.run_request(session, &mut ctx.plugin_ctx).await? {
            METRICS.active_requests.dec();
            resp.send(session).await?;
            return Ok(true);
        }

        // Route matching
        let route_index = match self.match_route(&path) {
            Some(idx) => idx,
            None => {
                METRICS.active_requests.dec();
                return Self::send_error_response(
                    session,
                    404,
                    &format!("No route matches path: {path}"),
                )
                .await;
            }
        };

        ctx.route_index = Some(route_index);
        ctx.plugin_ctx.route_path = self.routes[route_index].path.clone();
        let route = &self.routes[route_index];

        // Store route-specific guardrail scope
        if !route.guardrails.is_empty() {
            ctx.route_guardrails = Some(route.guardrails.clone());
        }

        // Run route-specific plugin chain (rate_limit, cors, etc.)
        if let Some(resp) = self.route_chains[route_index]
            .run_request(session, &mut ctx.plugin_ctx)
            .await?
        {
            METRICS.active_requests.dec();
            resp.send(session).await?;
            return Ok(true);
        }

        // Select provider (with circuit breaker check and routing strategy)
        let _provider = match self.select_provider(route, ctx) {
            Some(p) => p,
            None => {
                METRICS.active_requests.dec();
                return Self::send_error_response(
                    session,
                    503,
                    "All providers are unavailable (circuit breakers open)",
                )
                .await;
            }
        };

        // Sync plugin ctx with request ctx
        ctx.plugin_ctx.provider_name = ctx.provider_name.clone();

        Ok(false)
    }

    async fn upstream_peer(
        &self,
        _session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> pingora::Result<Box<HttpPeer>> {
        let provider = self
            .providers
            .get(&ctx.provider_name)
            .expect("provider must exist after request_filter");

        let mut peer = HttpPeer::new(
            (&*provider.host, provider.port),
            provider.tls,
            provider.sni.clone(),
        );
        peer.options.connection_timeout = Some(std::time::Duration::from_secs(10));
        peer.options.read_timeout = Some(std::time::Duration::from_secs(120));
        peer.options.write_timeout = Some(std::time::Duration::from_secs(30));

        Ok(Box::new(peer))
    }

    async fn upstream_request_filter(
        &self,
        session: &mut Session,
        upstream_request: &mut RequestHeader,
        ctx: &mut Self::CTX,
    ) -> pingora::Result<()> {
        let provider = self
            .providers
            .get(&ctx.provider_name)
            .expect("provider must exist");

        // Set upstream path (Gemini needs dynamic path with model + query param auth)
        let path = if provider.kind == ProviderKind::Gemini {
            let model = provider.default_model.as_deref().unwrap_or("gemini-2.5-flash");
            format!("/v1beta/models/{}:generateContent?key={}", model, provider.api_key)
        } else {
            provider.translator.upstream_path().to_string()
        };
        upstream_request.set_uri(path.parse().expect("valid path"));

        // Remove downstream auth header — providers use their own auth
        let _ = upstream_request.remove_header("Authorization");

        // Set provider-specific headers
        for (key, value) in provider.translator.upstream_headers(&provider.api_key) {
            upstream_request.insert_header(key, &value)?;
        }

        // Set Host header
        upstream_request.insert_header("Host", &provider.host)?;

        // Remove the original Content-Length since body will be translated
        let _ = upstream_request.remove_header("Content-Length");
        upstream_request.insert_header("Transfer-Encoding", "chunked")?;

        // Run plugin chains on upstream request
        self.global_chain
            .run_upstream_request(session, upstream_request, &mut ctx.plugin_ctx)
            .await?;
        if let Some(idx) = ctx.route_index {
            self.route_chains[idx]
                .run_upstream_request(session, upstream_request, &mut ctx.plugin_ctx)
                .await?;
        }

        // Mark when we hand off to upstream — used to compute overhead
        ctx.upstream_send_time = Some(std::time::Instant::now());

        Ok(())
    }

    async fn request_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> pingora::Result<()> {
        // Buffer incoming body chunks
        if let Some(data) = body.take() {
            ctx.request_body_buf.extend_from_slice(&data);
        }

        if end_of_stream && !ctx.body_translated {
            ctx.body_translated = true;

            // Extract and resolve model name from request
            let parsed = serde_json::from_slice::<serde_json::Value>(&ctx.request_body_buf).ok();

            if let Some(ref parsed) = parsed {
                if let Some(model) = parsed.get("model").and_then(|m| m.as_str()) {
                    let resolved = self.resolve_model_alias(model);
                    ctx.model = resolved.clone();
                    ctx.plugin_ctx.model = resolved;
                }

                // Extract client-requested guardrails from request body
                if let Some(guardrails) = parsed.get("guardrails").and_then(|g| g.as_array()) {
                    ctx.requested_guardrails = Some(
                        guardrails
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect(),
                    );
                }
            }

            // Run pre-call guardrails on message content
            if let Some(ref engine) = self.guardrail_engine {
                if engine.has_pre_call() {
                    let content = parsed
                        .as_ref()
                        .map(guardrail::extract_message_content)
                        .unwrap_or_default();

                    if !content.is_empty() {
                        let input = GuardrailInput {
                            content: &content,
                            model: &ctx.model,
                            client_id: ctx.plugin_ctx.client_id.as_deref(),
                        };

                        let (verdict, applied) = engine
                            .run_pre_call(&input, ctx.requested_guardrails.as_deref(), ctx.route_guardrails.as_deref())
                            .await;
                        ctx.applied_guardrails = applied;

                        match verdict {
                            GuardrailVerdict::Block { ref guardrail, ref reason } => {
                                // Write guardrail error response directly to client
                                let error_body = serde_json::json!({
                                    "error": {
                                        "message": reason,
                                        "type": "guardrail_violation",
                                        "guardrail": guardrail,
                                        "code": 400
                                    }
                                });
                                let error_bytes = error_body.to_string().into_bytes();
                                let mut resp =
                                    pingora::http::ResponseHeader::build(400, Some(4))?;
                                resp.insert_header("Content-Type", "application/json")?;
                                resp.insert_header(
                                    "Content-Length",
                                    &error_bytes.len().to_string(),
                                )?;
                                resp.insert_header(
                                    "X-MaxLLM-Applied-Guardrails",
                                    &ctx.applied_guardrails.join(", "),
                                )?;
                                resp.insert_header(
                                    "X-MaxLLM-Guardrail-Blocked",
                                    guardrail,
                                )?;
                                _session
                                    .write_response_header(Box::new(resp), false)
                                    .await?;
                                _session
                                    .write_response_body(
                                        Some(Bytes::from(error_bytes)),
                                        true,
                                    )
                                    .await?;
                                _session.set_keepalive(None);
                                ctx.guardrail_blocked = Some(verdict);

                                return Err(pingora::Error::explain(
                                    pingora::ErrorType::HTTPStatus(400),
                                    "guardrail blocked request",
                                ));
                            }
                            GuardrailVerdict::Modify {
                                content: new_content,
                                ..
                            } => {
                                // Replace message content in request body with redacted version
                                if let Some(mut parsed) = parsed.clone() {
                                    if let Some(messages) = parsed
                                        .get_mut("messages")
                                        .and_then(|m| m.as_array_mut())
                                    {
                                        // Replace last user message content with redacted text
                                        for msg in messages.iter_mut().rev() {
                                            if msg.get("role").and_then(|r| r.as_str())
                                                == Some("user")
                                            {
                                                msg["content"] =
                                                    serde_json::Value::String(new_content.clone());
                                                break;
                                            }
                                        }
                                    }
                                    // Remove guardrails field before sending upstream
                                    if let Some(obj) = parsed.as_object_mut() {
                                        obj.remove("guardrails");
                                    }
                                    ctx.request_body_buf =
                                        serde_json::to_vec(&parsed).unwrap_or_else(|_| std::mem::take(&mut ctx.request_body_buf));
                                }
                            }
                            _ => {
                                // Pass or Log — remove guardrails field from body
                                if let Some(mut parsed) = parsed.clone() {
                                    if let Some(obj) = parsed.as_object_mut() {
                                        if obj.remove("guardrails").is_some() {
                                            ctx.request_body_buf = serde_json::to_vec(&parsed)
                                                .unwrap_or_else(|_| std::mem::take(&mut ctx.request_body_buf));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // No guardrail engine — still strip guardrails field from body
                if let Some(mut parsed) = parsed.clone() {
                    if let Some(obj) = parsed.as_object_mut() {
                        if obj.remove("guardrails").is_some() {
                            ctx.request_body_buf =
                                serde_json::to_vec(&parsed).unwrap_or_else(|_| std::mem::take(&mut ctx.request_body_buf));
                        }
                    }
                }
            }

            // Translate request body
            let provider = self
                .providers
                .get(&ctx.provider_name)
                .expect("provider must exist");

            match provider
                .translator
                .translate_request(&ctx.request_body_buf, None)
            {
                Ok(translated) => {
                    if translated.is_streaming {
                        ctx.is_streaming = true;
                        *ctx.stream_translator.lock().unwrap() =
                            Some(provider.translator.streaming_translator());
                    }
                    *body = Some(Bytes::from(translated.body));
                }
                Err(e) => {
                    warn!(error = %e, "Failed to translate request body");
                    *body = Some(Bytes::from(std::mem::take(&mut ctx.request_body_buf)));
                }
            }
        }

        Ok(())
    }

    async fn response_filter(
        &self,
        session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut Self::CTX,
    ) -> pingora::Result<()> {
        let status = upstream_response.status.as_u16();
        let provider = self.providers.get(&ctx.provider_name);

        // Record circuit breaker state
        if let Some(p) = provider {
            if (500..=599).contains(&status) || status == 429 {
                p.circuit_breaker.record_failure();
            } else if (200..=299).contains(&status) {
                p.circuit_breaker.record_success();
            }
        }

        // Only remove Content-Length for non-passthrough providers where body
        // translation changes the size. Passthrough (OpenAI) keeps the original
        // Content-Length for proper keep-alive and avoids chunked encoding overhead.
        if let Some(p) = provider {
            if p.translator.name() != "openai" {
                let _ = upstream_response.remove_header("Content-Length");
            }
        }

        // Add gateway headers
        upstream_response.insert_header("X-MaxLLM-Provider", &ctx.provider_name)?;
        if ctx.fallback_used {
            if let Some(original) = &ctx.original_provider {
                upstream_response.insert_header("X-MaxLLM-Fallback-From", original)?;
            }
        }

        // Add timing headers: upstream latency and gateway overhead
        if let Some(send_time) = ctx.upstream_send_time {
            let now = std::time::Instant::now();
            let upstream_ms = now.duration_since(send_time).as_millis();
            let total_ms = now.duration_since(ctx.request_start).as_millis();
            let overhead_ms = total_ms.saturating_sub(upstream_ms);

            upstream_response.insert_header("X-MaxLLM-Upstream-Ms", &upstream_ms.to_string())?;
            upstream_response.insert_header("X-MaxLLM-Overhead-Ms", &overhead_ms.to_string())?;
        }

        // Add applied guardrails header
        if !ctx.applied_guardrails.is_empty() {
            upstream_response.insert_header(
                "X-MaxLLM-Applied-Guardrails",
                &ctx.applied_guardrails.join(", "),
            )?;
        }

        // Run plugin chains on response
        self.global_chain
            .run_response(session, upstream_response, &mut ctx.plugin_ctx)
            .await?;
        if let Some(idx) = ctx.route_index {
            self.route_chains[idx]
                .run_response(session, upstream_response, &mut ctx.plugin_ctx)
                .await?;
        }

        Ok(())
    }

    fn upstream_response_body_filter(
        &self,
        session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> pingora::Result<Option<std::time::Duration>> {
        if ctx.is_streaming {
            if let Some(data) = body.as_ref() {
                let mut guard = ctx.stream_translator.lock().unwrap();
                if let Some(translator) = guard.as_mut() {
                    let translated = translator.process_chunk(data, end_of_stream);
                    drop(guard);
                    *body = Some(Bytes::from(translated));
                }
            }
        } else {
            if let Some(data) = body.take() {
                ctx.response_body_buf.extend_from_slice(&data);
            }

            if end_of_stream && !ctx.response_body_buf.is_empty() {
                if let Ok(parsed) =
                    serde_json::from_slice::<serde_json::Value>(&ctx.response_body_buf)
                {
                    extract_usage(&parsed, ctx);
                }

                let provider = self.providers.get(&ctx.provider_name);
                if let Some(p) = provider {
                    match p.translator.translate_response(&ctx.response_body_buf) {
                        Ok(mut translated) => {
                            // Run post-call guardrails on non-streaming response
                            if let Some(ref engine) = self.guardrail_engine {
                                if engine.has_post_call() {
                                    if let Ok(resp_parsed) =
                                        serde_json::from_slice::<serde_json::Value>(&translated)
                                    {
                                        let response_content = guardrail::extract_response_content(&resp_parsed);
                                        if !response_content.is_empty() {
                                            let output = GuardrailOutput {
                                                content: &response_content,
                                                model: &ctx.model,
                                            };
                                            let (verdict, post_applied) =
                                                tokio::task::block_in_place(|| {
                                                    tokio::runtime::Handle::current().block_on(
                                                        engine.run_post_call(
                                                            &output,
                                                            ctx.requested_guardrails.as_deref(),
                                                            ctx.route_guardrails.as_deref(),
                                                        ),
                                                    )
                                                });
                                            ctx.applied_guardrails.extend(post_applied);

                                            match verdict {
                                                GuardrailVerdict::Block { guardrail, reason } => {
                                                    let error_body = serde_json::json!({
                                                        "error": {
                                                            "message": reason,
                                                            "type": "guardrail_violation",
                                                            "guardrail": guardrail,
                                                            "code": 400
                                                        }
                                                    });
                                                    translated = error_body.to_string().into_bytes();
                                                }
                                                GuardrailVerdict::Modify { content, .. } => {
                                                    // Replace response content
                                                    let mut resp_json = resp_parsed;
                                                    if let Some(choices) = resp_json
                                                        .get_mut("choices")
                                                        .and_then(|c| c.as_array_mut())
                                                    {
                                                        if let Some(choice) = choices.first_mut() {
                                                            if let Some(msg) =
                                                                choice.get_mut("message")
                                                            {
                                                                msg["content"] =
                                                                    serde_json::Value::String(
                                                                        content,
                                                                    );
                                                            }
                                                        }
                                                    }
                                                    translated = serde_json::to_vec(&resp_json)
                                                        .unwrap_or(translated);
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }

                            *body = Some(Bytes::from(translated));
                        }
                        Err(e) => {
                            warn!(error = %e, "Failed to translate response body");
                            *body = Some(Bytes::from(std::mem::take(
                                &mut ctx.response_body_buf,
                            )));
                        }
                    }
                } else {
                    *body = Some(Bytes::from(std::mem::take(&mut ctx.response_body_buf)));
                }
            }
        }

        // Run plugin chains on response body
        self.global_chain
            .run_response_body(session, body, end_of_stream, &mut ctx.plugin_ctx)?;
        if let Some(idx) = ctx.route_index {
            self.route_chains[idx]
                .run_response_body(session, body, end_of_stream, &mut ctx.plugin_ctx)?;
        }

        Ok(None)
    }

    async fn logging(
        &self,
        session: &mut Session,
        e: Option<&pingora::Error>,
        ctx: &mut Self::CTX,
    ) {
        if ctx.skip_logging {
            return;
        }

        METRICS.active_requests.dec();

        let duration = ctx.request_start.elapsed();

        // Record metrics
        METRICS
            .request_duration_seconds
            .with_label_values(&[&ctx.provider_name])
            .observe(duration.as_secs_f64());

        if ctx.tokens_in > 0 || ctx.tokens_out > 0 {
            METRICS
                .tokens_in_total
                .with_label_values(&[&ctx.provider_name, &ctx.model])
                .inc_by(ctx.tokens_in);
            METRICS
                .tokens_out_total
                .with_label_values(&[&ctx.provider_name, &ctx.model])
                .inc_by(ctx.tokens_out);
        }

        // Calculate cost (skip when no tokens to avoid unnecessary lookups)
        let cost = if ctx.tokens_in > 0 || ctx.tokens_out > 0 {
            self.cost_calculator
                .calculate_cost(&ctx.model, ctx.tokens_in, ctx.tokens_out)
        } else {
            0.0
        };
        ctx.cost_usd = cost;

        // Run plugin logging chains
        self.global_chain
            .run_logging(session, e, &mut ctx.plugin_ctx)
            .await;
        if let Some(idx) = ctx.route_index {
            self.route_chains[idx]
                .run_logging(session, e, &mut ctx.plugin_ctx)
                .await;
        }

        info!(
            provider = %ctx.provider_name,
            model = %ctx.model,
            request_id = ?ctx.plugin_ctx.request_id,
            tokens_in = ctx.tokens_in,
            tokens_out = ctx.tokens_out,
            cost_usd = format!("{:.6}", cost),
            latency_ms = duration.as_millis() as u64,
            fallback = ctx.fallback_used,
            "request completed"
        );
    }
}

impl AiGateway {
    /// Handle admin API requests (/admin/*).
    async fn handle_admin_request(
        &self,
        session: &mut Session,
        _path: &str,
    ) -> pingora::Result<bool> {
        // Return 404 if admin is not configured
        let body = serde_json::json!({
            "error": {
                "message": "Admin API is not enabled. Configure [admin] in maxllm.toml.",
                "type": "gateway_error",
                "code": 404
            }
        });
        let body_bytes = body.to_string().into_bytes();
        let content_len = body_bytes.len().to_string();

        let mut resp = ResponseHeader::build(404, Some(2))?;
        resp.insert_header("Content-Type", "application/json")?;
        resp.insert_header("Content-Length", &content_len)?;
        session
            .write_response_header(Box::new(resp), false)
            .await?;
        session
            .write_response_body(Some(Bytes::from(body_bytes)), true)
            .await?;
        METRICS.active_requests.dec();
        Ok(true)
    }
}

/// Extract token usage from a response body.
fn extract_usage(body: &serde_json::Value, ctx: &mut RequestCtx) {
    if let Some(usage) = body.get("usage") {
        // OpenAI format
        if let Some(pt) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
            ctx.tokens_in = pt;
        }
        if let Some(ct) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
            ctx.tokens_out = ct;
        }
        // Anthropic format
        if let Some(it) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
            ctx.tokens_in = it;
        }
        if let Some(ot) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
            ctx.tokens_out = ot;
        }
    }

    // Extract model from response if not set
    if ctx.model.is_empty() {
        if let Some(model) = body.get("model").and_then(|m| m.as_str()) {
            ctx.model = model.to_string();
        }
    }
}
