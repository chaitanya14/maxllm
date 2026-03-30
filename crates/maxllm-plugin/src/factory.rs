// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use crate::builtin;
use crate::Plugin;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("missing 'category' field in plugin config")]
    MissingCategory,
    #[error("unknown plugin category: {0}")]
    UnknownCategory(String),
    #[error("plugin config error: {0}")]
    Config(String),
}

/// Create a plugin instance from a TOML config table.
/// The `category` field determines which plugin type to construct.
pub fn create_plugin(name: &str, config: &toml::Table) -> Result<Arc<dyn Plugin>, PluginError> {
    let category = config
        .get("category")
        .and_then(|v| v.as_str())
        .ok_or(PluginError::MissingCategory)?;

    match category {
        "key_auth" => Ok(Arc::new(builtin::KeyAuthPlugin::from_config(name, config)?)),
        "rate_limit" => Ok(Arc::new(builtin::RateLimitPlugin::from_config(
            name, config,
        )?)),
        "request_id" => Ok(Arc::new(builtin::RequestIdPlugin::from_config(
            name, config,
        )?)),
        "cors" => Ok(Arc::new(builtin::CorsPlugin::from_config(name, config)?)),
        "ip_restriction" => Ok(Arc::new(builtin::IpRestrictionPlugin::from_config(
            name, config,
        )?)),
        "cache" => Ok(Arc::new(builtin::CachePlugin::from_config(name, config)?)),
        "webhook" => Ok(Arc::new(builtin::WebhookPlugin::from_config(name, config)?)),
        "pii_filter" => Ok(Arc::new(builtin::PiiFilterPlugin::from_config(
            name, config,
        )?)),
        "keyword_block" => Ok(Arc::new(builtin::KeywordBlockPlugin::from_config(
            name, config,
        )?)),
        "max_size" => Ok(Arc::new(builtin::MaxSizePlugin::from_config(name, config)?)),
        "prompt_guard" => Ok(Arc::new(builtin::PromptGuardPlugin::from_config(
            name, config,
        )?)),
        "secret_scan" => Ok(Arc::new(builtin::SecretScanPlugin::from_config(
            name, config,
        )?)),
        "regex_guard" => Ok(Arc::new(builtin::RegexGuardPlugin::from_config(
            name, config,
        )?)),
        other => Err(PluginError::UnknownCategory(other.to_string())),
    }
}
