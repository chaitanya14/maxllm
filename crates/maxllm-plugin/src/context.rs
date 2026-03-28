// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use ahash::AHashMap;

/// Plugin-visible context carried through the request lifecycle.
/// Populated by the gateway before plugin chains run; plugins can read and write.
pub struct PluginCtx {
    /// Route path that matched (empty if pre-routing).
    pub route_path: String,
    /// Selected provider name.
    pub provider_name: String,
    /// Model from request body.
    pub model: String,
    /// Client identity set by auth plugins.
    pub client_id: Option<String>,
    /// Request ID set by request_id plugin.
    pub request_id: Option<String>,
    /// Client IP address.
    pub client_ip: Option<String>,
    /// Arbitrary key-value store for plugin-to-plugin data.
    pub extensions: AHashMap<String, String>,
}

impl PluginCtx {
    pub fn new() -> Self {
        Self {
            route_path: String::new(),
            provider_name: String::new(),
            model: String::new(),
            client_id: None,
            request_id: None,
            client_ip: None,
            extensions: AHashMap::new(),
        }
    }
}

impl Default for PluginCtx {
    fn default() -> Self {
        Self::new()
    }
}
