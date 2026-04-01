// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

pub mod auto_compaction;
mod cache;
mod cors;
mod ip_restriction;
mod key_auth;
mod keyword_block;
mod max_size;
pub mod pii_filter;
pub mod prompt_guard;
mod rate_limit;
pub mod regex_guard;
mod request_id;
pub mod secret_scan;
mod webhook;

pub use auto_compaction::AutoCompactionPlugin;
pub use cache::CachePlugin;
pub use cors::CorsPlugin;
pub use ip_restriction::IpRestrictionPlugin;
pub use key_auth::KeyAuthPlugin;
pub use keyword_block::KeywordBlockPlugin;
pub use max_size::MaxSizePlugin;
pub use pii_filter::{GuardrailMode, PiiFilterPlugin, PiiMatch};
pub use prompt_guard::{InjectionMatch, PromptGuardPlugin};
pub use rate_limit::RateLimitPlugin;
pub use regex_guard::{RegexGuardMatch, RegexGuardPlugin};
pub use request_id::RequestIdPlugin;
pub use secret_scan::{SecretMatch, SecretScanPlugin};
pub use webhook::WebhookPlugin;
