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

/// Extension key signaling that keyword checking is enabled for the request body.
/// The gateway should call `KeywordBlockPlugin::contains_blocked_keyword()`
/// on the parsed body content.
const EXT_KEYWORD_CHECK: &str = "keyword_check";

/// Keyword blocking plugin.
///
/// Blocks requests containing specified keywords or phrases. Since the
/// request body is not directly accessible in plugin hooks, this plugin
/// checks URL path and query parameters in `on_request` and exposes a
/// `contains_blocked_keyword()` utility for the gateway to use during
/// body processing.
pub struct KeywordBlockPlugin {
    name: String,
    keywords: Vec<String>,
    /// Lowercased keywords for case-insensitive matching.
    keywords_lower: Vec<String>,
    case_sensitive: bool,
}

impl KeywordBlockPlugin {
    pub fn from_config(name: &str, config: &toml::Table) -> Result<Self, PluginError> {
        let keywords: Vec<String> = config
            .get("keywords")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if keywords.is_empty() {
            return Err(PluginError::Config(
                "keyword_block requires at least one keyword".into(),
            ));
        }

        let case_sensitive = config
            .get("case_sensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let keywords_lower = keywords.iter().map(|k| k.to_lowercase()).collect();

        Ok(Self {
            name: name.to_string(),
            keywords,
            keywords_lower,
            case_sensitive,
        })
    }

    /// Check if the given text contains any blocked keyword.
    /// Returns the first matching keyword, or None if no match.
    pub fn contains_blocked_keyword<'a>(&'a self, text: &str) -> Option<&'a str> {
        if self.case_sensitive {
            for keyword in &self.keywords {
                if text.contains(keyword.as_str()) {
                    return Some(keyword);
                }
            }
        } else {
            let text_lower = text.to_lowercase();
            for (i, keyword_lower) in self.keywords_lower.iter().enumerate() {
                if text_lower.contains(keyword_lower.as_str()) {
                    return Some(&self.keywords[i]);
                }
            }
        }
        None
    }
}

#[async_trait]
impl Plugin for KeywordBlockPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_request(
        &self,
        session: &mut Session,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<RequestAction> {
        // Set flag so the gateway runs keyword checks during body processing.
        ctx.extensions
            .insert(EXT_KEYWORD_CHECK.into(), "enabled".into());

        // Check URL path.
        let path = session.req_header().uri.path().to_string();
        if let Some(keyword) = self.contains_blocked_keyword(&path) {
            tracing::warn!(
                plugin = self.name.as_str(),
                keyword = keyword,
                "blocked keyword detected in request path"
            );
            return Ok(RequestAction::Respond(HttpResponse::json_error(
                400,
                &format!("Request blocked: prohibited content detected"),
            )));
        }

        // Check query string.
        if let Some(query) = session.req_header().uri.query() {
            if let Some(keyword) = self.contains_blocked_keyword(query) {
                tracing::warn!(
                    plugin = self.name.as_str(),
                    keyword = keyword,
                    "blocked keyword detected in query string"
                );
                return Ok(RequestAction::Respond(HttpResponse::json_error(
                    400,
                    &format!("Request blocked: prohibited content detected"),
                )));
            }
        }

        Ok(RequestAction::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plugin(keywords: Vec<&str>, case_sensitive: bool) -> KeywordBlockPlugin {
        let mut config = toml::Table::new();
        config.insert("category".into(), "keyword_block".into());
        config.insert(
            "keywords".into(),
            toml::Value::Array(keywords.into_iter().map(|k| k.into()).collect()),
        );
        config.insert("case_sensitive".into(), case_sensitive.into());
        KeywordBlockPlugin::from_config("kw_guard", &config).unwrap()
    }

    #[test]
    fn test_from_config() {
        let plugin = make_plugin(vec!["DROP TABLE", "jailbreak"], false);
        assert_eq!(plugin.keywords.len(), 2);
        assert!(!plugin.case_sensitive);
    }

    #[test]
    fn test_empty_keywords_fails() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "keyword_block".into());
        config.insert("keywords".into(), toml::Value::Array(vec![]));

        assert!(KeywordBlockPlugin::from_config("kw", &config).is_err());
    }

    #[test]
    fn test_case_insensitive_match() {
        let plugin = make_plugin(vec!["DROP TABLE"], false);
        assert!(plugin
            .contains_blocked_keyword("please drop table users")
            .is_some());
        assert!(plugin
            .contains_blocked_keyword("DROP TABLE users")
            .is_some());
        assert!(plugin
            .contains_blocked_keyword("no bad words here")
            .is_none());
    }

    #[test]
    fn test_case_sensitive_match() {
        let plugin = make_plugin(vec!["DROP TABLE"], true);
        assert!(plugin
            .contains_blocked_keyword("DROP TABLE users")
            .is_some());
        assert!(plugin
            .contains_blocked_keyword("drop table users")
            .is_none());
    }

    #[test]
    fn test_multiple_keywords() {
        let plugin = make_plugin(vec!["jailbreak", "ignore previous"], false);
        assert_eq!(
            plugin.contains_blocked_keyword("try to jailbreak the model"),
            Some("jailbreak")
        );
        assert_eq!(
            plugin.contains_blocked_keyword("ignore previous instructions"),
            Some("ignore previous")
        );
        assert!(plugin.contains_blocked_keyword("normal request").is_none());
    }

    #[test]
    fn test_no_match_clean_text() {
        let plugin = make_plugin(vec!["malicious"], false);
        assert!(plugin
            .contains_blocked_keyword("This is a perfectly normal prompt about AI safety")
            .is_none());
    }
}
