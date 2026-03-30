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
use ipnet::IpNet;
use pingora::proxy::Session;
use std::net::IpAddr;

#[derive(Debug, Clone, Copy, PartialEq)]
enum RestrictionType {
    Allow,
    Deny,
}

/// IP restriction plugin. Allows or denies requests based on client IP.
pub struct IpRestrictionPlugin {
    name: String,
    restriction_type: RestrictionType,
    networks: Vec<IpNet>,
    message: String,
}

impl IpRestrictionPlugin {
    pub fn from_config(name: &str, config: &toml::Table) -> Result<Self, PluginError> {
        let restriction_type = match config
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("deny")
        {
            "allow" => RestrictionType::Allow,
            "deny" => RestrictionType::Deny,
            other => {
                return Err(PluginError::Config(format!(
                    "ip_restriction type must be 'allow' or 'deny', got '{other}'"
                )));
            }
        };

        let networks: Vec<IpNet> = config
            .get("ip_list")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .filter_map(|s| {
                        // Try parsing as network first, then as single IP
                        s.parse::<IpNet>()
                            .ok()
                            .or_else(|| s.parse::<IpAddr>().ok().map(IpNet::from))
                    })
                    .collect()
            })
            .unwrap_or_default();

        if networks.is_empty() {
            return Err(PluginError::Config(
                "ip_restriction requires at least one IP/CIDR in 'ip_list'".into(),
            ));
        }

        let message = config
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("Access denied")
            .to_string();

        Ok(Self {
            name: name.to_string(),
            restriction_type,
            networks,
            message,
        })
    }

    fn ip_matches(&self, ip: &IpAddr) -> bool {
        self.networks.iter().any(|net| net.contains(ip))
    }
}

#[async_trait]
impl Plugin for IpRestrictionPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    async fn on_request(
        &self,
        _session: &mut Session,
        ctx: &mut PluginCtx,
    ) -> pingora::Result<RequestAction> {
        let ip = match ctx
            .client_ip
            .as_deref()
            .and_then(|s| s.parse::<IpAddr>().ok())
        {
            Some(ip) => ip,
            None => return Ok(RequestAction::Continue),
        };

        let matched = self.ip_matches(&ip);

        let denied = match self.restriction_type {
            RestrictionType::Allow => !matched,
            RestrictionType::Deny => matched,
        };

        if denied {
            Ok(RequestAction::Respond(HttpResponse::json_error(
                403,
                &self.message,
            )))
        } else {
            Ok(RequestAction::Continue)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deny_list() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "ip_restriction".into());
        config.insert("type".into(), "deny".into());
        config.insert(
            "ip_list".into(),
            toml::Value::Array(vec!["10.0.0.0/8".into(), "192.168.1.1".into()]),
        );

        let plugin = IpRestrictionPlugin::from_config("ip_guard", &config).unwrap();
        assert!(plugin.ip_matches(&"10.0.0.5".parse().unwrap()));
        assert!(plugin.ip_matches(&"192.168.1.1".parse().unwrap()));
        assert!(!plugin.ip_matches(&"8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn test_allow_list() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "ip_restriction".into());
        config.insert("type".into(), "allow".into());
        config.insert(
            "ip_list".into(),
            toml::Value::Array(vec!["172.16.0.0/12".into()]),
        );

        let plugin = IpRestrictionPlugin::from_config("ip_guard", &config).unwrap();
        assert_eq!(plugin.restriction_type, RestrictionType::Allow);
        assert!(plugin.ip_matches(&"172.16.5.10".parse().unwrap()));
    }

    #[test]
    fn test_empty_ip_list_fails() {
        let mut config = toml::Table::new();
        config.insert("category".into(), "ip_restriction".into());
        config.insert("ip_list".into(), toml::Value::Array(vec![]));

        assert!(IpRestrictionPlugin::from_config("ip_guard", &config).is_err());
    }
}
