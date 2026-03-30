// Copyright 2025 MaxLLM Contributors.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

/// Validate a configuration file.
pub fn check(config_path: PathBuf) -> i32 {
    if !config_path.exists() {
        eprintln!("Error: config file not found: {}", config_path.display());
        return 1;
    }

    match maxllm_config::Config::from_file(&config_path) {
        Ok(config) => {
            println!("Configuration is valid.");
            println!();
            println!("  Listen:     {}", config.server.listen);
            println!("  Providers:  {}", config.providers.len());
            println!("  Routes:     {}", config.routes.len());
            println!("  Plugins:    {}", config.plugins.len());
            println!("  Guardrails: {}", config.guardrails.len());
            if !config.model_aliases.is_empty() {
                println!("  Aliases:    {}", config.model_aliases.len());
            }
            if config.admin.is_some() {
                println!("  Admin API:  enabled");
            }
            0
        }
        Err(e) => {
            eprintln!("Configuration error: {e}");
            1
        }
    }
}

/// List configured providers.
pub fn providers(config_path: PathBuf) -> i32 {
    if !config_path.exists() {
        eprintln!("Error: config file not found: {}", config_path.display());
        return 1;
    }

    let config = match maxllm_config::Config::from_file(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading config: {e}");
            return 1;
        }
    };

    if config.providers.is_empty() {
        println!("No providers configured.");
        return 0;
    }

    println!(
        "{:<20} {:<15} {:<40} DEFAULT MODEL",
        "NAME", "KIND", "BASE URL"
    );
    println!("{}", "-".repeat(90));

    let mut providers: Vec<_> = config.providers.iter().collect();
    providers.sort_by_key(|(name, _)| (*name).clone());

    for (name, provider) in providers {
        println!(
            "{:<20} {:<15} {:<40} {}",
            name,
            format!("{:?}", provider.kind).to_lowercase(),
            provider.base_url,
            provider.default_model.as_deref().unwrap_or("-"),
        );
    }

    0
}
