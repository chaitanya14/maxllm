// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

mod circuit_breaker;
mod ctx;
mod gateway;
mod metrics;
mod routing;

use clap::Parser;
use maxllm_config::Config;
use pingora::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "maxllm", version, about = "MaxLLM - AI Gateway built on Pingora")]
struct Args {
    /// Path to the configuration file
    #[arg(short, long, default_value = "maxllm.toml")]
    config: PathBuf,

    /// Enable daemon mode
    #[arg(short, long)]
    daemon: bool,
}

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    // Load configuration
    let config = Config::from_file(&args.config).unwrap_or_else(|e| {
        eprintln!("Failed to load config from {:?}: {e}", args.config);
        std::process::exit(1);
    });

    tracing::info!(
        listen = %config.server.listen,
        providers = config.providers.len(),
        routes = config.routes.len(),
        "Starting MaxLLM gateway"
    );

    // Build the gateway
    let gateway = gateway::AiGateway::new(&config).unwrap_or_else(|e| {
        eprintln!("Failed to initialize gateway: {e}");
        std::process::exit(1);
    });

    // Configure Pingora server
    let mut opt = Opt::default();
    if args.daemon {
        opt.daemon = true;
    }

    let mut server = Server::new(Some(opt)).expect("Failed to create Pingora server");

    // Apply server configuration from maxllm config
    {
        let conf = Arc::get_mut(&mut server.configuration)
            .expect("configuration must be uniquely owned before bootstrap");
        if let Some(threads) = config.server.threads {
            conf.threads = threads;
        }
        conf.work_stealing = true;
    }

    server.bootstrap();

    // Create the HTTP proxy service
    let mut service = http_proxy_service(&server.configuration, gateway);
    service.add_tcp(&config.server.listen.to_string());

    server.add_service(service);

    tracing::info!("MaxLLM gateway is running on {}", config.server.listen);
    server.run_forever();
}
