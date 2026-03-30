// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod commands;

#[derive(Parser)]
#[command(name = "maxllm", version, about = "MaxLLM — AI Gateway CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the gateway server
    Start {
        /// Path to the configuration file
        #[arg(short, long, default_value = "maxllm.toml")]
        config: PathBuf,
        /// Port to listen on (overrides config)
        #[arg(short, long)]
        port: Option<u16>,
        /// Run as a background daemon
        #[arg(short, long)]
        daemon: bool,
    },
    /// Stop a running gateway
    Stop,
    /// Show gateway status
    Status,
    /// Check if the gateway is healthy
    Health {
        /// Gateway URL
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,
    },
    /// Generate a starter configuration file
    Init {
        /// Output file path
        #[arg(short, long, default_value = "maxllm.toml")]
        output: PathBuf,
    },
    /// Test connectivity to each configured provider
    Test {
        /// Path to the configuration file
        #[arg(short, long, default_value = "maxllm.toml")]
        config: PathBuf,
    },
    /// Manage virtual API keys
    Keys {
        #[command(subcommand)]
        command: KeysCommands,
    },
    /// Manage configuration
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },
}

#[derive(Subcommand)]
enum KeysCommands {
    /// List virtual keys
    List {
        /// Admin API URL
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// Master admin key
        #[arg(long, env = "MAXLLM_ADMIN_KEY")]
        admin_key: String,
    },
    /// Create a new virtual key
    Create {
        /// Human-readable name for the key
        #[arg(long)]
        name: String,
        /// Admin API URL
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// Master admin key
        #[arg(long, env = "MAXLLM_ADMIN_KEY")]
        admin_key: String,
    },
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Validate a configuration file
    Check {
        /// Path to the configuration file
        #[arg(short, long, default_value = "maxllm.toml")]
        config: PathBuf,
    },
    /// List configured providers
    Providers {
        /// Path to the configuration file
        #[arg(short, long, default_value = "maxllm.toml")]
        config: PathBuf,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let exit_code = match cli.command {
        Commands::Start { config, port, daemon } => {
            commands::start::run(config, port, daemon)
        }
        Commands::Stop => commands::stop::run(),
        Commands::Status => commands::status::run().await,
        Commands::Health { url } => commands::health::run(&url).await,
        Commands::Init { output } => commands::init::run(output),
        Commands::Test { config } => commands::test::run(config).await,
        Commands::Keys { command } => match command {
            KeysCommands::List { url, admin_key } => {
                commands::keys::list(&url, &admin_key).await
            }
            KeysCommands::Create { name, url, admin_key } => {
                commands::keys::create(&url, &admin_key, &name).await
            }
        },
        Commands::Config { command } => match command {
            ConfigCommands::Check { config } => commands::config::check(config),
            ConfigCommands::Providers { config } => commands::config::providers(config),
        },
    };

    std::process::exit(exit_code);
}
