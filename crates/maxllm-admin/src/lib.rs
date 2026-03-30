// Copyright 2024 MaxLLM Contributors
// SPDX-License-Identifier: Apache-2.0

//! `maxllm-admin` — virtual key management, cost tracking, and budget
//! enforcement for the MaxLLM AI gateway.
//!
//! This crate is framework-agnostic: it exposes plain Rust types and
//! functions that the gateway integration layer (`maxllm-gateway`) calls
//! from its Pingora request/response filters.
//!
//! # Quick start
//!
//! ```rust
//! use std::sync::Arc;
//! use maxllm_admin::{AdminApi, CostCalculator, InMemoryStore};
//!
//! let store = Arc::new(InMemoryStore::new());
//! let calc  = Arc::new(CostCalculator::new());
//! let api   = AdminApi::new(store, calc, "my-master-admin-key");
//!
//! let resp = api.handle_request("GET", "/admin/keys", &[], "my-master-admin-key");
//! assert_eq!(resp.status, 200);
//! ```

pub mod api;
pub mod budget;
pub mod costs;
pub mod keys;
pub mod models;
pub mod sqlite_store;
pub mod store;
pub mod teams;

// Re-exports for convenience.
pub use api::{AdminApi, ApiResponse};
pub use budget::BudgetEnforcer;
pub use costs::CostCalculator;
pub use models::*;
pub use sqlite_store::SqliteStore;
pub use store::{AdminStore, InMemoryStore};
