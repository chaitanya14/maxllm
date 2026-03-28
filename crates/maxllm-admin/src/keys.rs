// Copyright 2024 MaxLLM Contributors
// SPDX-License-Identifier: Apache-2.0

//! Virtual key lifecycle: creation, validation, revocation, and listing.
//!
//! Keys follow the format `sk-maxllm-{uuid}` and are hashed with SHA-256
//! for fast O(1) lookups.  The raw key is returned exactly once at creation
//! time and is never stored.

use chrono::{Duration, Utc};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::models::*;
use crate::store::{AdminStore, StoreError};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum KeyError {
    #[error("store error: {0}")]
    Store(#[from] StoreError),
    #[error("key expired")]
    Expired,
    #[error("key revoked")]
    Revoked,
    #[error("budget exceeded")]
    BudgetExceeded,
    #[error("key not found")]
    NotFound,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the SHA-256 hex digest of a raw key string.
pub fn hash_key(raw: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(raw.as_bytes());
    hex::encode(hasher.finalize())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate a new virtual key, persist it, and return the raw key exactly once.
pub fn generate_key(
    store: &dyn AdminStore,
    request: KeyCreateRequest,
) -> Result<KeyCreateResponse, KeyError> {
    let raw_key = format!("sk-maxllm-{}", Uuid::new_v4());
    let key_hash = hash_key(&raw_key);
    let key_prefix = raw_key[..12].to_string();
    let id = Uuid::new_v4().to_string();
    let now = Utc::now();

    let expires_at = request
        .expires_in_days
        .map(|d| now + Duration::days(i64::from(d)));

    let budget_reset_at = request
        .budget_reset_days
        .map(|d| now + Duration::days(i64::from(d)));

    let vk = VirtualKey {
        id: id.clone(),
        key_hash,
        key_prefix: key_prefix.clone(),
        name: request.name.clone(),
        team_id: request.team_id,
        allowed_models: request.allowed_models,
        max_budget_usd: request.max_budget_usd,
        budget_reset_days: request.budget_reset_days,
        budget_spent_usd: 0.0,
        budget_reset_at,
        rpm_limit: request.rpm_limit,
        tpm_limit: request.tpm_limit,
        max_parallel_requests: request.max_parallel_requests,
        expires_at,
        created_at: now,
        last_used_at: None,
        is_active: true,
        metadata: request.metadata.unwrap_or_default(),
        total_requests: 0,
        total_tokens_in: 0,
        total_tokens_out: 0,
        total_spend_usd: 0.0,
    };

    store.create_key(vk)?;

    tracing::info!(key_id = %id, key_prefix = %key_prefix, "virtual key created");

    Ok(KeyCreateResponse {
        key: raw_key,
        key_id: id,
        name: request.name,
        key_prefix,
    })
}

/// Validate a raw key: look it up by hash, check active/expiry/budget.
///
/// Returns `Ok(Some(key))` if valid, `Ok(None)` if the key does not exist,
/// or an error describing *why* the key is invalid.
pub fn validate_key(
    store: &dyn AdminStore,
    raw_key: &str,
) -> Result<Option<VirtualKey>, KeyError> {
    let key_hash = hash_key(raw_key);
    let key = match store.get_key_by_hash(&key_hash)? {
        Some(k) => k,
        None => return Ok(None),
    };

    if !key.is_active {
        return Err(KeyError::Revoked);
    }

    if let Some(exp) = key.expires_at {
        if Utc::now() > exp {
            return Err(KeyError::Expired);
        }
    }

    if let Some(max) = key.max_budget_usd {
        if key.budget_spent_usd >= max {
            return Err(KeyError::BudgetExceeded);
        }
    }

    Ok(Some(key))
}

/// Mark a key as inactive (soft delete).
pub fn revoke_key(store: &dyn AdminStore, key_id: &str) -> Result<bool, KeyError> {
    let key = match store.get_key_by_id(key_id)? {
        Some(mut k) => {
            k.is_active = false;
            store.update_key(k)?;
            true
        }
        None => false,
    };
    if key {
        tracing::info!(key_id = %key_id, "virtual key revoked");
    }
    Ok(key)
}

/// List keys with pagination.
pub fn list_keys(
    store: &dyn AdminStore,
    offset: usize,
    limit: usize,
) -> Result<Vec<VirtualKey>, KeyError> {
    Ok(store.list_keys(offset, limit)?)
}

/// Get a single key by ID.
pub fn get_key_info(
    store: &dyn AdminStore,
    key_id: &str,
) -> Result<Option<VirtualKey>, KeyError> {
    Ok(store.get_key_by_id(key_id)?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InMemoryStore;

    fn default_request(name: &str) -> KeyCreateRequest {
        KeyCreateRequest {
            name: name.to_string(),
            team_id: None,
            allowed_models: vec![],
            max_budget_usd: None,
            budget_reset_days: None,
            rpm_limit: None,
            tpm_limit: None,
            max_parallel_requests: None,
            expires_in_days: None,
            metadata: None,
        }
    }

    #[test]
    fn generate_and_validate() {
        let store = InMemoryStore::new();
        let resp = generate_key(&store, default_request("test")).unwrap();

        assert!(resp.key.starts_with("sk-maxllm-"));
        assert!(!resp.key_id.is_empty());

        let validated = validate_key(&store, &resp.key).unwrap().unwrap();
        assert_eq!(validated.id, resp.key_id);
        assert!(validated.is_active);
    }

    #[test]
    fn validate_unknown_key_returns_none() {
        let store = InMemoryStore::new();
        assert!(validate_key(&store, "sk-maxllm-bogus").unwrap().is_none());
    }

    #[test]
    fn revoke_prevents_validation() {
        let store = InMemoryStore::new();
        let resp = generate_key(&store, default_request("test")).unwrap();
        assert!(revoke_key(&store, &resp.key_id).unwrap());

        match validate_key(&store, &resp.key) {
            Err(KeyError::Revoked) => {} // expected
            other => panic!("expected Revoked, got {other:?}"),
        }
    }

    #[test]
    fn budget_exceeded() {
        let store = InMemoryStore::new();
        let req = KeyCreateRequest {
            max_budget_usd: Some(0.01),
            ..default_request("budget-test")
        };
        let resp = generate_key(&store, req).unwrap();

        // Simulate spending over budget.
        let mut key = store.get_key_by_id(&resp.key_id).unwrap().unwrap();
        key.budget_spent_usd = 0.02;
        store.update_key(key).unwrap();

        match validate_key(&store, &resp.key) {
            Err(KeyError::BudgetExceeded) => {}
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
    }

    #[test]
    fn expired_key() {
        let store = InMemoryStore::new();
        let resp = generate_key(&store, default_request("exp")).unwrap();

        // Set expiration to the past.
        let mut key = store.get_key_by_id(&resp.key_id).unwrap().unwrap();
        key.expires_at = Some(Utc::now() - Duration::hours(1));
        store.update_key(key).unwrap();

        match validate_key(&store, &resp.key) {
            Err(KeyError::Expired) => {}
            other => panic!("expected Expired, got {other:?}"),
        }
    }

    #[test]
    fn list_and_get() {
        let store = InMemoryStore::new();
        let r1 = generate_key(&store, default_request("a")).unwrap();
        let _r2 = generate_key(&store, default_request("b")).unwrap();

        assert_eq!(list_keys(&store, 0, 10).unwrap().len(), 2);
        assert!(get_key_info(&store, &r1.key_id).unwrap().is_some());
        assert!(get_key_info(&store, "nonexistent").unwrap().is_none());
    }
}
