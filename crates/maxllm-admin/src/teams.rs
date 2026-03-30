// Copyright 2024 MaxLLM Contributors
// SPDX-License-Identifier: Apache-2.0

//! Team management: creation, membership, and listing.

use chrono::Utc;
use uuid::Uuid;

use crate::models::*;
use crate::store::{AdminStore, StoreError};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum TeamError {
    #[error("store error: {0}")]
    Store(#[from] StoreError),
    #[error("team not found: {0}")]
    NotFound(String),
    #[error("key not found: {0}")]
    KeyNotFound(String),
    #[error("key already a member of this team")]
    AlreadyMember,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Create a new team.
pub fn create_team(store: &dyn AdminStore, request: TeamCreateRequest) -> Result<Team, TeamError> {
    let team = Team {
        id: Uuid::new_v4().to_string(),
        name: request.name,
        max_budget_usd: request.max_budget_usd,
        budget_spent_usd: 0.0,
        members: vec![],
        created_at: Utc::now(),
        metadata: request.metadata.unwrap_or_default(),
    };
    store.create_team(team.clone())?;
    tracing::info!(team_id = %team.id, name = %team.name, "team created");
    Ok(team)
}

/// Add a virtual key to a team.
pub fn add_member(store: &dyn AdminStore, team_id: &str, key_id: &str) -> Result<(), TeamError> {
    // Verify the key exists.
    let mut key = store
        .get_key_by_id(key_id)?
        .ok_or_else(|| TeamError::KeyNotFound(key_id.to_string()))?;

    let mut team = store
        .get_team(team_id)?
        .ok_or_else(|| TeamError::NotFound(team_id.to_string()))?;

    if team.members.contains(&key_id.to_string()) {
        return Err(TeamError::AlreadyMember);
    }

    team.members.push(key_id.to_string());
    store.update_team(team)?;

    key.team_id = Some(team_id.to_string());
    store.update_key(key).map_err(TeamError::Store)?;

    tracing::info!(team_id = %team_id, key_id = %key_id, "member added to team");
    Ok(())
}

/// Remove a virtual key from a team.
pub fn remove_member(store: &dyn AdminStore, team_id: &str, key_id: &str) -> Result<(), TeamError> {
    let mut team = store
        .get_team(team_id)?
        .ok_or_else(|| TeamError::NotFound(team_id.to_string()))?;

    let before = team.members.len();
    team.members.retain(|m| m != key_id);
    if team.members.len() == before {
        return Err(TeamError::KeyNotFound(key_id.to_string()));
    }
    store.update_team(team)?;

    // Clear the key's team_id if the key still exists.
    if let Some(mut key) = store.get_key_by_id(key_id)? {
        key.team_id = None;
        store.update_key(key).map_err(TeamError::Store)?;
    }

    tracing::info!(team_id = %team_id, key_id = %key_id, "member removed from team");
    Ok(())
}

/// List all teams.
pub fn list_teams(store: &dyn AdminStore) -> Result<Vec<Team>, TeamError> {
    Ok(store.list_teams()?)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys;
    use crate::models::KeyCreateRequest;
    use crate::store::InMemoryStore;

    fn make_store_with_key() -> (InMemoryStore, String) {
        let store = InMemoryStore::new();
        let resp = keys::generate_key(
            &store,
            KeyCreateRequest {
                name: "k1".into(),
                team_id: None,
                allowed_models: vec![],
                max_budget_usd: None,
                budget_reset_days: None,
                rpm_limit: None,
                tpm_limit: None,
                max_parallel_requests: None,
                expires_in_days: None,
                metadata: None,
            },
        )
        .unwrap();
        (store, resp.key_id)
    }

    #[test]
    fn create_team_and_add_member() {
        let (store, key_id) = make_store_with_key();
        let team = create_team(
            &store,
            TeamCreateRequest {
                name: "eng".into(),
                max_budget_usd: Some(500.0),
                metadata: None,
            },
        )
        .unwrap();

        add_member(&store, &team.id, &key_id).unwrap();

        let updated_team = store.get_team(&team.id).unwrap().unwrap();
        assert_eq!(updated_team.members, vec![key_id.clone()]);

        let updated_key = store.get_key_by_id(&key_id).unwrap().unwrap();
        assert_eq!(updated_key.team_id, Some(team.id.clone()));
    }

    #[test]
    fn duplicate_member_rejected() {
        let (store, key_id) = make_store_with_key();
        let team = create_team(
            &store,
            TeamCreateRequest {
                name: "eng".into(),
                max_budget_usd: None,
                metadata: None,
            },
        )
        .unwrap();
        add_member(&store, &team.id, &key_id).unwrap();
        assert!(matches!(
            add_member(&store, &team.id, &key_id),
            Err(TeamError::AlreadyMember)
        ));
    }

    #[test]
    fn remove_member_clears_team_id() {
        let (store, key_id) = make_store_with_key();
        let team = create_team(
            &store,
            TeamCreateRequest {
                name: "eng".into(),
                max_budget_usd: None,
                metadata: None,
            },
        )
        .unwrap();
        add_member(&store, &team.id, &key_id).unwrap();
        remove_member(&store, &team.id, &key_id).unwrap();

        let updated_team = store.get_team(&team.id).unwrap().unwrap();
        assert!(updated_team.members.is_empty());

        let updated_key = store.get_key_by_id(&key_id).unwrap().unwrap();
        assert!(updated_key.team_id.is_none());
    }

    #[test]
    fn list_teams_returns_all() {
        let store = InMemoryStore::new();
        for name in &["a", "b", "c"] {
            create_team(
                &store,
                TeamCreateRequest {
                    name: name.to_string(),
                    max_budget_usd: None,
                    metadata: None,
                },
            )
            .unwrap();
        }
        assert_eq!(list_teams(&store).unwrap().len(), 3);
    }
}
