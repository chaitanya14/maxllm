// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

use std::collections::HashSet;

/// Check if the request has a valid API key in the Authorization header.
/// Returns the bearer token if valid, None if invalid.
pub fn check_auth(auth_header: Option<&str>, valid_keys: &HashSet<String>) -> Option<String> {
    let header = auth_header?;
    let token = header.strip_prefix("Bearer ")?;
    if valid_keys.contains(token) {
        Some(token.to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_key() {
        let keys: HashSet<String> = ["key1".to_string()].into();
        assert_eq!(
            check_auth(Some("Bearer key1"), &keys),
            Some("key1".to_string())
        );
    }

    #[test]
    fn test_invalid_key() {
        let keys: HashSet<String> = ["key1".to_string()].into();
        assert_eq!(check_auth(Some("Bearer wrong"), &keys), None);
    }

    #[test]
    fn test_missing_header() {
        let keys: HashSet<String> = ["key1".to_string()].into();
        assert_eq!(check_auth(None, &keys), None);
    }

    #[test]
    fn test_no_bearer_prefix() {
        let keys: HashSet<String> = ["key1".to_string()].into();
        assert_eq!(check_auth(Some("key1"), &keys), None);
    }
}
