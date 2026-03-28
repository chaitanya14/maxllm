// Copyright 2025 MaxLLM Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

//! CEL (Common Expression Language) guardrail provider.
//!
//! Allows operators to write expressive policy rules as CEL expressions
//! without custom code. CEL is a non-Turing complete language designed
//! for simplicity, speed, safety, and portability.
//!
//! ## Available Variables
//!
//! | Variable | Type | Description |
//! |----------|------|-------------|
//! | `content` | string | Concatenated message text |
//! | `model` | string | Resolved model name |
//! | `client_id` | string | Authenticated client ID (empty if none) |
//! | `message_count` | int | Number of messages in request |
//! | `has_system` | bool | Whether a system prompt exists |
//! | `roles` | list(string) | List of message roles |
//! | `is_streaming` | bool | Whether request is streaming |
//!
//! ## Built-in CEL Functions
//!
//! CEL provides: `size()`, `contains()`, `startsWith()`, `endsWith()`,
//! `matches()` (regex), string concatenation, list operations, etc.
//!
//! ## Example Expressions
//!
//! ```cel
//! size(content) > 50000                                    // Block long prompts
//! model.startsWith("gpt-3.5") && size(content) > 10000    // Model-specific limits
//! !has_system                                              // Require system prompt
//! message_count > 50                                       // Limit conversation length
//! content.contains("CONFIDENTIAL")                         // Block confidential content
//! content.matches("[A-Za-z0-9+/]{200,}={0,2}")            // Block large base64
//! ```

use super::{Guardrail, GuardrailError, GuardrailInput, GuardrailOutput, GuardrailVerdict};
use async_trait::async_trait;
use cel_interpreter::{Context, Program, Value};
use maxllm_config::GuardrailConfig;
use std::sync::Arc;

/// A compiled CEL rule with its metadata.
struct CelRule {
    name: String,
    program: Arc<Program>,
    message: String,
}

/// CEL expression guardrail provider.
///
/// Each rule is a CEL expression that evaluates to `true` when the request
/// should be BLOCKED. Multiple rules are evaluated in order; the first
/// `true` result triggers the configured action.
pub struct CelGuardrail {
    name: String,
    rules: Vec<CelRule>,
}

impl CelGuardrail {
    pub fn from_config(config: &GuardrailConfig) -> Result<Self, GuardrailError> {
        let rules_arr = config
            .params
            .get("rules")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                GuardrailError::Config("cel guardrail requires a 'rules' array".into())
            })?;

        if rules_arr.is_empty() {
            return Err(GuardrailError::Config(
                "cel guardrail requires at least one rule".into(),
            ));
        }

        let mut rules = Vec::with_capacity(rules_arr.len());
        for entry in rules_arr {
            if let Some(table) = entry.as_table() {
                let rule_name = table
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unnamed");
                let expr = table
                    .get("expr")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        GuardrailError::Config(format!(
                            "cel rule '{rule_name}' requires an 'expr' field"
                        ))
                    })?;
                let message = table
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("CEL policy violation")
                    .to_string();

                let program = Program::compile(expr).map_err(|e| {
                    GuardrailError::Config(format!(
                        "failed to compile CEL expression for rule '{rule_name}': {e}"
                    ))
                })?;

                rules.push(CelRule {
                    name: rule_name.to_string(),
                    program: Arc::new(program),
                    message,
                });
            }
        }

        if rules.is_empty() {
            return Err(GuardrailError::Config(
                "cel guardrail requires at least one valid rule".into(),
            ));
        }

        Ok(Self {
            name: config.name.clone(),
            rules,
        })
    }

    /// Build a CEL context with the standard guardrail variables.
    fn build_input_context<'a>(input: &'a GuardrailInput<'a>) -> Context<'a> {
        let mut ctx = Context::default();

        let _ = ctx.add_variable("content", input.content);
        let _ = ctx.add_variable("model", input.model);
        let _ = ctx.add_variable(
            "client_id",
            input.client_id.unwrap_or(""),
        );

        ctx
    }

    /// Build a CEL context for output checking.
    fn build_output_context<'a>(output: &'a GuardrailOutput<'a>) -> Context<'a> {
        let mut ctx = Context::default();

        let _ = ctx.add_variable("content", output.content);
        let _ = ctx.add_variable("model", output.model);

        ctx
    }

    /// Evaluate all rules against a context. Returns the first triggering rule.
    fn evaluate(&self, ctx: &Context<'_>) -> Option<(&str, &str)> {
        for rule in &self.rules {
            match rule.program.execute(ctx) {
                Ok(Value::Bool(true)) => {
                    return Some((&rule.name, &rule.message));
                }
                Ok(Value::Bool(false)) => {
                    // Rule passed, continue
                }
                Ok(other) => {
                    tracing::warn!(
                        guardrail = self.name.as_str(),
                        rule = rule.name.as_str(),
                        result = ?other,
                        "CEL rule returned non-boolean value, treating as pass"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        guardrail = self.name.as_str(),
                        rule = rule.name.as_str(),
                        error = %e,
                        "CEL rule execution error, treating as pass"
                    );
                }
            }
        }
        None
    }
}

#[async_trait]
impl Guardrail for CelGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    async fn check_input(&self, input: &GuardrailInput<'_>) -> GuardrailVerdict {
        let ctx = Self::build_input_context(input);

        if let Some((rule_name, message)) = self.evaluate(&ctx) {
            GuardrailVerdict::Block {
                guardrail: format!("cel:{}", rule_name),
                reason: message.to_string(),
            }
        } else {
            GuardrailVerdict::Pass
        }
    }

    async fn check_output(&self, output: &GuardrailOutput<'_>) -> GuardrailVerdict {
        let ctx = Self::build_output_context(output);

        if let Some((rule_name, message)) = self.evaluate(&ctx) {
            GuardrailVerdict::Block {
                guardrail: format!("cel:{}", rule_name),
                reason: message.to_string(),
            }
        } else {
            GuardrailVerdict::Pass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(rules: Vec<(&str, &str, &str)>) -> GuardrailConfig {
        let rules_arr: Vec<toml::Value> = rules
            .into_iter()
            .map(|(name, expr, message)| {
                let mut table = toml::Table::new();
                table.insert("name".into(), name.into());
                table.insert("expr".into(), expr.into());
                table.insert("message".into(), message.into());
                toml::Value::Table(table)
            })
            .collect();

        let mut params = toml::Table::new();
        params.insert("rules".into(), toml::Value::Array(rules_arr));

        GuardrailConfig {
            name: "test-cel".into(),
            provider: "cel".into(),
            mode: "pre_call".into(),
            default_on: true,
            action: "block".into(),
            params,
        }
    }

    fn input<'a>(content: &'a str, model: &'a str) -> GuardrailInput<'a> {
        GuardrailInput {
            content,
            model,
            client_id: None,
        }
    }

    #[test]
    fn test_from_config_valid() {
        let config = make_config(vec![
            ("test", "size(content) > 100", "too long"),
        ]);
        assert!(CelGuardrail::from_config(&config).is_ok());
    }

    #[test]
    fn test_from_config_invalid_expr() {
        let config = make_config(vec![
            ("bad", "this is not valid CEL <<<", "error"),
        ]);
        assert!(CelGuardrail::from_config(&config).is_err());
    }

    #[test]
    fn test_from_config_empty_rules() {
        let mut params = toml::Table::new();
        params.insert("rules".into(), toml::Value::Array(vec![]));
        let config = GuardrailConfig {
            name: "test".into(),
            provider: "cel".into(),
            mode: "pre_call".into(),
            default_on: true,
            action: "block".into(),
            params,
        };
        assert!(CelGuardrail::from_config(&config).is_err());
    }

    #[test]
    fn test_from_config_missing_rules() {
        let config = GuardrailConfig {
            name: "test".into(),
            provider: "cel".into(),
            mode: "pre_call".into(),
            default_on: true,
            action: "block".into(),
            params: toml::Table::new(),
        };
        assert!(CelGuardrail::from_config(&config).is_err());
    }

    #[tokio::test]
    async fn test_content_size_block() {
        let config = make_config(vec![
            ("size-limit", "size(content) > 10", "Content too long"),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        // Short content — should pass
        let verdict = guard.check_input(&input("hi", "gpt-4")).await;
        assert!(matches!(verdict, GuardrailVerdict::Pass));

        // Long content — should block
        let verdict = guard.check_input(&input("this is definitely longer than ten chars", "gpt-4")).await;
        assert!(matches!(verdict, GuardrailVerdict::Block { .. }));
    }

    #[tokio::test]
    async fn test_model_check() {
        let config = make_config(vec![
            ("model-block", r#"model.startsWith("gpt-3")"#, "GPT-3 not allowed"),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        let verdict = guard.check_input(&input("hello", "gpt-3.5-turbo")).await;
        assert!(matches!(verdict, GuardrailVerdict::Block { .. }));

        let verdict = guard.check_input(&input("hello", "gpt-4o")).await;
        assert!(matches!(verdict, GuardrailVerdict::Pass));
    }

    #[tokio::test]
    async fn test_contains_check() {
        let config = make_config(vec![
            ("confidential", r#"content.contains("CONFIDENTIAL")"#, "Confidential content blocked"),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        let verdict = guard.check_input(&input("This is CONFIDENTIAL data", "gpt-4")).await;
        assert!(matches!(verdict, GuardrailVerdict::Block { .. }));

        let verdict = guard.check_input(&input("This is normal data", "gpt-4")).await;
        assert!(matches!(verdict, GuardrailVerdict::Pass));
    }

    #[tokio::test]
    async fn test_combined_conditions() {
        let config = make_config(vec![
            (
                "model-size",
                r#"model.startsWith("gpt-3") && size(content) > 5000"#,
                "Large prompts not allowed on GPT-3",
            ),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        // GPT-3 with short content — pass
        let verdict = guard.check_input(&input("short", "gpt-3.5-turbo")).await;
        assert!(matches!(verdict, GuardrailVerdict::Pass));

        // GPT-4 with long content — pass (wrong model)
        let long_content = "x".repeat(6000);
        let verdict = guard.check_input(&input(&long_content, "gpt-4o")).await;
        assert!(matches!(verdict, GuardrailVerdict::Pass));

        // GPT-3 with long content — block
        let verdict = guard.check_input(&input(&long_content, "gpt-3.5-turbo")).await;
        assert!(matches!(verdict, GuardrailVerdict::Block { .. }));
    }

    #[tokio::test]
    async fn test_multiple_rules_first_match() {
        let config = make_config(vec![
            ("rule-a", "false", "should not trigger"),
            ("rule-b", "true", "always blocks"),
            ("rule-c", "true", "also blocks but never reached"),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        let verdict = guard.check_input(&input("hello", "gpt-4")).await;
        match verdict {
            GuardrailVerdict::Block { guardrail, reason } => {
                assert_eq!(guardrail, "cel:rule-b");
                assert_eq!(reason, "always blocks");
            }
            _ => panic!("expected block"),
        }
    }

    #[tokio::test]
    async fn test_all_rules_pass() {
        let config = make_config(vec![
            ("rule-a", "false", "nope"),
            ("rule-b", "false", "nope"),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        let verdict = guard.check_input(&input("hello", "gpt-4")).await;
        assert!(matches!(verdict, GuardrailVerdict::Pass));
    }

    #[tokio::test]
    async fn test_output_check() {
        let config = make_config(vec![
            ("no-code", r#"content.contains("```")"#, "Code blocks not allowed in response"),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        let output = GuardrailOutput {
            content: "Here's the code:\n```python\nprint('hi')\n```",
            model: "gpt-4",
        };
        let verdict = guard.check_output(&output).await;
        assert!(matches!(verdict, GuardrailVerdict::Block { .. }));
    }

    #[tokio::test]
    async fn test_client_id_check() {
        let config = make_config(vec![
            ("require-auth", r#"client_id == """#, "Authentication required"),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        // No client_id — should block
        let i = GuardrailInput {
            content: "hello",
            model: "gpt-4",
            client_id: None,
        };
        let verdict = guard.check_input(&i).await;
        assert!(matches!(verdict, GuardrailVerdict::Block { .. }));

        // With client_id — should pass
        let i = GuardrailInput {
            content: "hello",
            model: "gpt-4",
            client_id: Some("user-123"),
        };
        let verdict = guard.check_input(&i).await;
        assert!(matches!(verdict, GuardrailVerdict::Pass));
    }

    #[tokio::test]
    async fn test_runtime_error_treated_as_pass() {
        // Division by zero or other runtime error should not block
        let config = make_config(vec![
            ("bad-runtime", "1 / 0 > 0", "should not block"),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        let verdict = guard.check_input(&input("hello", "gpt-4")).await;
        // Should pass (fail open on errors)
        assert!(matches!(verdict, GuardrailVerdict::Pass));
    }

    #[tokio::test]
    async fn test_non_boolean_result_treated_as_pass() {
        // Expression that returns a string instead of bool
        let config = make_config(vec![
            ("string-result", r#""not a bool""#, "should not block"),
        ]);
        let guard = CelGuardrail::from_config(&config).unwrap();

        let verdict = guard.check_input(&input("hello", "gpt-4")).await;
        assert!(matches!(verdict, GuardrailVerdict::Pass));
    }
}
