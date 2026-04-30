//! Provider configuration. Phase 2.1 reads from environment variables; a config-file path
//! lands in a later phase.

use anyhow::{Context, Result, bail};

#[derive(Clone, Debug)]
pub struct ProxyConfig {
    pub provider: Provider,
}

#[derive(Clone, Debug)]
pub enum Provider {
    Anthropic {
        auth: AnthropicAuth,
        base_url: String,
        model: String,
    },
    OpenAiCompat {
        api_key: Option<String>,
        base_url: String,
        model: String,
    },
    Ollama {
        base_url: String,
        model: String,
    },
}

/// Anthropic authentication mode. The Messages API accepts either a direct API key
/// (`x-api-key` header, the standard `sk-ant-...` keys) or a bearer token (typically an
/// OAuth/OIDC-issued credential).
#[derive(Clone, Debug)]
pub enum AnthropicAuth {
    /// `x-api-key: <key>`
    ApiKey(String),
    /// `Authorization: Bearer <token>`
    BearerToken(String),
}

impl ProxyConfig {
    /// Build a [`ProxyConfig`] from environment variables.
    ///
    /// `OPEN_WARP_PROVIDER` selects the provider (`anthropic`, `openai_compat`, `ollama`). The
    /// remaining keys depend on the provider:
    ///
    /// - **anthropic**: one of `OPEN_WARP_API_KEY` (uses `x-api-key`) or
    ///   `OPEN_WARP_BEARER_TOKEN` (uses `Authorization: Bearer`, for OIDC/OAuth-issued
    ///   credentials). `OPEN_WARP_MODEL` (default `claude-sonnet-4-5`),
    ///   `OPEN_WARP_BASE_URL` (default `https://api.anthropic.com`).
    /// - **openai_compat**: `OPEN_WARP_BASE_URL` (required), `OPEN_WARP_MODEL` (required),
    ///   `OPEN_WARP_API_KEY` (optional).
    /// - **ollama**: `OPEN_WARP_BASE_URL` (default `http://127.0.0.1:11434`),
    ///   `OPEN_WARP_MODEL` (default `llama3.1`).
    pub fn from_env() -> Result<Self> {
        let kind = std::env::var("OPEN_WARP_PROVIDER")
            .ok()
            .unwrap_or_else(|| "anthropic".to_string())
            .to_ascii_lowercase();

        let provider = match kind.as_str() {
            "anthropic" => Provider::Anthropic {
                // Falls back to a placeholder when neither env var is set so the proxy can start
                // and rely entirely on the in-app BYOK key flowing in via request.settings.api_keys.
                auth: anthropic_auth_from_env().unwrap_or_else(|_| {
                    log::warn!(
                        "no OPEN_WARP_API_KEY or OPEN_WARP_BEARER_TOKEN set; relying on in-app BYOK \
                         key (configure in open-warp Settings → AI)."
                    );
                    AnthropicAuth::ApiKey(String::new())
                }),
                base_url: env_or("OPEN_WARP_BASE_URL", "https://api.anthropic.com"),
                model: env_or("OPEN_WARP_MODEL", "claude-sonnet-4-5"),
            },
            "openai_compat" | "openai-compat" | "openai" => Provider::OpenAiCompat {
                api_key: std::env::var("OPEN_WARP_API_KEY").ok(),
                base_url: env_required("OPEN_WARP_BASE_URL")
                    .context("openai_compat requires OPEN_WARP_BASE_URL")?,
                model: env_required("OPEN_WARP_MODEL")
                    .context("openai_compat requires OPEN_WARP_MODEL")?,
            },
            "ollama" => Provider::Ollama {
                base_url: env_or("OPEN_WARP_BASE_URL", "http://127.0.0.1:11434"),
                model: env_or("OPEN_WARP_MODEL", "llama3.1"),
            },
            other => bail!("unknown OPEN_WARP_PROVIDER {other:?}"),
        };

        Ok(Self { provider })
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_required(key: &str) -> Result<String> {
    let value = std::env::var(key).map_err(|_| anyhow::anyhow!("{key} is not set"))?;
    if value.trim().is_empty() {
        bail!("{key} is empty");
    }
    Ok(value)
}

fn anthropic_auth_from_env() -> Result<AnthropicAuth> {
    if let Ok(token) = std::env::var("OPEN_WARP_BEARER_TOKEN") {
        if !token.trim().is_empty() {
            return Ok(AnthropicAuth::BearerToken(token));
        }
    }
    let key = env_required("OPEN_WARP_API_KEY")?;
    Ok(AnthropicAuth::ApiKey(key))
}
