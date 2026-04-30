//! `open-warp-proxy` — entrypoint for the local proxy.
//!
//! Reads configuration from environment variables (see [`ProxyConfig::from_env`]):
//!
//! ```text
//! OPEN_WARP_PROVIDER   anthropic | openai_compat | ollama   (default: anthropic)
//! OPEN_WARP_API_KEY    provider key (anthropic, openai_compat)
//! OPEN_WARP_BASE_URL   provider base URL (defaults vary by provider)
//! OPEN_WARP_MODEL      model name (defaults vary by provider)
//! OPEN_WARP_PROXY_BIND host:port to listen on (default 127.0.0.1:8939)
//! ```

use std::net::SocketAddr;
use warp_proxy_server::{ProxyConfig, serve};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Required by reqwest's rustls-tls-native-roots-no-provider feature: install the default
    // aws-lc-rs crypto provider before any TLS request goes out.
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("failed to install rustls crypto provider"))?;

    let bind = std::env::var("OPEN_WARP_PROXY_BIND").unwrap_or_else(|_| "127.0.0.1:8939".into());
    let addr: SocketAddr = bind.parse()?;
    let config = ProxyConfig::from_env()?;

    log::info!("provider configured: {:?}", redact_secrets(&config));
    serve(addr, config).await
}

/// Stringifies the provider config with API keys redacted, for logging.
fn redact_secrets(cfg: &ProxyConfig) -> String {
    use warp_proxy_server::config::AnthropicAuth;
    use warp_proxy_server::Provider;
    match &cfg.provider {
        Provider::Anthropic { auth, base_url, model } => {
            let auth_kind = match auth {
                AnthropicAuth::ApiKey(_) => "api_key",
                AnthropicAuth::BearerToken(_) => "bearer_token",
            };
            format!("Anthropic {{ base_url: {base_url:?}, model: {model:?}, auth: {auth_kind} }}")
        }
        Provider::OpenAiCompat {
            base_url,
            model,
            api_key,
        } => {
            format!(
                "OpenAiCompat {{ base_url: {base_url:?}, model: {model:?}, api_key: {} }}",
                if api_key.is_some() { "<set>" } else { "<unset>" }
            )
        }
        Provider::Ollama { base_url, model } => {
            format!("Ollama {{ base_url: {base_url:?}, model: {model:?} }}")
        }
    }
}
