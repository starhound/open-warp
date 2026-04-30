//! `open-warp-proxy` — the self-hosted Warp multi-agent API proxy and its companion
//! login flow.
//!
//! Subcommands:
//!
//! ```text
//! open-warp-proxy serve        Run the proxy server (default if no subcommand).
//! open-warp-proxy login        Run the OAuth/OIDC login flow and print a bearer token.
//! ```
//!
//! Environment variables:
//!
//! ```text
//! Provider config (serve):
//!   OPEN_WARP_PROVIDER          anthropic | openai_compat | ollama   (default: anthropic)
//!   OPEN_WARP_API_KEY           Anthropic API key (for x-api-key auth)
//!   OPEN_WARP_BEARER_TOKEN      Anthropic OAuth/OIDC bearer token (for `Authorization: Bearer`)
//!   OPEN_WARP_BASE_URL          provider base URL (defaults vary)
//!   OPEN_WARP_MODEL             model name (defaults vary)
//!   OPEN_WARP_PROXY_BIND        host:port to listen on (default 127.0.0.1:8939)
//!
//! Login config (login):
//!   OPEN_WARP_OIDC_ISSUER          OIDC issuer URL (uses /.well-known/openid-configuration)
//!   OPEN_WARP_OIDC_AUTHORIZE_URL   explicit authorize endpoint (skips discovery)
//!   OPEN_WARP_OIDC_TOKEN_URL       explicit token endpoint (skips discovery)
//!   OPEN_WARP_OIDC_CLIENT_ID       OAuth client id (required)
//!   OPEN_WARP_OIDC_CLIENT_SECRET   optional, for confidential clients
//!   OPEN_WARP_OIDC_SCOPE           scopes (default "openid profile email")
//!   OPEN_WARP_OIDC_REDIRECT_PORT   local port for the redirect catcher (default 7777)
//! ```

use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use warp_proxy_server::{ProxyConfig, login, serve};

#[derive(Parser)]
#[command(name = "open-warp-proxy")]
#[command(about = "Self-hosted proxy for Warp's multi-agent API")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the proxy server (this is the default if no subcommand is given).
    Serve,
    /// Run the OAuth/OIDC authorization-code+PKCE flow and print the resulting access token.
    Login {
        /// Print only the access token (no surrounding text). Useful in scripts:
        /// `OPEN_WARP_BEARER_TOKEN=$(open-warp-proxy login --raw)`
        #[arg(long)]
        raw: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| anyhow::anyhow!("failed to install rustls crypto provider"))?;

    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => run_serve().await,
        Command::Login { raw } => run_login_cmd(raw).await,
    }
}

async fn run_serve() -> anyhow::Result<()> {
    let bind = std::env::var("OPEN_WARP_PROXY_BIND").unwrap_or_else(|_| "127.0.0.1:8939".into());
    let addr: SocketAddr = bind.parse()?;
    let config = ProxyConfig::from_env()?;
    log::info!("provider configured: {}", redact_secrets(&config));
    serve(addr, config).await
}

async fn run_login_cmd(raw: bool) -> anyhow::Result<()> {
    let cfg = login::LoginConfig::from_env()?;
    let tokens = login::run_login(cfg).await?;
    if raw {
        println!("{}", tokens.access_token);
    } else {
        println!();
        println!("Login complete.");
        println!();
        println!("access_token:");
        println!("{}", tokens.access_token);
        println!();
        if let Some(expires_in) = tokens.expires_in {
            println!("expires_in: {expires_in}s");
        }
        if tokens.refresh_token.is_some() {
            println!("refresh_token: <issued, not printed>");
        }
        println!();
        println!("Use it with the proxy:");
        println!("    OPEN_WARP_BEARER_TOKEN={} open-warp-proxy serve", tokens.access_token);
    }
    Ok(())
}

/// Stringifies the provider config with API keys redacted, for logging.
fn redact_secrets(cfg: &ProxyConfig) -> String {
    use warp_proxy_server::Provider;
    use warp_proxy_server::config::AnthropicAuth;
    match &cfg.provider {
        Provider::Anthropic { auth, base_url, model } => {
            let auth_kind = match auth {
                AnthropicAuth::ApiKey(_) => "api_key",
                AnthropicAuth::BearerToken(_) => "bearer_token",
            };
            format!("Anthropic {{ base_url: {base_url:?}, model: {model:?}, auth: {auth_kind} }}")
        }
        Provider::OpenAiCompat { base_url, model, api_key } => format!(
            "OpenAiCompat {{ base_url: {base_url:?}, model: {model:?}, api_key: {} }}",
            if api_key.is_some() { "<set>" } else { "<unset>" }
        ),
        Provider::Ollama { base_url, model } => {
            format!("Ollama {{ base_url: {base_url:?}, model: {model:?} }}")
        }
    }
}
