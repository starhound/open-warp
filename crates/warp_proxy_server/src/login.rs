//! OAuth 2.0 Authorization Code + PKCE login flow with OIDC discovery.
//!
//! `LoginConfig::from_env` builds a config from `OPEN_WARP_OIDC_*` variables; `run_login`
//! drives the flow: discover endpoints, open the user's browser to the authorize URL,
//! catch the redirect on `http://127.0.0.1:<port>/callback`, exchange the code for an
//! access token, and return it.
//!
//! The returned `access_token` is what `OPEN_WARP_BEARER_TOKEN` expects when the proxy
//! is run with the Anthropic provider in bearer-token mode.

use anyhow::{Context, Result, bail};
use axum::{
    Router,
    extract::{Query, State},
    response::Html,
    routing::get,
};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD as BASE64;
use rand::RngCore;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::oneshot;

const DEFAULT_SCOPE: &str = "openid profile email";
const DEFAULT_PORT: u16 = 7777;

#[derive(Debug, Clone)]
pub struct LoginConfig {
    pub issuer: Option<String>,
    pub client_id: String,
    pub client_secret: Option<String>,
    pub scope: String,
    pub redirect_port: u16,
    /// Optional explicit endpoints — if both are set we skip OIDC discovery.
    pub authorize_endpoint: Option<String>,
    pub token_endpoint: Option<String>,
}

impl LoginConfig {
    pub fn from_env() -> Result<Self> {
        let client_id = std::env::var("OPEN_WARP_OIDC_CLIENT_ID")
            .context("OPEN_WARP_OIDC_CLIENT_ID is required for login")?;
        let issuer = std::env::var("OPEN_WARP_OIDC_ISSUER").ok();
        let authorize_endpoint = std::env::var("OPEN_WARP_OIDC_AUTHORIZE_URL").ok();
        let token_endpoint = std::env::var("OPEN_WARP_OIDC_TOKEN_URL").ok();
        if issuer.is_none() && (authorize_endpoint.is_none() || token_endpoint.is_none()) {
            bail!(
                "either OPEN_WARP_OIDC_ISSUER, or both OPEN_WARP_OIDC_AUTHORIZE_URL and \
                 OPEN_WARP_OIDC_TOKEN_URL, must be set"
            );
        }
        Ok(Self {
            issuer,
            client_id,
            client_secret: std::env::var("OPEN_WARP_OIDC_CLIENT_SECRET").ok(),
            scope: std::env::var("OPEN_WARP_OIDC_SCOPE")
                .unwrap_or_else(|_| DEFAULT_SCOPE.to_string()),
            redirect_port: std::env::var("OPEN_WARP_OIDC_REDIRECT_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_PORT),
            authorize_endpoint,
            token_endpoint,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[serde(default)]
    pub token_type: String,
    #[serde(default)]
    pub expires_in: Option<u64>,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub id_token: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
}

/// Run the full login flow. Returns the token response on success.
pub async fn run_login(cfg: LoginConfig) -> Result<TokenResponse> {
    let http = reqwest::Client::new();

    let (authorize_endpoint, token_endpoint) = match (
        cfg.authorize_endpoint.clone(),
        cfg.token_endpoint.clone(),
    ) {
        (Some(a), Some(t)) => (a, t),
        _ => {
            let issuer = cfg.issuer.as_ref().context(
                "OPEN_WARP_OIDC_ISSUER required when authorize/token URLs aren't set",
            )?;
            discover(&http, issuer).await?
        }
    };

    let pkce = Pkce::generate();
    let state = random_state();
    let redirect_uri = format!("http://127.0.0.1:{}/callback", cfg.redirect_port);

    let authorize_url = build_authorize_url(
        &authorize_endpoint,
        &cfg.client_id,
        &cfg.scope,
        &redirect_uri,
        &pkce.challenge,
        &state,
    );

    let (code_tx, code_rx) = oneshot::channel::<CallbackResult>();
    let server = spawn_callback_server(cfg.redirect_port, state.clone(), code_tx).await?;

    println!();
    println!("Open this URL in your browser to log in:");
    println!();
    println!("    {authorize_url}");
    println!();
    println!("Listening on {redirect_uri} for the OAuth redirect…");
    let _ = try_open_browser(&authorize_url);

    let result = code_rx
        .await
        .context("callback server channel dropped before receiving a code")?;

    server.abort();

    let code = match result {
        CallbackResult::Code(code) => code,
        CallbackResult::Error { error, description } => bail!(
            "OAuth provider returned an error: {error}{}",
            description.map(|d| format!(" ({d})")).unwrap_or_default(),
        ),
        CallbackResult::StateMismatch => bail!(
            "OAuth state parameter mismatch — possible CSRF; aborting login",
        ),
    };

    let tokens = exchange_code(
        &http,
        &token_endpoint,
        &cfg.client_id,
        cfg.client_secret.as_deref(),
        &code,
        &redirect_uri,
        &pkce.verifier,
    )
    .await?;
    Ok(tokens)
}

fn try_open_browser(url: &str) -> Result<()> {
    use std::process::Command;
    // Best-effort: try common openers. Failures are silently ignored — we already
    // printed the URL.
    for prog in ["wslview", "xdg-open", "open", "explorer.exe"] {
        if Command::new(prog).arg(url).spawn().is_ok() {
            return Ok(());
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct DiscoveryDoc {
    authorization_endpoint: String,
    token_endpoint: String,
}

async fn discover(http: &reqwest::Client, issuer: &str) -> Result<(String, String)> {
    let url = format!(
        "{}/.well-known/openid-configuration",
        issuer.trim_end_matches('/')
    );
    let doc: DiscoveryDoc = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("failed to fetch {url}"))?
        .error_for_status()?
        .json()
        .await
        .context("failed to parse OIDC discovery document")?;
    Ok((doc.authorization_endpoint, doc.token_endpoint))
}

struct Pkce {
    verifier: String,
    challenge: String,
}

impl Pkce {
    fn generate() -> Self {
        let mut bytes = [0u8; 64];
        rand::thread_rng().fill_bytes(&mut bytes);
        let verifier = BASE64.encode(bytes);
        let challenge = BASE64.encode(Sha256::digest(verifier.as_bytes()));
        Self { verifier, challenge }
    }
}

fn random_state() -> String {
    let mut bytes = [0u8; 24];
    rand::thread_rng().fill_bytes(&mut bytes);
    BASE64.encode(bytes)
}

fn build_authorize_url(
    base: &str,
    client_id: &str,
    scope: &str,
    redirect_uri: &str,
    code_challenge: &str,
    state: &str,
) -> String {
    let q = [
        ("response_type", "code"),
        ("client_id", client_id),
        ("redirect_uri", redirect_uri),
        ("scope", scope),
        ("state", state),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
    ];
    let mut url = base.to_string();
    url.push(if base.contains('?') { '&' } else { '?' });
    let qs = q
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencode(v)))
        .collect::<Vec<_>>()
        .join("&");
    url.push_str(&qs);
    url
}

fn urlencode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

#[derive(Debug)]
enum CallbackResult {
    Code(String),
    Error {
        error: String,
        description: Option<String>,
    },
    StateMismatch,
}

#[derive(Clone)]
struct CallbackState {
    expected_state: String,
    sender: Arc<tokio::sync::Mutex<Option<oneshot::Sender<CallbackResult>>>>,
}

#[derive(Debug, Deserialize)]
struct CallbackQuery {
    #[serde(default)] code: Option<String>,
    #[serde(default)] state: Option<String>,
    #[serde(default)] error: Option<String>,
    #[serde(default)] error_description: Option<String>,
}

async fn spawn_callback_server(
    port: u16,
    expected_state: String,
    sender: oneshot::Sender<CallbackResult>,
) -> Result<tokio::task::JoinHandle<()>> {
    let state = CallbackState {
        expected_state,
        sender: Arc::new(tokio::sync::Mutex::new(Some(sender))),
    };
    let app = Router::new()
        .route("/callback", get(callback_handler))
        .with_state(state);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind callback server on {addr}"))?;
    Ok(tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    }))
}

async fn callback_handler(
    State(state): State<CallbackState>,
    Query(q): Query<CallbackQuery>,
) -> Html<&'static str> {
    let result = if let Some(error) = q.error {
        CallbackResult::Error { error, description: q.error_description }
    } else if q.state.as_deref() != Some(state.expected_state.as_str()) {
        CallbackResult::StateMismatch
    } else if let Some(code) = q.code {
        CallbackResult::Code(code)
    } else {
        CallbackResult::Error {
            error: "no_code".into(),
            description: Some("callback missing both `code` and `error`".into()),
        }
    };

    let mut guard = state.sender.lock().await;
    if let Some(tx) = guard.take() {
        let _ = tx.send(result);
    }

    Html(
        "<!doctype html><html><body style=\"font-family:system-ui,sans-serif;text-align:center;padding:64px;\">\
         <h1>open-warp login complete</h1>\
         <p>You can close this tab and return to the terminal.</p>\
         </body></html>",
    )
}

async fn exchange_code(
    http: &reqwest::Client,
    token_endpoint: &str,
    client_id: &str,
    client_secret: Option<&str>,
    code: &str,
    redirect_uri: &str,
    code_verifier: &str,
) -> Result<TokenResponse> {
    let mut params: Vec<(&str, &str)> = vec![
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("client_id", client_id),
        ("code_verifier", code_verifier),
    ];
    if let Some(secret) = client_secret {
        params.push(("client_secret", secret));
    }
    let response = http
        .post(token_endpoint)
        .form(&params)
        .send()
        .await
        .context("token exchange request failed")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("token endpoint returned HTTP {status}: {body}");
    }
    let tokens: TokenResponse = response
        .json()
        .await
        .context("failed to parse token response as JSON")?;
    Ok(tokens)
}
