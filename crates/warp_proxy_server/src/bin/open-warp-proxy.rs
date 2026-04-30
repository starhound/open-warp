//! `open-warp-proxy` — entrypoint for the local proxy. Reads `OPEN_WARP_PROXY_BIND`
//! (default `127.0.0.1:8939`) and serves until killed.
//!
//! Provider config is not yet wired (Phase 2.0 scaffold). Future work: read provider
//! kind + endpoint + key from environment or `~/.config/open-warp/provider.toml`.

use std::net::SocketAddr;
use warp_proxy_server::{ProxyConfig, serve};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let bind = std::env::var("OPEN_WARP_PROXY_BIND").unwrap_or_else(|_| "127.0.0.1:8939".into());
    let addr: SocketAddr = bind.parse()?;

    serve(addr, ProxyConfig::default()).await
}
