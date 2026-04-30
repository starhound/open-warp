//! `warp_proxy_server` — self-hosted proxy implementing Warp's multi-agent API
//! over HTTP+SSE+protobuf. Translates incoming Warp `Request` protos to user-configured
//! LLM provider calls (Ollama, Anthropic, OpenAI-compatible) and emits Warp `ResponseEvent`
//! protos as SSE.
//!
//! Phase 2.0 status: scaffold only. The handler accepts requests, decodes the proto, logs
//! it, and returns a `StreamFinished` event with an error message. Translation is a TODO
//! tracked under Phase 2.1+.

use anyhow::Result;
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Event},
    routing::post,
};
use bytes::Bytes;
use prost::Message;
use std::{net::SocketAddr, sync::Arc};
use warp_multi_agent_api as api;

#[derive(Clone, Default)]
pub struct ProxyConfig {
    /// Default provider to route requests to. Phase 2.0 is unimplemented; this field
    /// is a placeholder for the upcoming provider config (Ollama URL, Anthropic key, etc.).
    pub provider: ProviderKind,
}

#[derive(Clone, Debug, Default)]
pub enum ProviderKind {
    /// No provider configured. Returns a `StreamFinished` error event.
    #[default]
    Unconfigured,
    /// Ollama-compatible HTTP endpoint.
    Ollama { base_url: String, model: String },
    /// Anthropic Messages API.
    Anthropic { api_key: String, model: String },
    /// OpenAI Chat Completions–compatible endpoint (OpenAI, OpenRouter, vLLM, LM Studio…).
    OpenAiCompat {
        base_url: String,
        api_key: Option<String>,
        model: String,
    },
}

/// Build the axum `Router` for the proxy. The caller is responsible for binding it to a
/// listener.
pub fn router(config: ProxyConfig) -> Router {
    Router::new()
        .route("/ai/multi-agent", post(handle_multi_agent))
        .route("/ai/passive-suggestions", post(handle_passive_suggestions))
        .with_state(Arc::new(config))
}

/// Bind to `addr` and serve until the process is killed. Convenience helper for the binary.
pub async fn serve(addr: SocketAddr, config: ProxyConfig) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!("warp_proxy_server listening on http://{addr}");
    axum::serve(listener, router(config)).await?;
    Ok(())
}

async fn handle_multi_agent(
    State(_config): State<Arc<ProxyConfig>>,
    body: Bytes,
) -> Response {
    let request = match api::Request::decode(body.as_ref()) {
        Ok(r) => r,
        Err(e) => {
            log::warn!("failed to decode multi-agent request: {e}");
            return (StatusCode::BAD_REQUEST, format!("decode error: {e}")).into_response();
        }
    };

    log::debug!("received multi-agent request: input={:?}", request.input);

    let unimplemented = unimplemented_event(
        "open-warp-proxy: provider translation is not implemented yet (Phase 2.0 scaffold).",
    );
    sse_response(vec![unimplemented])
}

async fn handle_passive_suggestions(
    State(_config): State<Arc<ProxyConfig>>,
    body: Bytes,
) -> Response {
    if let Err(e) = api::Request::decode(body.as_ref()) {
        log::warn!("failed to decode passive-suggestions request: {e}");
        return (StatusCode::BAD_REQUEST, format!("decode error: {e}")).into_response();
    }
    let unimplemented = unimplemented_event(
        "open-warp-proxy: passive suggestions not implemented (Phase 2.0 scaffold).",
    );
    sse_response(vec![unimplemented])
}

fn unimplemented_event(message: &str) -> api::ResponseEvent {
    use api::response_event::stream_finished;

    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(
            api::response_event::StreamFinished {
                reason: Some(stream_finished::Reason::InternalError(
                    stream_finished::InternalError {
                        message: message.to_string(),
                    },
                )),
                ..Default::default()
            },
        )),
    }
}

fn sse_response(events: Vec<api::ResponseEvent>) -> Response {
    use axum::response::sse::{KeepAlive, Sse};
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE as BASE64_URL_SAFE;
    use futures_util::stream;

    let stream = stream::iter(events.into_iter().map(|event| {
        let mut buf = Vec::with_capacity(event.encoded_len());
        event.encode(&mut buf).expect("ResponseEvent::encode is infallible into Vec");
        let payload = format!("\"{}\"", BASE64_URL_SAFE.encode(buf));
        Ok::<Event, std::convert::Infallible>(Event::default().data(payload))
    }));

    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}
