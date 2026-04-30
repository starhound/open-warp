//! `warp_proxy_server` — self-hosted proxy that speaks Warp's multi-agent API
//! over HTTP+SSE+protobuf, translating each request into a call to a user-configured
//! provider. Anthropic Messages API is fully wired (chat + tool use); OpenAI-compat and
//! Ollama are stubbed for follow-up phases.

pub mod config;
pub mod login;
pub mod providers;
pub mod tools;
pub mod translate;

use anyhow::Result;
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, sse::Event},
    routing::post,
};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE as BASE64_URL_SAFE;
use bytes::Bytes;
use futures_util::StreamExt;
use prost::Message;
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use warp_multi_agent_api as api;

pub use config::{AnthropicAuth, Provider, ProxyConfig};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ProxyConfig>,
    pub http: reqwest::Client,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/ai/multi-agent", post(handle_multi_agent))
        .route("/ai/passive-suggestions", post(handle_passive_suggestions))
        .with_state(state)
}

pub async fn serve(addr: SocketAddr, config: ProxyConfig) -> Result<()> {
    let state = AppState {
        config: Arc::new(config),
        http: reqwest::Client::new(),
    };
    let listener = tokio::net::TcpListener::bind(addr).await?;
    log::info!("warp_proxy_server listening on http://{addr}");
    axum::serve(listener, router(state)).await?;
    Ok(())
}

async fn handle_multi_agent(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match api::Request::decode(body.as_ref()) {
        Ok(r) => r,
        Err(e) => {
            log::warn!("failed to decode multi-agent request: {e}");
            return (StatusCode::BAD_REQUEST, format!("decode error: {e}")).into_response();
        }
    };

    let (chat, conv_ctx) = match translate::request_to_chat(&request) {
        Ok(pair) => pair,
        Err(reason) => {
            log::warn!("translate: {reason}");
            return sse_response(vec![translate::build::finished_error(reason.to_string())]);
        }
    };

    let ids = translate::StreamIds::new_for(&request, &conv_ctx);
    let in_app_keys = translate::RequestApiKeys::from_request(&request);

    let chat_stream = match &state.config.provider {
        Provider::Anthropic {
            auth,
            base_url,
            model,
        } => {
            // The in-app BYOK key (entered in open-warp Settings → AI) takes precedence over
            // the proxy's env-configured credential, so users can rotate keys without
            // restarting the proxy.
            let effective_auth = match &in_app_keys.anthropic {
                Some(key) => config::AnthropicAuth::ApiKey(key.clone()),
                None => auth.clone(),
            };
            providers::anthropic::chat_stream(
                state.http.clone(),
                base_url.clone(),
                effective_auth,
                model.clone(),
                chat,
            )
        }
        Provider::OpenAiCompat { .. } | Provider::Ollama { .. } => {
            return sse_response(vec![translate::build::finished_error(
                "provider not yet wired (Phase 2.3)".into(),
            )]);
        }
    };

    let response_stream = build_response_stream(ids, conv_ctx, chat_stream);
    sse_stream_response(response_stream)
}

async fn handle_passive_suggestions(State(_state): State<AppState>, body: Bytes) -> Response {
    if let Err(e) = api::Request::decode(body.as_ref()) {
        log::warn!("failed to decode passive-suggestions request: {e}");
        return (StatusCode::BAD_REQUEST, format!("decode error: {e}")).into_response();
    }
    sse_response(vec![translate::build::finished_done()])
}

/// State for tracking in-progress content blocks that arrive interleaved on the SSE stream.
/// The provider may emit multiple text blocks and/or tool_use blocks, identified by the
/// block index. Each tool_use block accumulates partial JSON until its `BlockStop`.
#[derive(Default)]
struct StreamState {
    /// `index` -> block kind state
    blocks: HashMap<u32, BlockState>,
    /// The most-recent assistant text-message ID we've created. Append-deltas target this.
    current_text_message_id: Option<String>,
}

enum BlockState {
    Text { message_id: String },
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
}

fn build_response_stream(
    ids: translate::StreamIds,
    conv_ctx: translate::ConversationCtx,
    mut chat_stream: providers::ChatStream,
) -> impl futures_util::Stream<Item = api::ResponseEvent> + Send {
    use async_stream::stream;
    use providers::ChatEvent;

    stream! {
        yield translate::build::stream_init(&ids);

        // Setup: open transaction, create task if needed, add the user message + an empty
        // assistant text message that the first text-delta block will append into.
        let mut setup_actions: Vec<api::ClientAction> =
            vec![translate::build::begin_transaction()];
        if conv_ctx.active_task_id.is_empty() {
            setup_actions.push(translate::build::create_task(&ids.task_id));
        }
        setup_actions.push(translate::build::add_user_and_assistant(
            &ids,
            &conv_ctx.new_user_text,
            &ids.request_id,
        ));
        yield translate::build::client_actions(setup_actions);

        let mut state = StreamState::default();
        // The initial assistant text message has the assistant_message_id from setup.
        state.current_text_message_id = Some(ids.assistant_message_id.clone());

        let mut errored: Option<String> = None;

        while let Some(event) = chat_stream.next().await {
            match event {
                ChatEvent::TextDelta { index, text } => {
                    if text.is_empty() { continue; }
                    let message_id = match state.blocks.get(&index) {
                        Some(BlockState::Text { message_id }) => message_id.clone(),
                        _ => {
                            // First time we're seeing this index: it's either the initial
                            // assistant message (already created in setup) or a new text
                            // segment after a tool call.
                            let need_new_message = !matches!(
                                state.current_text_message_id.as_deref(),
                                Some(id) if id == ids.assistant_message_id && state.blocks.is_empty()
                            );
                            let message_id = if need_new_message
                                && state.current_text_message_id.is_some()
                                && state
                                    .blocks
                                    .values()
                                    .any(|b| matches!(b, BlockState::ToolUse { .. }))
                            {
                                let id = translate::new_uuid();
                                yield translate::build::client_actions(vec![
                                    translate::build::add_assistant_text_message(
                                        &id, &ids.task_id, &ids.request_id,
                                    ),
                                ]);
                                id
                            } else {
                                state
                                    .current_text_message_id
                                    .clone()
                                    .unwrap_or_else(|| ids.assistant_message_id.clone())
                            };
                            state
                                .blocks
                                .insert(index, BlockState::Text { message_id: message_id.clone() });
                            state.current_text_message_id = Some(message_id.clone());
                            message_id
                        }
                    };
                    yield translate::build::client_actions(vec![
                        translate::build::append_to_assistant(&message_id, &ids.task_id, text),
                    ]);
                }
                ChatEvent::ToolUseStart { index, id, name } => {
                    state.blocks.insert(
                        index,
                        BlockState::ToolUse {
                            id,
                            name,
                            input_json: String::new(),
                        },
                    );
                }
                ChatEvent::ToolUseInputDelta { index, partial_json } => {
                    if let Some(BlockState::ToolUse { input_json, .. }) =
                        state.blocks.get_mut(&index)
                    {
                        input_json.push_str(&partial_json);
                    }
                }
                ChatEvent::BlockStop { index } => {
                    if let Some(block) = state.blocks.remove(&index) {
                        match block {
                            BlockState::Text { .. } => {
                                // Nothing to do — text was streamed via deltas.
                            }
                            BlockState::ToolUse {
                                id,
                                name,
                                input_json,
                            } => {
                                let input_value: serde_json::Value =
                                    if input_json.trim().is_empty() {
                                        serde_json::Value::Object(Default::default())
                                    } else {
                                        match serde_json::from_str(&input_json) {
                                            Ok(v) => v,
                                            Err(e) => {
                                                log::warn!(
                                                    "failed to parse tool input JSON for {name}: {e}; raw: {input_json}"
                                                );
                                                serde_json::Value::Object(Default::default())
                                            }
                                        }
                                    };
                                match tools::anthropic_tool_use_to_warp(&name, id.clone(), input_value) {
                                    Ok(call) => {
                                        // Use the provider's tool_use id as the Warp Message ID
                                        // too, so the next-round ToolCallResult.tool_call_id maps
                                        // back to this one.
                                        yield translate::build::client_actions(vec![
                                            translate::build::add_tool_call_message(
                                                &id,
                                                &ids.task_id,
                                                &ids.request_id,
                                                call,
                                            ),
                                        ]);
                                    }
                                    Err(e) => {
                                        log::warn!(
                                            "failed to translate tool_use {name}: {e}"
                                        );
                                        errored = Some(format!(
                                            "open-warp-proxy: tool {name} not supported"
                                        ));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                ChatEvent::Done { .. } => break,
                ChatEvent::Error(msg) => {
                    errored = Some(msg);
                    break;
                }
            }
        }

        yield translate::build::client_actions(vec![translate::build::commit_transaction()]);

        match errored {
            None => yield translate::build::finished_done(),
            Some(msg) => {
                let lower = msg.to_ascii_lowercase();
                if lower.contains("invalid_api_key")
                    || lower.contains("authentication_error")
                    || lower.contains("401")
                {
                    yield translate::build::finished_invalid_api_key("anthropic");
                } else {
                    yield translate::build::finished_error(msg);
                }
            }
        }
    }
}

/// Convert a stream of `ResponseEvent`s into an SSE response, encoding each as a base64
/// payload (matching what the Warp client decodes via `BASE64_URL_SAFE.decode`).
fn sse_stream_response(
    stream: impl futures_util::Stream<Item = api::ResponseEvent> + Send + 'static,
) -> Response {
    use axum::response::sse::{KeepAlive, Sse};

    let sse = stream.map(|event| {
        let mut buf = Vec::with_capacity(event.encoded_len());
        event
            .encode(&mut buf)
            .expect("ResponseEvent::encode is infallible into Vec");
        let payload = format!("\"{}\"", BASE64_URL_SAFE.encode(buf));
        Ok::<Event, std::convert::Infallible>(Event::default().data(payload))
    });

    Sse::new(sse).keep_alive(KeepAlive::default()).into_response()
}

/// Convenience for emitting a small bounded list of events as SSE — used for error-only
/// responses where there's no provider stream to consume.
fn sse_response(events: Vec<api::ResponseEvent>) -> Response {
    use futures_util::stream;
    sse_stream_response(stream::iter(events))
}
