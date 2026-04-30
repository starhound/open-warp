//! Anthropic Messages API client. Streams `messages.create` with `stream:true` and translates
//! the SSE event payloads into [`ChatEvent`]s.
//!
//! Reference: <https://docs.anthropic.com/en/api/messages-streaming>.

use futures_util::StreamExt;
use reqwest_eventsource::{Event as SseEvent, EventSource};
use serde::{Deserialize, Serialize};

use super::{ChatEvent, ChatRequest, ChatRole, ChatStream};

const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<AnthropicMessage<'a>>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SsePayload {
    MessageStart {},
    Ping {},
    ContentBlockStart {},
    ContentBlockDelta { delta: ContentDelta },
    ContentBlockStop {},
    MessageDelta { delta: MessageDeltaInner },
    MessageStop {},
    Error { error: AnthropicError },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentDelta {
    TextDelta { text: String },
    /// Anthropic also emits `input_json_delta`, `thinking_delta`, etc. — accepted but ignored
    /// in Phase 2.1 (chat-only). Tools land in 2.2+.
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct MessageDeltaInner {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

pub fn chat_stream(
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    request: ChatRequest,
) -> ChatStream {
    use async_stream::stream;

    let messages: Vec<AnthropicMessage<'_>> = request
        .messages
        .iter()
        .map(|m| AnthropicMessage {
            role: match m.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            },
            content: m.text.as_str(),
        })
        .collect();

    let body = AnthropicRequest {
        model: model.as_str(),
        max_tokens: request.max_tokens,
        stream: true,
        system: request.system.as_deref(),
        messages,
    };

    let body_json = match serde_json::to_string(&body) {
        Ok(s) => s,
        Err(e) => {
            return Box::pin(async_stream::stream! {
                yield ChatEvent::Error(format!("failed to serialize anthropic request: {e}"));
            });
        }
    };

    let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
    let req_builder = client
        .post(url)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json")
        .body(body_json);

    let mut event_source = match EventSource::new(req_builder) {
        Ok(es) => es,
        Err(e) => {
            return Box::pin(async_stream::stream! {
                yield ChatEvent::Error(format!("failed to start anthropic SSE: {e}"));
            });
        }
    };

    Box::pin(stream! {
        let mut last_stop_reason: Option<String> = None;
        while let Some(event) = event_source.next().await {
            match event {
                Ok(SseEvent::Open) => continue,
                Ok(SseEvent::Message(msg)) => {
                    let payload: SsePayload = match serde_json::from_str(&msg.data) {
                        Ok(p) => p,
                        Err(e) => {
                            log::warn!("anthropic: failed to parse SSE payload {:?}: {e}", msg.data);
                            continue;
                        }
                    };
                    match payload {
                        SsePayload::ContentBlockDelta { delta: ContentDelta::TextDelta { text } } => {
                            yield ChatEvent::TextDelta(text);
                        }
                        SsePayload::MessageDelta { delta } => {
                            if delta.stop_reason.is_some() {
                                last_stop_reason = delta.stop_reason;
                            }
                        }
                        SsePayload::MessageStop {} => {
                            event_source.close();
                            yield ChatEvent::Done { stop_reason: last_stop_reason.take() };
                            return;
                        }
                        SsePayload::Error { error } => {
                            event_source.close();
                            yield ChatEvent::Error(format!("anthropic {}: {}", error.kind, error.message));
                            return;
                        }
                        SsePayload::MessageStart {}
                        | SsePayload::Ping {}
                        | SsePayload::ContentBlockStart {}
                        | SsePayload::ContentBlockStop {}
                        | SsePayload::ContentBlockDelta { delta: ContentDelta::Other } => {
                            // Phase 2.1 ignores these (they carry tool/thinking data we'll wire in 2.2+).
                        }
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) => {
                    yield ChatEvent::Done { stop_reason: last_stop_reason.take() };
                    return;
                }
                Err(e) => {
                    yield ChatEvent::Error(format!("anthropic SSE error: {e}"));
                    return;
                }
            }
        }
        yield ChatEvent::Done { stop_reason: last_stop_reason.take() };
    })
}
