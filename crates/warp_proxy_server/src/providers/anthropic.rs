//! Anthropic Messages API client. Streams `messages.create` with `stream:true` and translates
//! the SSE event payloads into [`ChatEvent`]s. Supports tool use and tool results in addition
//! to plain text.
//!
//! Reference: <https://docs.anthropic.com/en/api/messages-streaming>.

use futures_util::StreamExt;
use reqwest_eventsource::{Event as SseEvent, EventSource};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ChatEvent, ChatRequest, ChatRole, ChatStream, ContentBlock};

const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: Vec<AnthropicContentBlock<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock<'a> {
    Text {
        text: &'a str,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: Value,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: &'a str,
        #[serde(skip_serializing_if = "is_false")]
        is_error: bool,
    },
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Serialize)]
struct AnthropicTool<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a Value,
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<AnthropicMessage<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool<'a>>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SsePayload {
    MessageStart {},
    Ping {},
    ContentBlockStart {
        index: u32,
        content_block: ContentBlockStart,
    },
    ContentBlockDelta {
        index: u32,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaInner,
    },
    MessageStop {},
    Error {
        error: AnthropicError,
    },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockStart {
    Text {
        #[serde(default)]
        #[allow(dead_code)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        // `input` arrives as `{}` initially and is then filled in via input_json_delta.
        #[serde(default)]
        #[allow(dead_code)]
        input: Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    /// Anthropic also emits `thinking_delta` for extended thinking. Phase 2.2 ignores it.
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

    // Build messages with content-block arrays. Anthropic accepts either a string or an
    // array; we always send arrays so tool_use / tool_result can mix with text.
    let messages: Vec<AnthropicMessage<'_>> = request
        .messages
        .iter()
        .map(|m| AnthropicMessage {
            role: match m.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            },
            content: m
                .content
                .iter()
                .map(|c| match c {
                    ContentBlock::Text(t) => AnthropicContentBlock::Text { text: t.as_str() },
                    ContentBlock::ToolUse { id, name, input } => AnthropicContentBlock::ToolUse {
                        id: id.as_str(),
                        name: name.as_str(),
                        input: input.clone(),
                    },
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => AnthropicContentBlock::ToolResult {
                        tool_use_id: tool_use_id.as_str(),
                        content: content.as_str(),
                        is_error: *is_error,
                    },
                })
                .collect(),
        })
        .collect();

    let tools: Vec<AnthropicTool<'_>> = request
        .tools
        .iter()
        .map(|t| AnthropicTool {
            name: t.name.as_str(),
            description: t.description.as_str(),
            input_schema: &t.input_schema,
        })
        .collect();

    let body = AnthropicRequest {
        model: model.as_str(),
        max_tokens: request.max_tokens,
        stream: true,
        system: request.system.as_deref(),
        messages,
        tools,
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
                        SsePayload::ContentBlockStart {
                            index,
                            content_block: ContentBlockStart::ToolUse { id, name, .. },
                        } => {
                            yield ChatEvent::ToolUseStart { index, id, name };
                        }
                        SsePayload::ContentBlockStart { .. } => {
                            // Text block start carries no useful payload until the first delta.
                        }
                        SsePayload::ContentBlockDelta {
                            index,
                            delta: ContentDelta::TextDelta { text },
                        } => {
                            yield ChatEvent::TextDelta { index, text };
                        }
                        SsePayload::ContentBlockDelta {
                            index,
                            delta: ContentDelta::InputJsonDelta { partial_json },
                        } => {
                            yield ChatEvent::ToolUseInputDelta { index, partial_json };
                        }
                        SsePayload::ContentBlockDelta {
                            delta: ContentDelta::Other, ..
                        } => {}
                        SsePayload::ContentBlockStop { index } => {
                            yield ChatEvent::BlockStop { index };
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
                        SsePayload::MessageStart {} | SsePayload::Ping {} => {}
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
