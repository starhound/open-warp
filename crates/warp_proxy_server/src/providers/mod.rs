//! Provider clients. Each provider exposes an async `chat_stream` returning a stream of
//! [`ChatEvent`]s — a provider-neutral wire format consumed by `translate::response`.

pub mod anthropic;

use futures_util::Stream;
use std::pin::Pin;

/// Provider-neutral streaming event. The translator turns each into a Warp `ResponseEvent`.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// A delta of assistant-visible text.
    TextDelta(String),
    /// The model reported its stop reason; the stream is winding down.
    Done { stop_reason: Option<String> },
    /// A non-recoverable error from the provider. The stream ends after this event.
    Error(String),
}

/// A boxed chat stream. The stream ends after a `Done` or `Error` event.
pub type ChatStream = Pin<Box<dyn Stream<Item = ChatEvent> + Send>>;

/// A provider-agnostic chat request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub system: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}
