//! Provider clients. Each provider exposes an async `chat_stream` returning a stream of
//! [`ChatEvent`]s — a provider-neutral wire format consumed by `translate::response`.

pub mod anthropic;

use futures_util::Stream;
use std::pin::Pin;

/// Provider-neutral streaming event. The translator turns each into a Warp `ResponseEvent`.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// A delta of assistant-visible text. `index` is the provider's content-block index so the
    /// translator can correlate deltas with the right text message when the model produces
    /// multiple text segments interleaved with tool calls.
    TextDelta {
        index: u32,
        text: String,
    },
    /// A new tool-use content block has started. Subsequent `ToolUseInputDelta`s with the same
    /// `index` belong to this call until the matching `ToolUseStop`.
    ToolUseStart {
        index: u32,
        id: String,
        name: String,
    },
    /// A fragment of the tool's input JSON. Concatenate fragments for the same `index` to
    /// reconstruct the full input value.
    ToolUseInputDelta {
        index: u32,
        partial_json: String,
    },
    /// The block at `index` is complete. The translator should now finalize whatever message
    /// it was building for that block.
    BlockStop {
        index: u32,
    },
    /// The model reported its stop reason; the stream is winding down.
    Done {
        stop_reason: Option<String>,
    },
    /// A non-recoverable error from the provider. The stream ends after this event.
    Error(String),
}

/// A boxed chat stream. The stream ends after a `Done` or `Error` event.
pub type ChatStream = Pin<Box<dyn Stream<Item = ChatEvent> + Send>>;

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub system: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolSpec>,
    pub max_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone)]
pub enum ContentBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}
