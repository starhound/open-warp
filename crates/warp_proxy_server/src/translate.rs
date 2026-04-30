//! Translates between Warp's multi-agent proto and the provider-neutral chat format.
//!
//! `request_to_chat` walks the existing task history + new user input to assemble a
//! provider-ready chat with tool-use / tool-result content blocks. The streaming side
//! (in `lib.rs`) drives a `ChatStream` and emits Warp `ResponseEvent`s — this file owns
//! the proto-construction helpers used there.

use prost_types::FieldMask;
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::providers::{ChatMessage, ChatRequest, ChatRole, ContentBlock, ToolSpec};
use crate::tools;

/// Walk the request's task history + new input and build a chat-history list together with
/// metadata about the active conversation.
pub fn request_to_chat(req: &api::Request) -> Result<(ChatRequest, ConversationCtx), &'static str> {
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut active_task_id: Option<String> = None;

    if let Some(ctx) = req.task_context.as_ref() {
        for task in &ctx.tasks {
            active_task_id.get_or_insert_with(|| task.id.clone());
            for m in &task.messages {
                if let Some(payload) = m.message.as_ref() {
                    add_history_message(&mut messages, payload);
                }
            }
        }
    }

    let new_user_text = match req.input.as_ref().and_then(|i| i.r#type.as_ref()) {
        Some(api::request::input::Type::UserInputs(inputs)) => {
            let mut text = String::new();
            for ui in &inputs.inputs {
                let Some(input) = ui.input.as_ref() else {
                    continue;
                };
                match input {
                    api::request::input::user_inputs::user_input::Input::UserQuery(uq) => {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(&uq.query);
                    }
                    api::request::input::user_inputs::user_input::Input::ToolCallResult(r) => {
                        push_input_tool_result(&mut messages, r);
                    }
                    _ => {}
                }
            }
            text
        }
        #[allow(deprecated)]
        Some(api::request::input::Type::UserQuery(uq)) => uq.query.clone(),
        _ => String::new(),
    };

    if !new_user_text.is_empty() {
        push_user_text(&mut messages, new_user_text.clone());
    }

    if messages.is_empty() {
        return Err("empty chat — no user input and no history");
    }

    let supported_tools: Vec<i32> = req
        .settings
        .as_ref()
        .map(|s| s.supported_tools.clone())
        .unwrap_or_default();
    let tools: Vec<ToolSpec> = tools::build_tool_specs(&supported_tools);

    let chat = ChatRequest {
        system: Some(tools::system_prompt()),
        messages,
        tools,
        max_tokens: 8192,
    };

    let ctx = ConversationCtx {
        active_task_id: active_task_id.unwrap_or_default(),
        new_user_text,
    };

    Ok((chat, ctx))
}

fn add_history_message(messages: &mut Vec<ChatMessage>, payload: &api::message::Message) {
    use api::message::Message as M;
    match payload {
        M::UserQuery(uq) => push_user_text(messages, uq.query.clone()),
        M::AgentOutput(ao) => push_assistant_text(messages, ao.text.clone()),
        M::ToolCall(call) => push_tool_call(messages, call),
        M::ToolCallResult(r) => push_tool_result(messages, r),
        // Other message types (reasoning, todos, citations, server events, etc.) are
        // currently summarized as system-style notes via `push_assistant_text` if
        // they carry display text, otherwise dropped.
        M::AgentReasoning(r) if !r.reasoning.is_empty() => {
            push_assistant_text(messages, r.reasoning.clone());
        }
        _ => {}
    }
}

fn push_user_text(messages: &mut Vec<ChatMessage>, text: String) {
    if text.trim().is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut() {
        if last.role == ChatRole::User {
            last.content.push(ContentBlock::Text(text));
            return;
        }
    }
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: vec![ContentBlock::Text(text)],
    });
}

fn push_assistant_text(messages: &mut Vec<ChatMessage>, text: String) {
    if text.trim().is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut() {
        if last.role == ChatRole::Assistant {
            last.content.push(ContentBlock::Text(text));
            return;
        }
    }
    messages.push(ChatMessage {
        role: ChatRole::Assistant,
        content: vec![ContentBlock::Text(text)],
    });
}

fn push_tool_call(messages: &mut Vec<ChatMessage>, call: &api::message::ToolCall) {
    let Some(tool) = call.tool.as_ref() else {
        return;
    };
    let Some(name) = tools::tool_call_to_anthropic_name(tool) else {
        return; // Tool not yet wired into the proxy.
    };
    let block = ContentBlock::ToolUse {
        id: call.tool_call_id.clone(),
        name: name.to_string(),
        input: tools::tool_call_to_anthropic_input(tool),
    };
    if let Some(last) = messages.last_mut() {
        if last.role == ChatRole::Assistant {
            last.content.push(block);
            return;
        }
    }
    messages.push(ChatMessage {
        role: ChatRole::Assistant,
        content: vec![block],
    });
}

fn push_tool_result(messages: &mut Vec<ChatMessage>, result: &api::message::ToolCallResult) {
    let (text, is_error) = tools::render_message_result(result);
    push_tool_result_block(messages, result.tool_call_id.clone(), text, is_error);
}

fn push_input_tool_result(
    messages: &mut Vec<ChatMessage>,
    result: &api::request::input::ToolCallResult,
) {
    let (text, is_error) = tools::render_input_result(result);
    push_tool_result_block(messages, result.tool_call_id.clone(), text, is_error);
}

fn push_tool_result_block(
    messages: &mut Vec<ChatMessage>,
    tool_use_id: String,
    content: String,
    is_error: bool,
) {
    let block = ContentBlock::ToolResult {
        tool_use_id,
        content,
        is_error,
    };
    if let Some(last) = messages.last_mut() {
        if last.role == ChatRole::User {
            last.content.push(block);
            return;
        }
    }
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: vec![block],
    });
}

pub struct ConversationCtx {
    pub active_task_id: String,
    pub new_user_text: String,
}

pub struct StreamIds {
    pub conversation_id: String,
    pub request_id: String,
    pub run_id: String,
    pub task_id: String,
    pub user_message_id: String,
    pub assistant_message_id: String,
}

impl StreamIds {
    pub fn new_for(req: &api::Request, ctx: &ConversationCtx) -> Self {
        let conversation_id = req
            .metadata
            .as_ref()
            .map(|m| m.conversation_id.clone())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(new_uuid);
        let task_id = if ctx.active_task_id.is_empty() {
            new_uuid()
        } else {
            ctx.active_task_id.clone()
        };
        Self {
            conversation_id,
            request_id: new_uuid(),
            run_id: new_uuid(),
            task_id,
            user_message_id: new_uuid(),
            assistant_message_id: new_uuid(),
        }
    }
}

pub fn new_uuid() -> String {
    Uuid::new_v4().to_string()
}

/// Helpers to build [`api::ResponseEvent`]s used by the SSE handler.
pub mod build {
    use super::*;

    pub fn stream_init(ids: &StreamIds) -> api::ResponseEvent {
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::Init(
                api::response_event::StreamInit {
                    conversation_id: ids.conversation_id.clone(),
                    request_id: ids.request_id.clone(),
                    run_id: ids.run_id.clone(),
                    ..Default::default()
                },
            )),
        }
    }

    pub fn client_actions(actions: Vec<api::ClientAction>) -> api::ResponseEvent {
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::ClientActions(
                api::response_event::ClientActions { actions },
            )),
        }
    }

    pub fn finished_done() -> api::ResponseEvent {
        use api::response_event::stream_finished;
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::Finished(
                api::response_event::StreamFinished {
                    reason: Some(stream_finished::Reason::Done(stream_finished::Done {})),
                    ..Default::default()
                },
            )),
        }
    }

    pub fn finished_error(message: String) -> api::ResponseEvent {
        use api::response_event::stream_finished;
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::Finished(
                api::response_event::StreamFinished {
                    reason: Some(stream_finished::Reason::InternalError(
                        stream_finished::InternalError { message },
                    )),
                    ..Default::default()
                },
            )),
        }
    }

    pub fn finished_invalid_api_key(model: &str) -> api::ResponseEvent {
        use api::response_event::stream_finished;
        api::ResponseEvent {
            r#type: Some(api::response_event::Type::Finished(
                api::response_event::StreamFinished {
                    reason: Some(stream_finished::Reason::InvalidApiKey(
                        stream_finished::InvalidApiKey {
                            model_name: model.to_string(),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                },
            )),
        }
    }

    pub fn begin_transaction() -> api::ClientAction {
        api::ClientAction {
            action: Some(api::client_action::Action::BeginTransaction(
                api::client_action::BeginTransaction {},
            )),
        }
    }

    pub fn commit_transaction() -> api::ClientAction {
        api::ClientAction {
            action: Some(api::client_action::Action::CommitTransaction(
                api::client_action::CommitTransaction {},
            )),
        }
    }

    pub fn create_task(task_id: &str) -> api::ClientAction {
        api::ClientAction {
            action: Some(api::client_action::Action::CreateTask(
                api::client_action::CreateTask {
                    task: Some(api::Task {
                        id: task_id.to_string(),
                        ..Default::default()
                    }),
                },
            )),
        }
    }

    /// Emit the user's message + an empty assistant text message that subsequent text deltas
    /// will append into. Returns the `Message` struct ID we used for the assistant message
    /// (caller-derived from `StreamIds`).
    pub fn add_user_and_assistant(
        ids: &StreamIds,
        user_text: &str,
        request_id: &str,
    ) -> api::ClientAction {
        let mut messages: Vec<api::Message> = Vec::new();
        if !user_text.is_empty() {
            messages.push(api::Message {
                id: ids.user_message_id.clone(),
                task_id: ids.task_id.clone(),
                request_id: request_id.to_string(),
                timestamp: Some(now_timestamp()),
                message: Some(api::message::Message::UserQuery(api::message::UserQuery {
                    query: user_text.to_string(),
                    ..Default::default()
                })),
                ..Default::default()
            });
        }
        messages.push(api::Message {
            id: ids.assistant_message_id.clone(),
            task_id: ids.task_id.clone(),
            request_id: request_id.to_string(),
            timestamp: Some(now_timestamp()),
            message: Some(api::message::Message::AgentOutput(
                api::message::AgentOutput {
                    text: String::new(),
                },
            )),
            ..Default::default()
        });
        api::ClientAction {
            action: Some(api::client_action::Action::AddMessagesToTask(
                api::client_action::AddMessagesToTask {
                    task_id: ids.task_id.clone(),
                    messages,
                },
            )),
        }
    }

    /// Append text to the *most recent* assistant text message identified by `assistant_message_id`.
    /// The proxy may emit several assistant text messages (one per Anthropic text content block);
    /// each is created via [`add_assistant_text_message`] and then appended into.
    pub fn append_to_assistant(message_id: &str, task_id: &str, delta: String) -> api::ClientAction {
        let message = api::Message {
            id: message_id.to_string(),
            task_id: task_id.to_string(),
            message: Some(api::message::Message::AgentOutput(
                api::message::AgentOutput { text: delta },
            )),
            ..Default::default()
        };
        api::ClientAction {
            action: Some(api::client_action::Action::AppendToMessageContent(
                api::client_action::AppendToMessageContent {
                    task_id: task_id.to_string(),
                    message: Some(message),
                    mask: Some(FieldMask {
                        paths: vec!["agent_output.text".to_string()],
                    }),
                },
            )),
        }
    }

    /// Add a fresh empty assistant-text Message to the task — used when the model produces
    /// a text content block *after* a tool call (so the prior assistant message is already
    /// closed).
    pub fn add_assistant_text_message(
        message_id: &str,
        task_id: &str,
        request_id: &str,
    ) -> api::ClientAction {
        api::ClientAction {
            action: Some(api::client_action::Action::AddMessagesToTask(
                api::client_action::AddMessagesToTask {
                    task_id: task_id.to_string(),
                    messages: vec![api::Message {
                        id: message_id.to_string(),
                        task_id: task_id.to_string(),
                        request_id: request_id.to_string(),
                        timestamp: Some(now_timestamp()),
                        message: Some(api::message::Message::AgentOutput(
                            api::message::AgentOutput {
                                text: String::new(),
                            },
                        )),
                        ..Default::default()
                    }],
                },
            )),
        }
    }

    /// Add a tool-call Message into the task. The message ID + tool_call_id come from the
    /// caller (typically `tool_use.id` straight from the provider so the Warp client and the
    /// next-round tool result line up).
    pub fn add_tool_call_message(
        message_id: &str,
        task_id: &str,
        request_id: &str,
        tool_call: api::message::ToolCall,
    ) -> api::ClientAction {
        api::ClientAction {
            action: Some(api::client_action::Action::AddMessagesToTask(
                api::client_action::AddMessagesToTask {
                    task_id: task_id.to_string(),
                    messages: vec![api::Message {
                        id: message_id.to_string(),
                        task_id: task_id.to_string(),
                        request_id: request_id.to_string(),
                        timestamp: Some(now_timestamp()),
                        message: Some(api::message::Message::ToolCall(tool_call)),
                        ..Default::default()
                    }],
                },
            )),
        }
    }
}

fn now_timestamp() -> prost_types::Timestamp {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    prost_types::Timestamp {
        seconds: now.as_secs() as i64,
        nanos: now.subsec_nanos() as i32,
    }
}
