//! Translates between Warp's multi-agent proto and the provider-neutral [`ChatRequest`].

use prost_types::FieldMask;
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::providers::{ChatMessage, ChatRequest, ChatRole};

/// Walk the request's task history + new input and build a chat-history list. Phase 2.1 only
/// extracts plain text — tool calls, todos, reasoning, etc. are left for later phases.
pub fn request_to_chat(req: &api::Request) -> Result<(ChatRequest, ConversationCtx), &'static str> {
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut active_task_id: Option<String> = None;

    if let Some(ctx) = req.task_context.as_ref() {
        for task in &ctx.tasks {
            active_task_id.get_or_insert_with(|| task.id.clone());
            for m in &task.messages {
                if let Some(payload) = m.message.as_ref() {
                    if let Some((role, text)) = message_to_text(payload) {
                        if !text.trim().is_empty() {
                            messages.push(ChatMessage { role, text });
                        }
                    }
                }
            }
        }
    }

    let new_user_text = match req.input.as_ref().and_then(|i| i.r#type.as_ref()) {
        Some(api::request::input::Type::UserInputs(inputs)) => {
            let mut text = String::new();
            for ui in &inputs.inputs {
                if let Some(api::request::input::user_inputs::user_input::Input::UserQuery(uq)) =
                    ui.input.as_ref()
                {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(&uq.query);
                }
            }
            text
        }
        #[allow(deprecated)]
        Some(api::request::input::Type::UserQuery(uq)) => uq.query.clone(),
        _ => String::new(),
    };

    if new_user_text.trim().is_empty() && messages.is_empty() {
        return Err("no user text in request");
    }

    if !new_user_text.is_empty() {
        messages.push(ChatMessage {
            role: ChatRole::User,
            text: new_user_text.clone(),
        });
    }

    let chat = ChatRequest {
        system: None,
        messages,
        max_tokens: 4096,
    };

    let ctx = ConversationCtx {
        active_task_id: active_task_id.unwrap_or_default(),
        new_user_text,
    };

    Ok((chat, ctx))
}

pub struct ConversationCtx {
    pub active_task_id: String,
    pub new_user_text: String,
}

fn message_to_text(payload: &api::message::Message) -> Option<(ChatRole, String)> {
    use api::message::Message as M;
    match payload {
        M::UserQuery(uq) => Some((ChatRole::User, uq.query.clone())),
        M::AgentOutput(ao) => Some((ChatRole::Assistant, ao.text.clone())),
        // Phase 2.2+ will surface tool calls/results.
        _ => None,
    }
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

fn new_uuid() -> String {
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

    pub fn add_user_and_assistant(
        ids: &StreamIds,
        user_text: &str,
        request_id: &str,
    ) -> api::ClientAction {
        let user_msg = api::Message {
            id: ids.user_message_id.clone(),
            task_id: ids.task_id.clone(),
            request_id: request_id.to_string(),
            timestamp: Some(now_timestamp()),
            message: Some(api::message::Message::UserQuery(
                api::message::UserQuery {
                    query: user_text.to_string(),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        let assistant_msg = api::Message {
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
        };
        let messages = if user_text.is_empty() {
            vec![assistant_msg]
        } else {
            vec![user_msg, assistant_msg]
        };
        api::ClientAction {
            action: Some(api::client_action::Action::AddMessagesToTask(
                api::client_action::AddMessagesToTask {
                    task_id: ids.task_id.clone(),
                    messages,
                },
            )),
        }
    }

    pub fn append_to_assistant(ids: &StreamIds, delta: String) -> api::ClientAction {
        let message = api::Message {
            id: ids.assistant_message_id.clone(),
            task_id: ids.task_id.clone(),
            message: Some(api::message::Message::AgentOutput(
                api::message::AgentOutput { text: delta },
            )),
            ..Default::default()
        };
        api::ClientAction {
            action: Some(api::client_action::Action::AppendToMessageContent(
                api::client_action::AppendToMessageContent {
                    task_id: ids.task_id.clone(),
                    message: Some(message),
                    mask: Some(FieldMask {
                        paths: vec!["agent_output.text".to_string()],
                    }),
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
