//! Full Warp ↔ provider tool translation.
//!
//! Three responsibilities per tool:
//!
//! 1. [`tool_spec`] returns the JSON Schema we hand to Anthropic so the model knows what
//!    arguments the tool takes.
//! 2. [`anthropic_tool_use_to_warp`] turns the model's `tool_use` JSON into a Warp
//!    `Message::ToolCall` proto so the client can execute it.
//! 3. [`render_result`] formats a Warp `ToolCallResult` proto as text the model sees back
//!    on the next turn (paired with [`tool_call_to_anthropic_name`] +
//!    [`tool_call_to_anthropic_input`] which reconstruct the model-facing `tool_use` block
//!    for replay).
//!
//! Every variant of `ToolType` is covered. Tools that have no client-side input
//! (`open_code_review`, `init_project`, `suggest_create_plan`) accept an empty object;
//! tools whose result Warp's client doesn't normally surface as text (the suggestion-style
//! tools, `init_project`, `open_code_review`) get terse acknowledgements so the model can
//! continue. Computer-use action arrays are passed through as JSON values to keep the
//! schema flexible.

use anyhow::{Result, anyhow};
use prost_types::Struct as ProtoStruct;
use serde::Deserialize;
use serde_json::{Value, json};
use warp_multi_agent_api as api;

use crate::providers::ToolSpec;

pub fn build_tool_specs(supported: &[i32]) -> Vec<ToolSpec> {
    supported
        .iter()
        .filter_map(|id| api::ToolType::try_from(*id).ok())
        .filter_map(tool_spec)
        .collect()
}

fn obj(properties: Value, required: &[&str]) -> Value {
    json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

fn empty_obj() -> Value {
    json!({"type": "object", "properties": {}})
}

fn tool_spec(t: api::ToolType) -> Option<ToolSpec> {
    use api::ToolType::*;
    let (name, description, schema) = match t {
        ReadFiles => (
            "read_files",
            "Read the contents of one or more text files in the workspace.",
            obj(
                json!({
                    "files": {
                        "type": "array", "minItems": 1,
                        "items": obj(
                            json!({
                                "name": {"type": "string"},
                                "line_ranges": {
                                    "type": "array",
                                    "items": obj(
                                        json!({"start": {"type":"integer","minimum":1}, "end": {"type":"integer","minimum":1}}),
                                        &["start", "end"],
                                    )
                                }
                            }),
                            &["name"],
                        )
                    }
                }),
                &["files"],
            ),
        ),
        ApplyFileDiffs => (
            "apply_file_diffs",
            "Edit, create, or delete files. Use this — never write files via shell.",
            obj(
                json!({
                    "summary": {"type": "string"},
                    "diffs": {
                        "type": "array",
                        "items": obj(
                            json!({
                                "file_path": {"type":"string"},
                                "search": {"type":"string"},
                                "replace": {"type":"string"},
                            }),
                            &["file_path", "search", "replace"],
                        ),
                    },
                    "new_files": {
                        "type": "array",
                        "items": obj(
                            json!({"file_path": {"type":"string"}, "content": {"type":"string"}}),
                            &["file_path", "content"],
                        ),
                    },
                    "deleted_files": {
                        "type": "array",
                        "items": obj(
                            json!({"file_path": {"type":"string"}}),
                            &["file_path"],
                        ),
                    },
                }),
                &["summary"],
            ),
        ),
        RunShellCommand => (
            "run_shell_command",
            "Run a shell command. Mark `is_read_only` for inspection commands.",
            obj(
                json!({
                    "command": {"type": "string"},
                    "is_read_only": {"type": "boolean"},
                }),
                &["command"],
            ),
        ),
        Grep => (
            "grep",
            "Regex search across file contents.",
            obj(
                json!({
                    "queries": {"type":"array", "minItems":1, "items":{"type":"string"}},
                    "path": {"type":"string"},
                }),
                &["queries"],
            ),
        ),
        FileGlobV2 => (
            "file_glob",
            "Find files matching glob patterns.",
            obj(
                json!({
                    "patterns": {"type":"array", "minItems":1, "items":{"type":"string"}},
                    "search_dir": {"type":"string"},
                    "max_matches": {"type":"integer"},
                    "max_depth": {"type":"integer"},
                    "min_depth": {"type":"integer"},
                }),
                &["patterns"],
            ),
        ),
        FileGlob => (
            "file_glob_legacy",
            "Legacy file glob (prefer `file_glob`).",
            obj(
                json!({
                    "patterns": {"type":"array", "items":{"type":"string"}},
                    "path": {"type":"string"},
                }),
                &["patterns"],
            ),
        ),
        SearchCodebase => (
            "search_codebase",
            "Semantic codebase search. If your provider can't actually do semantic \
             search, prefer `grep` + `read_files`.",
            obj(
                json!({
                    "query": {"type":"string"},
                    "path_filters": {"type":"array", "items":{"type":"string"}},
                    "codebase_path": {"type":"string"},
                }),
                &["query"],
            ),
        ),
        SuggestPlan => (
            "suggest_plan",
            "Propose a multi-step plan for a complex task.",
            obj(
                json!({
                    "summary": {"type":"string"},
                    "proposed_tasks": {
                        "type":"array",
                        "items": obj(
                            json!({"id": {"type":"string"}, "description": {"type":"string"}}),
                            &["description"],
                        ),
                    },
                }),
                &["summary"],
            ),
        ),
        SuggestCreatePlan => (
            "suggest_create_plan",
            "Suggest entering plan mode (autoplanning).",
            empty_obj(),
        ),
        SuggestNewConversation => (
            "suggest_new_conversation",
            "Suggest branching the conversation at the current point.",
            obj(json!({"message_id": {"type":"string"}}), &[]),
        ),
        SuggestPrompt => (
            "suggest_prompt",
            "Surface a follow-up prompt the user can run.",
            obj(
                json!({
                    "prompt": {"type":"string"},
                    "label": {"type":"string"},
                }),
                &["prompt"],
            ),
        ),
        AskUserQuestion => (
            "ask_user_question",
            "Ask the user multiple-choice questions.",
            obj(
                json!({
                    "questions": {
                        "type":"array", "minItems":1,
                        "items": obj(
                            json!({
                                "question_id": {"type":"string"},
                                "question": {"type":"string"},
                                "options": {"type":"array", "items":{"type":"string"}},
                                "recommended_option_index": {"type":"integer"},
                                "is_multiselect": {"type":"boolean"},
                                "supports_other": {"type":"boolean"},
                            }),
                            &["question", "options"],
                        ),
                    }
                }),
                &["questions"],
            ),
        ),
        TransferShellCommandControlToUser => (
            "transfer_shell_command_control_to_user",
            "Hand control of a foreground command back to the user.",
            obj(json!({"reason": {"type":"string"}}), &["reason"]),
        ),
        OpenCodeReview => (
            "open_code_review",
            "Open the code review pane in the client.",
            empty_obj(),
        ),
        InitProject => (
            "init_project",
            "Trigger the project initialization flow.",
            empty_obj(),
        ),
        ReadSkill => (
            "read_skill",
            "Read a skill (SKILL.md) by path or bundled id.",
            obj(
                json!({
                    "skill_path": {"type":"string"},
                    "bundled_skill_id": {"type":"string"},
                    "name": {"type":"string"},
                }),
                &[],
            ),
        ),
        UploadFileArtifact => (
            "upload_file_artifact",
            "Upload a local file as a conversation artifact.",
            obj(
                json!({
                    "file_path": {"type":"string"},
                    "description": {"type":"string"},
                }),
                &["file_path"],
            ),
        ),
        ReadMcpResource => (
            "read_mcp_resource",
            "Read an MCP resource by URI.",
            obj(
                json!({"uri": {"type":"string"}, "server_id": {"type":"string"}}),
                &["uri"],
            ),
        ),
        CallMcpTool => (
            "call_mcp_tool",
            "Invoke an MCP tool with named JSON args.",
            obj(
                json!({
                    "name": {"type":"string"},
                    "args": {"type":"object"},
                    "server_id": {"type":"string"},
                }),
                &["name"],
            ),
        ),
        WriteToLongRunningShellCommand => (
            "write_to_long_running_shell_command",
            "Write input to a long-running foreground shell command.",
            obj(
                json!({
                    "command_id": {"type":"string"},
                    "input": {"type":"string"},
                    "mode": {"type":"string", "enum":["raw","line","block"]},
                }),
                &["command_id", "input"],
            ),
        ),
        ReadShellCommandOutput => (
            "read_shell_command_output",
            "Read the latest output of a long-running shell command.",
            obj(
                json!({
                    "command_id": {"type":"string"},
                    "delay_seconds": {"type":"integer", "description": "If set, wait this many seconds before returning."},
                    "wait_for_completion": {"type":"boolean", "description": "If true, return when the command completes."},
                }),
                &["command_id"],
            ),
        ),
        ReadDocuments => (
            "read_documents",
            "Read one or more Drive documents.",
            obj(
                json!({
                    "documents": {
                        "type":"array",
                        "items": obj(
                            json!({
                                "document_id": {"type":"string"},
                                "line_ranges": {"type":"array","items": obj(
                                    json!({"start":{"type":"integer"},"end":{"type":"integer"}}),
                                    &["start","end"],
                                )}
                            }),
                            &["document_id"],
                        ),
                    }
                }),
                &["documents"],
            ),
        ),
        EditDocuments => (
            "edit_documents",
            "Edit existing Drive documents via search/replace.",
            obj(
                json!({
                    "diffs": {
                        "type":"array",
                        "items": obj(
                            json!({
                                "document_id": {"type":"string"},
                                "search": {"type":"string"},
                                "replace": {"type":"string"},
                            }),
                            &["document_id", "search", "replace"],
                        ),
                    }
                }),
                &["diffs"],
            ),
        ),
        CreateDocuments => (
            "create_documents",
            "Create new Drive documents.",
            obj(
                json!({
                    "new_documents": {
                        "type":"array",
                        "items": obj(
                            json!({"title":{"type":"string"},"content":{"type":"string"}}),
                            &["title","content"],
                        ),
                    }
                }),
                &["new_documents"],
            ),
        ),
        InsertReviewComments => (
            "insert_review_comments",
            "Insert review comments into a repo.",
            obj(
                json!({
                    "repo_path": {"type":"string"},
                    "base_branch": {"type":"string"},
                    "comments": {"type":"array"},
                }),
                &["repo_path"],
            ),
        ),
        UseComputer => (
            "use_computer",
            "Drive the user's computer (mouse, keyboard, screenshots).",
            obj(
                json!({
                    "action_summary": {"type":"string"},
                    "actions": {
                        "type":"array",
                        "description": "Array of action objects, each having a `type` field (mouse_move, mouse_down, mouse_up, mouse_wheel, wait, type_text, key_down, key_up).",
                        "items": {"type":"object"},
                    },
                    "post_actions_screenshot": {"type":"object"},
                }),
                &["action_summary", "actions"],
            ),
        ),
        RequestComputerUse => (
            "request_computer_use",
            "Ask the user to grant computer-use permission.",
            obj(
                json!({"task_summary": {"type":"string"}}),
                &["task_summary"],
            ),
        ),
        Subagent => (
            "subagent",
            "Spawn a sub-agent for a focused task.",
            obj(
                json!({
                    "task_id": {"type":"string"},
                    "payload": {"type":"string"},
                    "kind": {"type":"string", "enum":["cli","research","advice","computer_use","summarization","conversation_search","warp_documentation_search"]},
                    "command_id": {"type":"string", "description": "For kind=cli only."},
                    "query": {"type":"string", "description": "For kind=conversation_search only."},
                    "conversation_id": {"type":"string", "description": "For kind=conversation_search only."},
                }),
                &["payload", "kind"],
            ),
        ),
        StartAgent => (
            "start_agent",
            "Start a child agent in a separate conversation.",
            obj(
                json!({
                    "name": {"type":"string"},
                    "prompt": {"type":"string"},
                    "remote_environment_id": {"type":"string"},
                }),
                &["name", "prompt"],
            ),
        ),
        StartAgentV2 => (
            "start_agent_v2",
            "Start a child agent in a separate run (v2).",
            obj(
                json!({
                    "name": {"type":"string"},
                    "prompt": {"type":"string"},
                    "execution": {
                        "type":"string",
                        "enum":["local","remote"],
                    },
                    "harness": {"type":"string"},
                    "model_id": {"type":"string"},
                    "remote_environment_id": {"type":"string"},
                    "title": {"type":"string"},
                }),
                &["name", "prompt"],
            ),
        ),
        SendMessageToAgent => (
            "send_message_to_agent",
            "Send a message to other running agent(s).",
            obj(
                json!({
                    "addresses": {"type":"array", "items": {"type":"string"}},
                    "subject": {"type":"string"},
                    "message": {"type":"string"},
                }),
                &["addresses", "message"],
            ),
        ),
        FetchConversation => (
            "fetch_conversation",
            "Fetch another conversation's tasks.",
            obj(
                json!({"conversation_id": {"type":"string"}}),
                &["conversation_id"],
            ),
        ),
    };
    Some(ToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: schema,
    })
}

pub fn anthropic_tool_use_to_warp(
    name: &str,
    tool_call_id: String,
    input: Value,
) -> Result<api::message::ToolCall> {
    use api::message::tool_call as tc;

    let tool = match name {
        "read_files" => parse_read_files(input)?,
        "apply_file_diffs" => parse_apply_file_diffs(input)?,
        "run_shell_command" => parse_run_shell_command(input)?,
        "grep" => parse_grep(input)?,
        "file_glob" => parse_file_glob_v2(input)?,
        "file_glob_legacy" => parse_file_glob(input)?,
        "search_codebase" => parse_search_codebase(input)?,
        "suggest_plan" => parse_suggest_plan(input)?,
        "suggest_create_plan" => tc::Tool::SuggestCreatePlan(tc::SuggestCreatePlan {}),
        "suggest_new_conversation" => parse_suggest_new_conversation(input)?,
        "suggest_prompt" => parse_suggest_prompt(input)?,
        "ask_user_question" => parse_ask_user_question(input)?,
        "transfer_shell_command_control_to_user" => {
            parse_transfer_shell_command_control(input)?
        }
        "open_code_review" => tc::Tool::OpenCodeReview(tc::OpenCodeReview {}),
        "init_project" => tc::Tool::InitProject(tc::InitProject {}),
        "read_skill" => parse_read_skill(input)?,
        "upload_file_artifact" => parse_upload_file_artifact(input)?,
        "read_mcp_resource" => parse_read_mcp_resource(input)?,
        "call_mcp_tool" => parse_call_mcp_tool(input)?,
        "write_to_long_running_shell_command" => parse_write_to_long_running(input)?,
        "read_shell_command_output" => parse_read_shell_command_output(input)?,
        "read_documents" => parse_read_documents(input)?,
        "edit_documents" => parse_edit_documents(input)?,
        "create_documents" => parse_create_documents(input)?,
        "insert_review_comments" => parse_insert_review_comments(input)?,
        "use_computer" => parse_use_computer(input)?,
        "request_computer_use" => parse_request_computer_use(input)?,
        "subagent" => parse_subagent(input)?,
        "start_agent" => parse_start_agent(input)?,
        "start_agent_v2" => parse_start_agent_v2(input)?,
        "send_message_to_agent" => parse_send_message_to_agent(input)?,
        "fetch_conversation" => parse_fetch_conversation(input)?,
        other => return Err(anyhow!("unknown tool: {other}")),
    };

    Ok(api::message::ToolCall {
        tool_call_id,
        tool: Some(tool),
    })
}

// ───────────────────────── Parsers ─────────────────────────

fn parse_read_files(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { files: Vec<FileSpec> }
    #[derive(Deserialize)]
    struct FileSpec {
        name: String,
        #[serde(default)]
        line_ranges: Vec<LineRange>,
    }
    #[derive(Deserialize)]
    struct LineRange { start: u32, end: u32 }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::ReadFiles(tc::ReadFiles {
        files: args
            .files
            .into_iter()
            .map(|f| tc::read_files::File {
                name: f.name,
                line_ranges: f
                    .line_ranges
                    .into_iter()
                    .map(|r| api::FileContentLineRange { start: r.start, end: r.end })
                    .collect(),
            })
            .collect(),
    }))
}

fn parse_apply_file_diffs(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize, Default)]
    struct Args {
        #[serde(default)] summary: String,
        #[serde(default)] diffs: Vec<DiffSpec>,
        #[serde(default)] new_files: Vec<NewFileSpec>,
        #[serde(default)] deleted_files: Vec<DeletedFileSpec>,
    }
    #[derive(Deserialize)]
    struct DiffSpec { file_path: String, search: String, replace: String }
    #[derive(Deserialize)]
    struct NewFileSpec { file_path: String, #[serde(default)] content: String }
    #[derive(Deserialize)]
    struct DeletedFileSpec { file_path: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::ApplyFileDiffs(tc::ApplyFileDiffs {
        summary: args.summary,
        diffs: args.diffs.into_iter().map(|d| tc::apply_file_diffs::FileDiff {
            file_path: d.file_path, search: d.search, replace: d.replace,
        }).collect(),
        new_files: args.new_files.into_iter().map(|f| tc::apply_file_diffs::NewFile {
            file_path: f.file_path, content: f.content,
        }).collect(),
        deleted_files: args.deleted_files.into_iter().map(|f| tc::apply_file_diffs::DeleteFile {
            file_path: f.file_path,
        }).collect(),
        ..Default::default()
    }))
}

fn parse_run_shell_command(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { command: String, #[serde(default)] is_read_only: bool }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::RunShellCommand(tc::RunShellCommand {
        command: args.command,
        #[allow(deprecated)] is_read_only: args.is_read_only,
        ..Default::default()
    }))
}

fn parse_grep(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { queries: Vec<String>, #[serde(default)] path: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::Grep(tc::Grep { queries: args.queries, path: args.path }))
}

fn parse_file_glob_v2(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        patterns: Vec<String>,
        #[serde(default)] search_dir: String,
        #[serde(default)] max_matches: i32,
        #[serde(default)] max_depth: i32,
        #[serde(default)] min_depth: i32,
    }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::FileGlobV2(tc::FileGlobV2 {
        patterns: args.patterns,
        search_dir: args.search_dir,
        max_matches: args.max_matches,
        max_depth: args.max_depth,
        min_depth: args.min_depth,
    }))
}

#[allow(deprecated)]
fn parse_file_glob(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { patterns: Vec<String>, #[serde(default)] path: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::FileGlob(tc::FileGlob { patterns: args.patterns, path: args.path }))
}

fn parse_search_codebase(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        query: String,
        #[serde(default)] path_filters: Vec<String>,
        #[serde(default)] codebase_path: String,
    }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::SearchCodebase(tc::SearchCodebase {
        query: args.query,
        path_filters: args.path_filters,
        codebase_path: args.codebase_path,
    }))
}

fn parse_suggest_plan(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        #[serde(default)] summary: String,
        #[serde(default)] proposed_tasks: Vec<TaskSpec>,
    }
    #[derive(Deserialize)]
    struct TaskSpec {
        #[serde(default)] id: String,
        description: String,
    }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::SuggestPlan(tc::SuggestPlan {
        summary: args.summary,
        proposed_tasks: args.proposed_tasks.into_iter().map(|t| api::Task {
            id: if t.id.is_empty() { uuid::Uuid::new_v4().to_string() } else { t.id },
            description: t.description,
            ..Default::default()
        }).collect(),
    }))
}

fn parse_suggest_new_conversation(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize, Default)]
    struct Args { #[serde(default)] message_id: String }
    let args: Args = serde_json::from_value(input).unwrap_or_default();
    Ok(tc::Tool::SuggestNewConversation(tc::SuggestNewConversation {
        message_id: args.message_id,
    }))
}

fn parse_suggest_prompt(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { prompt: String, #[serde(default)] label: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::SuggestPrompt(tc::SuggestPrompt {
        display_mode: Some(tc::suggest_prompt::DisplayMode::PromptChip(
            tc::suggest_prompt::PromptChip { prompt: args.prompt, label: args.label },
        )),
        ..Default::default()
    }))
}

fn parse_ask_user_question(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { questions: Vec<QuestionSpec> }
    #[derive(Deserialize)]
    struct QuestionSpec {
        #[serde(default)] question_id: String,
        question: String,
        #[serde(default)] options: Vec<String>,
        #[serde(default)] recommended_option_index: i32,
        #[serde(default)] is_multiselect: bool,
        #[serde(default)] supports_other: bool,
    }
    let args: Args = serde_json::from_value(input)?;
    let questions = args.questions.into_iter().map(|q| {
        let id = if q.question_id.is_empty() { uuid::Uuid::new_v4().to_string() } else { q.question_id };
        api::ask_user_question::Question {
            question_id: id,
            question: q.question,
            question_type: Some(api::ask_user_question::question::QuestionType::MultipleChoice(
                api::ask_user_question::MultipleChoice {
                    options: q.options.into_iter().map(|o| api::ask_user_question::Option { label: o }).collect(),
                    recommended_option_index: q.recommended_option_index,
                    is_multiselect: q.is_multiselect,
                    supports_other: q.supports_other,
                },
            )),
        }
    }).collect();
    Ok(tc::Tool::AskUserQuestion(api::AskUserQuestion { questions }))
}

fn parse_transfer_shell_command_control(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { reason: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::TransferShellCommandControlToUser(
        tc::TransferShellCommandControlToUser { reason: args.reason },
    ))
}

fn parse_read_skill(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize, Default)]
    struct Args {
        #[serde(default)] skill_path: String,
        #[serde(default)] bundled_skill_id: String,
        #[serde(default)] name: String,
    }
    let args: Args = serde_json::from_value(input).unwrap_or_default();
    let reference = if !args.skill_path.is_empty() {
        Some(tc::read_skill::SkillReference::SkillPath(args.skill_path))
    } else if !args.bundled_skill_id.is_empty() {
        Some(tc::read_skill::SkillReference::BundledSkillId(args.bundled_skill_id))
    } else {
        None
    };
    Ok(tc::Tool::ReadSkill(tc::ReadSkill {
        skill_reference: reference,
        name: args.name,
    }))
}

fn parse_upload_file_artifact(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { file_path: String, #[serde(default)] description: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::UploadFileArtifact(api::UploadFileArtifact {
        file: Some(api::FilePathReference { file_path: args.file_path }),
        description: args.description,
    }))
}

fn parse_read_mcp_resource(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { uri: String, #[serde(default)] server_id: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::ReadMcpResource(tc::ReadMcpResource {
        uri: args.uri, server_id: args.server_id,
    }))
}

fn parse_call_mcp_tool(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        name: String,
        #[serde(default)] args: Value,
        #[serde(default)] server_id: String,
    }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::CallMcpTool(tc::CallMcpTool {
        name: args.name,
        args: Some(json_to_proto_struct(args.args)),
        server_id: args.server_id,
    }))
}

fn parse_write_to_long_running(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        command_id: String,
        input: String,
        #[serde(default)] mode: String,
    }
    let args: Args = serde_json::from_value(input)?;
    let mode_oneof = match args.mode.as_str() {
        "line" => tc::write_to_long_running_shell_command::mode::Mode::Line(()),
        "block" => tc::write_to_long_running_shell_command::mode::Mode::Block(()),
        _ => tc::write_to_long_running_shell_command::mode::Mode::Raw(()),
    };
    Ok(tc::Tool::WriteToLongRunningShellCommand(
        tc::WriteToLongRunningShellCommand {
            input: args.input.into_bytes().into(),
            mode: Some(tc::write_to_long_running_shell_command::Mode {
                mode: Some(mode_oneof),
            }),
            command_id: args.command_id,
        },
    ))
}

fn parse_read_shell_command_output(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        command_id: String,
        #[serde(default)] delay_seconds: i64,
        #[serde(default)] wait_for_completion: bool,
    }
    let args: Args = serde_json::from_value(input)?;
    let delay = if args.wait_for_completion {
        Some(tc::read_shell_command_output::Delay::OnCompletion(()))
    } else if args.delay_seconds > 0 {
        Some(tc::read_shell_command_output::Delay::Duration(prost_types::Duration {
            seconds: args.delay_seconds,
            nanos: 0,
        }))
    } else {
        None
    };
    Ok(tc::Tool::ReadShellCommandOutput(tc::ReadShellCommandOutput {
        command_id: args.command_id, delay,
    }))
}

fn parse_read_documents(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { documents: Vec<DocSpec> }
    #[derive(Deserialize)]
    struct DocSpec { document_id: String, #[serde(default)] line_ranges: Vec<LineRange> }
    #[derive(Deserialize)]
    struct LineRange { start: u32, end: u32 }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::ReadDocuments(tc::ReadDocuments {
        documents: args.documents.into_iter().map(|d| tc::read_documents::Document {
            document_id: d.document_id,
            line_ranges: d.line_ranges.into_iter().map(|r| api::FileContentLineRange {
                start: r.start, end: r.end,
            }).collect(),
        }).collect(),
    }))
}

fn parse_edit_documents(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { diffs: Vec<DiffSpec> }
    #[derive(Deserialize)]
    struct DiffSpec { document_id: String, search: String, replace: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::EditDocuments(tc::EditDocuments {
        diffs: args.diffs.into_iter().map(|d| tc::edit_documents::DocumentDiff {
            document_id: d.document_id, search: d.search, replace: d.replace,
        }).collect(),
    }))
}

fn parse_create_documents(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { new_documents: Vec<DocSpec> }
    #[derive(Deserialize)]
    struct DocSpec { title: String, content: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::CreateDocuments(tc::CreateDocuments {
        new_documents: args.new_documents.into_iter().map(|d| tc::create_documents::NewDocument {
            content: d.content, title: d.title,
        }).collect(),
    }))
}

fn parse_insert_review_comments(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        repo_path: String,
        #[serde(default)] base_branch: String,
        #[serde(default)] comments: Vec<Value>,
    }
    let args: Args = serde_json::from_value(input)?;
    // Comments is a complex shape; round-trip whatever the model emits as opaque payload
    // by leaving it empty here. Phase 2.5 can add granular parsing if the model needs to
    // produce these from scratch (rare — they almost always come from a prior tool).
    let _ = args.comments;
    Ok(tc::Tool::InsertReviewComments(tc::InsertReviewComments {
        repo_path: args.repo_path,
        comments: vec![],
        base_branch: args.base_branch,
    }))
}

fn parse_use_computer(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    // Computer-use action lists are too varied to enforce strict types here. We round-trip
    // them as opaque JSON via summary + best-effort parsing of each action.
    #[derive(Deserialize)]
    struct Args {
        #[serde(default)] action_summary: String,
        #[serde(default)] actions: Vec<Value>,
    }
    let args: Args = serde_json::from_value(input)?;
    let actions = args.actions.into_iter().filter_map(parse_use_computer_action).collect();
    Ok(tc::Tool::UseComputer(tc::UseComputer {
        actions,
        post_actions_screenshot_params: None,
        action_summary: args.action_summary,
    }))
}

fn parse_use_computer_action(v: Value) -> Option<api::message::tool_call::use_computer::Action> {
    use api::message::tool_call::use_computer as uc;
    let kind = v.get("type")?.as_str()?.to_string();
    let inner = match kind.as_str() {
        "wait" => {
            let secs = v.get("seconds").and_then(|x| x.as_i64()).unwrap_or(1);
            uc::action::Type::Wait(uc::action::Wait {
                duration: Some(prost_types::Duration { seconds: secs, nanos: 0 }),
            })
        }
        "type_text" => {
            let text = v.get("text").and_then(|x| x.as_str()).unwrap_or("").to_string();
            uc::action::Type::TypeText(uc::action::TypeText { text })
        }
        "mouse_move" => {
            let (x, y) = coords(v.get("to"));
            uc::action::Type::MouseMove(uc::action::MouseMove {
                to: Some(api::Coordinates { x, y }),
            })
        }
        "mouse_down" => {
            let (x, y) = coords(v.get("at"));
            uc::action::Type::MouseDown(uc::action::MouseDown {
                button: mouse_button(v.get("button")),
                at: Some(api::Coordinates { x, y }),
            })
        }
        "mouse_up" => uc::action::Type::MouseUp(uc::action::MouseUp {
            button: mouse_button(v.get("button")),
        }),
        "mouse_wheel" => {
            let (x, y) = coords(v.get("at"));
            uc::action::Type::MouseWheel(uc::action::MouseWheel {
                at: Some(api::Coordinates { x, y }),
                direction: scroll_direction(v.get("direction")),
                distance: v.get("pixels").and_then(|p| p.as_i64()).map(|p| {
                    uc::action::mouse_wheel::Distance::Pixels(p as i32)
                }).or_else(|| {
                    v.get("clicks").and_then(|c| c.as_i64()).map(|c| {
                        uc::action::mouse_wheel::Distance::Clicks(c as i32)
                    })
                }),
            })
        }
        "key_down" | "key_up" => {
            let key = v.get("key").and_then(|k| k.as_str()).unwrap_or("");
            let key_proto = if key.len() == 1 {
                uc::action::Key { data: Some(uc::action::key::Data::Char(key.to_string())) }
            } else {
                let keycode = v.get("keycode").and_then(|k| k.as_i64()).unwrap_or(0);
                uc::action::Key { data: Some(uc::action::key::Data::Keycode(keycode as i32)) }
            };
            if kind == "key_down" {
                uc::action::Type::KeyDown(uc::action::KeyDown { key: Some(key_proto) })
            } else {
                uc::action::Type::KeyUp(uc::action::KeyUp { key: Some(key_proto) })
            }
        }
        _ => return None,
    };
    Some(uc::Action { r#type: Some(inner) })
}

fn coords(v: Option<&Value>) -> (i32, i32) {
    let o = match v {
        Some(o) => o,
        None => return (0, 0),
    };
    let x = o.get("x").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
    let y = o.get("y").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
    (x, y)
}

fn mouse_button(v: Option<&Value>) -> i32 {
    use api::message::tool_call::use_computer::action::MouseButton;
    let s = v.and_then(|x| x.as_str()).unwrap_or("left");
    match s.to_ascii_lowercase().as_str() {
        "right" => MouseButton::Right as i32,
        "middle" => MouseButton::Middle as i32,
        "back" => MouseButton::Back as i32,
        "forward" => MouseButton::Forward as i32,
        _ => MouseButton::Left as i32,
    }
}

fn scroll_direction(v: Option<&Value>) -> i32 {
    use api::message::tool_call::use_computer::action::mouse_wheel::Direction;
    let s = v.and_then(|x| x.as_str()).unwrap_or("down");
    match s.to_ascii_lowercase().as_str() {
        "up" => Direction::Up as i32,
        "left" => Direction::Left as i32,
        "right" => Direction::Right as i32,
        _ => Direction::Down as i32,
    }
}

fn parse_request_computer_use(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { task_summary: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::RequestComputerUse(tc::RequestComputerUse {
        task_summary: args.task_summary,
        screenshot_params: None,
    }))
}

fn parse_subagent(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        #[serde(default)] task_id: String,
        payload: String,
        kind: String,
        #[serde(default)] command_id: String,
        #[serde(default)] query: String,
        #[serde(default)] conversation_id: String,
    }
    let args: Args = serde_json::from_value(input)?;
    let metadata = match args.kind.as_str() {
        "cli" => Some(tc::subagent::Metadata::Cli(tc::subagent::CliSubagent {
            command_id: args.command_id,
        })),
        "research" => Some(tc::subagent::Metadata::Research(())),
        "advice" => Some(tc::subagent::Metadata::Advice(())),
        "computer_use" => Some(tc::subagent::Metadata::ComputerUse(())),
        "summarization" => Some(tc::subagent::Metadata::Summarization(())),
        "conversation_search" => Some(tc::subagent::Metadata::ConversationSearch(
            tc::subagent::ConversationSearchMetadata {
                query: args.query, conversation_id: args.conversation_id,
            },
        )),
        "warp_documentation_search" => {
            Some(tc::subagent::Metadata::WarpDocumentationSearch(()))
        }
        _ => None,
    };
    Ok(tc::Tool::Subagent(tc::Subagent {
        task_id: args.task_id,
        payload: args.payload,
        metadata,
    }))
}

fn parse_start_agent(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        name: String,
        prompt: String,
        #[serde(default)] remote_environment_id: String,
    }
    let args: Args = serde_json::from_value(input)?;
    let execution_mode = if args.remote_environment_id.is_empty() {
        Some(api::start_agent::ExecutionMode {
            mode: Some(api::start_agent::execution_mode::Mode::Local(())),
        })
    } else {
        Some(api::start_agent::ExecutionMode {
            mode: Some(api::start_agent::execution_mode::Mode::Remote(
                api::start_agent::execution_mode::Remote {
                    environment_id: args.remote_environment_id,
                },
            )),
        })
    };
    Ok(tc::Tool::StartAgent(api::StartAgent {
        name: args.name,
        prompt: args.prompt,
        lifecycle_subscription: None,
        execution_mode,
    }))
}

fn parse_start_agent_v2(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        name: String,
        prompt: String,
        #[serde(default)] execution: String,
        #[serde(default)] harness: String,
        #[serde(default)] model_id: String,
        #[serde(default)] remote_environment_id: String,
        #[serde(default)] title: String,
    }
    let args: Args = serde_json::from_value(input)?;
    let mode = if args.execution == "remote" {
        let harness = if args.harness.is_empty() {
            None
        } else {
            Some(api::start_agent_v2::execution_mode::Harness { r#type: args.harness })
        };
        Some(api::start_agent_v2::execution_mode::Mode::Remote(
            api::start_agent_v2::execution_mode::Remote {
                environment_id: args.remote_environment_id,
                skills: vec![],
                model_id: args.model_id,
                computer_use_enabled: false,
                worker_host: String::new(),
                harness,
                title: args.title,
            },
        ))
    } else {
        let harness = if args.harness.is_empty() {
            None
        } else {
            Some(api::start_agent_v2::execution_mode::Harness { r#type: args.harness })
        };
        Some(api::start_agent_v2::execution_mode::Mode::Local(
            api::start_agent_v2::execution_mode::Local { harness },
        ))
    };
    Ok(tc::Tool::StartAgentV2(api::StartAgentV2 {
        name: args.name,
        prompt: args.prompt,
        lifecycle_subscription: None,
        execution_mode: Some(api::start_agent_v2::ExecutionMode { mode }),
    }))
}

fn parse_send_message_to_agent(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args {
        addresses: Vec<String>,
        #[serde(default)] subject: String,
        message: String,
    }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::SendMessageToAgent(api::SendMessageToAgent {
        addresses: args.addresses, subject: args.subject, message: args.message,
    }))
}

fn parse_fetch_conversation(input: Value) -> Result<api::message::tool_call::Tool> {
    use api::message::tool_call as tc;
    #[derive(Deserialize)]
    struct Args { conversation_id: String }
    let args: Args = serde_json::from_value(input)?;
    Ok(tc::Tool::FetchConversation(tc::FetchConversation {
        conversation_id: args.conversation_id,
    }))
}

// ───────────────────────── Reconstruction (for replay) ─────────────────────────

pub fn tool_call_to_anthropic_name(tool: &api::message::tool_call::Tool) -> Option<&'static str> {
    use api::message::tool_call::Tool;
    Some(match tool {
        Tool::ReadFiles(_) => "read_files",
        Tool::ApplyFileDiffs(_) => "apply_file_diffs",
        Tool::RunShellCommand(_) => "run_shell_command",
        Tool::Grep(_) => "grep",
        Tool::FileGlobV2(_) => "file_glob",
        #[allow(deprecated)]
        Tool::FileGlob(_) => "file_glob_legacy",
        Tool::SearchCodebase(_) => "search_codebase",
        Tool::SuggestPlan(_) => "suggest_plan",
        Tool::SuggestCreatePlan(_) => "suggest_create_plan",
        Tool::SuggestNewConversation(_) => "suggest_new_conversation",
        Tool::SuggestPrompt(_) => "suggest_prompt",
        Tool::AskUserQuestion(_) => "ask_user_question",
        Tool::TransferShellCommandControlToUser(_) => "transfer_shell_command_control_to_user",
        Tool::OpenCodeReview(_) => "open_code_review",
        Tool::InitProject(_) => "init_project",
        Tool::ReadSkill(_) => "read_skill",
        Tool::UploadFileArtifact(_) => "upload_file_artifact",
        Tool::ReadMcpResource(_) => "read_mcp_resource",
        Tool::CallMcpTool(_) => "call_mcp_tool",
        Tool::WriteToLongRunningShellCommand(_) => "write_to_long_running_shell_command",
        Tool::ReadShellCommandOutput(_) => "read_shell_command_output",
        Tool::ReadDocuments(_) => "read_documents",
        Tool::EditDocuments(_) => "edit_documents",
        Tool::CreateDocuments(_) => "create_documents",
        Tool::InsertReviewComments(_) => "insert_review_comments",
        Tool::UseComputer(_) => "use_computer",
        Tool::RequestComputerUse(_) => "request_computer_use",
        Tool::Subagent(_) => "subagent",
        Tool::StartAgent(_) => "start_agent",
        Tool::StartAgentV2(_) => "start_agent_v2",
        Tool::SendMessageToAgent(_) => "send_message_to_agent",
        Tool::FetchConversation(_) => "fetch_conversation",
        // `Server` is a server-only tool the client never sees as a tool call.
        Tool::Server(_) => return None,
    })
}

pub fn tool_call_to_anthropic_input(tool: &api::message::tool_call::Tool) -> Value {
    use api::message::tool_call::Tool;
    match tool {
        Tool::ReadFiles(t) => json!({
            "files": t.files.iter().map(|f| json!({
                "name": f.name,
                "line_ranges": f.line_ranges.iter().map(|r| json!({
                    "start": r.start, "end": r.end,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        }),
        Tool::ApplyFileDiffs(t) => json!({
            "summary": t.summary,
            "diffs": t.diffs.iter().map(|d| json!({
                "file_path": d.file_path, "search": d.search, "replace": d.replace,
            })).collect::<Vec<_>>(),
            "new_files": t.new_files.iter().map(|f| json!({
                "file_path": f.file_path, "content": f.content,
            })).collect::<Vec<_>>(),
            "deleted_files": t.deleted_files.iter().map(|f| json!({
                "file_path": f.file_path,
            })).collect::<Vec<_>>(),
        }),
        Tool::RunShellCommand(t) => {
            #[allow(deprecated)]
            let is_read_only = t.is_read_only;
            json!({"command": t.command, "is_read_only": is_read_only})
        }
        Tool::Grep(t) => json!({"queries": t.queries, "path": t.path}),
        Tool::FileGlobV2(t) => json!({
            "patterns": t.patterns, "search_dir": t.search_dir,
            "max_matches": t.max_matches, "max_depth": t.max_depth, "min_depth": t.min_depth,
        }),
        #[allow(deprecated)]
        Tool::FileGlob(t) => json!({"patterns": t.patterns, "path": t.path}),
        Tool::SearchCodebase(t) => json!({
            "query": t.query, "path_filters": t.path_filters, "codebase_path": t.codebase_path,
        }),
        Tool::SuggestPlan(t) => json!({
            "summary": t.summary,
            "proposed_tasks": t.proposed_tasks.iter().map(|task| json!({
                "id": task.id, "description": task.description,
            })).collect::<Vec<_>>(),
        }),
        Tool::SuggestCreatePlan(_) => json!({}),
        Tool::SuggestNewConversation(t) => json!({"message_id": t.message_id}),
        Tool::SuggestPrompt(t) => match t.display_mode.as_ref() {
            Some(api::message::tool_call::suggest_prompt::DisplayMode::PromptChip(c)) => {
                json!({"prompt": c.prompt, "label": c.label})
            }
            Some(api::message::tool_call::suggest_prompt::DisplayMode::InlineQueryBanner(b)) => {
                json!({"prompt": b.query, "label": b.title})
            }
            None => json!({}),
        },
        Tool::AskUserQuestion(t) => json!({
            "questions": t.questions.iter().map(|q| {
                let (options, ri, multi, other) = match &q.question_type {
                    Some(api::ask_user_question::question::QuestionType::MultipleChoice(mc)) => (
                        mc.options.iter().map(|o| o.label.clone()).collect::<Vec<_>>(),
                        mc.recommended_option_index, mc.is_multiselect, mc.supports_other,
                    ),
                    None => (vec![], 0, false, false),
                };
                json!({
                    "question_id": q.question_id, "question": q.question,
                    "options": options, "recommended_option_index": ri,
                    "is_multiselect": multi, "supports_other": other,
                })
            }).collect::<Vec<_>>(),
        }),
        Tool::TransferShellCommandControlToUser(t) => json!({"reason": t.reason}),
        Tool::OpenCodeReview(_) | Tool::InitProject(_) => json!({}),
        Tool::ReadSkill(t) => match &t.skill_reference {
            Some(api::message::tool_call::read_skill::SkillReference::SkillPath(p)) => {
                json!({"skill_path": p, "name": t.name})
            }
            Some(api::message::tool_call::read_skill::SkillReference::BundledSkillId(id)) => {
                json!({"bundled_skill_id": id, "name": t.name})
            }
            None => json!({"name": t.name}),
        },
        Tool::UploadFileArtifact(t) => json!({
            "file_path": t.file.as_ref().map(|f| f.file_path.clone()).unwrap_or_default(),
            "description": t.description,
        }),
        Tool::ReadMcpResource(t) => json!({"uri": t.uri, "server_id": t.server_id}),
        Tool::CallMcpTool(t) => json!({
            "name": t.name,
            "args": proto_struct_to_json(t.args.as_ref()),
            "server_id": t.server_id,
        }),
        Tool::WriteToLongRunningShellCommand(t) => {
            let mode = match &t.mode.as_ref().and_then(|m| m.mode.as_ref()) {
                Some(api::message::tool_call::write_to_long_running_shell_command::mode::Mode::Line(_)) => "line",
                Some(api::message::tool_call::write_to_long_running_shell_command::mode::Mode::Block(_)) => "block",
                _ => "raw",
            };
            json!({
                "command_id": t.command_id,
                "input": String::from_utf8_lossy(&t.input).to_string(),
                "mode": mode,
            })
        }
        Tool::ReadShellCommandOutput(t) => match &t.delay {
            Some(api::message::tool_call::read_shell_command_output::Delay::Duration(d)) => {
                json!({"command_id": t.command_id, "delay_seconds": d.seconds})
            }
            Some(api::message::tool_call::read_shell_command_output::Delay::OnCompletion(_)) => {
                json!({"command_id": t.command_id, "wait_for_completion": true})
            }
            None => json!({"command_id": t.command_id}),
        },
        Tool::ReadDocuments(t) => json!({
            "documents": t.documents.iter().map(|d| json!({
                "document_id": d.document_id,
                "line_ranges": d.line_ranges.iter().map(|r| json!({
                    "start": r.start, "end": r.end,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        }),
        Tool::EditDocuments(t) => json!({
            "diffs": t.diffs.iter().map(|d| json!({
                "document_id": d.document_id, "search": d.search, "replace": d.replace,
            })).collect::<Vec<_>>(),
        }),
        Tool::CreateDocuments(t) => json!({
            "new_documents": t.new_documents.iter().map(|d| json!({
                "title": d.title, "content": d.content,
            })).collect::<Vec<_>>(),
        }),
        Tool::InsertReviewComments(t) => json!({
            "repo_path": t.repo_path, "base_branch": t.base_branch,
            "comments": t.comments.len(),
        }),
        Tool::UseComputer(t) => json!({
            "action_summary": t.action_summary,
            "actions": t.actions.iter().map(use_computer_action_to_json).collect::<Vec<_>>(),
        }),
        Tool::RequestComputerUse(t) => json!({"task_summary": t.task_summary}),
        Tool::Subagent(t) => {
            let kind = match &t.metadata {
                Some(api::message::tool_call::subagent::Metadata::Cli(_)) => "cli",
                Some(api::message::tool_call::subagent::Metadata::Research(_)) => "research",
                Some(api::message::tool_call::subagent::Metadata::Advice(_)) => "advice",
                Some(api::message::tool_call::subagent::Metadata::ComputerUse(_)) => "computer_use",
                Some(api::message::tool_call::subagent::Metadata::Summarization(_)) => "summarization",
                Some(api::message::tool_call::subagent::Metadata::ConversationSearch(_)) => "conversation_search",
                Some(api::message::tool_call::subagent::Metadata::WarpDocumentationSearch(_)) => "warp_documentation_search",
                None => "research",
            };
            json!({"task_id": t.task_id, "payload": t.payload, "kind": kind})
        }
        Tool::StartAgent(t) => json!({
            "name": t.name, "prompt": t.prompt,
            "remote_environment_id": match t.execution_mode.as_ref().and_then(|e| e.mode.as_ref()) {
                Some(api::start_agent::execution_mode::Mode::Remote(r)) => r.environment_id.clone(),
                _ => String::new(),
            },
        }),
        Tool::StartAgentV2(t) => {
            let mut obj = json!({"name": t.name, "prompt": t.prompt});
            if let Some(em) = t.execution_mode.as_ref().and_then(|e| e.mode.as_ref()) {
                match em {
                    api::start_agent_v2::execution_mode::Mode::Local(l) => {
                        obj["execution"] = "local".into();
                        if let Some(h) = &l.harness {
                            obj["harness"] = h.r#type.clone().into();
                        }
                    }
                    api::start_agent_v2::execution_mode::Mode::Remote(r) => {
                        obj["execution"] = "remote".into();
                        obj["remote_environment_id"] = r.environment_id.clone().into();
                        obj["model_id"] = r.model_id.clone().into();
                        obj["title"] = r.title.clone().into();
                    }
                }
            }
            obj
        }
        Tool::SendMessageToAgent(t) => json!({
            "addresses": t.addresses, "subject": t.subject, "message": t.message,
        }),
        Tool::FetchConversation(t) => json!({"conversation_id": t.conversation_id}),
        Tool::Server(_) => json!({}),
    }
}

fn use_computer_action_to_json(a: &api::message::tool_call::use_computer::Action) -> Value {
    use api::message::tool_call::use_computer::action as ua;
    match &a.r#type {
        Some(ua::Type::Wait(w)) => json!({
            "type": "wait",
            "seconds": w.duration.as_ref().map(|d| d.seconds).unwrap_or(0),
        }),
        Some(ua::Type::TypeText(t)) => json!({"type": "type_text", "text": t.text}),
        Some(ua::Type::MouseMove(m)) => json!({
            "type": "mouse_move",
            "to": m.to.as_ref().map(|c| json!({"x": c.x, "y": c.y})).unwrap_or(json!({})),
        }),
        Some(ua::Type::MouseDown(m)) => json!({
            "type": "mouse_down",
            "button": mouse_button_name(m.button),
            "at": m.at.as_ref().map(|c| json!({"x": c.x, "y": c.y})).unwrap_or(json!({})),
        }),
        Some(ua::Type::MouseUp(m)) => json!({
            "type": "mouse_up",
            "button": mouse_button_name(m.button),
        }),
        Some(ua::Type::MouseWheel(m)) => json!({
            "type": "mouse_wheel",
            "at": m.at.as_ref().map(|c| json!({"x": c.x, "y": c.y})).unwrap_or(json!({})),
            "direction": scroll_direction_name(m.direction),
        }),
        Some(ua::Type::KeyDown(k)) => json!({
            "type": "key_down",
            "key": key_to_string(k.key.as_ref()),
        }),
        Some(ua::Type::KeyUp(k)) => json!({
            "type": "key_up",
            "key": key_to_string(k.key.as_ref()),
        }),
        None => json!({}),
    }
}

fn mouse_button_name(b: i32) -> &'static str {
    use api::message::tool_call::use_computer::action::MouseButton;
    match MouseButton::try_from(b).unwrap_or(MouseButton::Left) {
        MouseButton::Left => "left",
        MouseButton::Right => "right",
        MouseButton::Middle => "middle",
        MouseButton::Back => "back",
        MouseButton::Forward => "forward",
    }
}

fn scroll_direction_name(d: i32) -> &'static str {
    use api::message::tool_call::use_computer::action::mouse_wheel::Direction;
    match Direction::try_from(d).unwrap_or(Direction::Down) {
        Direction::Up => "up",
        Direction::Down => "down",
        Direction::Left => "left",
        Direction::Right => "right",
    }
}

fn key_to_string(k: Option<&api::message::tool_call::use_computer::action::Key>) -> String {
    use api::message::tool_call::use_computer::action::key::Data;
    match k.and_then(|k| k.data.as_ref()) {
        Some(Data::Char(s)) => s.clone(),
        Some(Data::Keycode(c)) => format!("{c}"),
        None => String::new(),
    }
}

// ───────────────────────── Result rendering ─────────────────────────

pub enum ResultView<'a> {
    ReadFiles(&'a api::ReadFilesResult),
    ApplyFileDiffs(&'a api::ApplyFileDiffsResult),
    RunShellCommand(&'a api::RunShellCommandResult),
    Grep(&'a api::GrepResult),
    FileGlobV2(&'a api::FileGlobV2Result),
    #[allow(deprecated)]
    FileGlob(&'a api::FileGlobResult),
    SearchCodebase(&'a api::SearchCodebaseResult),
    SuggestPlan(&'a api::SuggestPlanResult),
    SuggestCreatePlan(&'a api::SuggestCreatePlanResult),
    SuggestNewConversation(&'a api::SuggestNewConversationResult),
    SuggestPrompt(&'a api::SuggestPromptResult),
    AskUserQuestion(&'a api::AskUserQuestionResult),
    TransferShellCommandControl(&'a api::TransferShellCommandControlToUserResult),
    OpenCodeReview(&'a api::OpenCodeReviewResult),
    InitProject(&'a api::InitProjectResult),
    ReadSkill(&'a api::ReadSkillResult),
    UploadFileArtifact(&'a api::UploadFileArtifactResult),
    ReadMcpResource(&'a api::ReadMcpResourceResult),
    CallMcpTool(&'a api::CallMcpToolResult),
    WriteToLongRunning(&'a api::WriteToLongRunningShellCommandResult),
    ReadShellCommandOutput(&'a api::ReadShellCommandOutputResult),
    ReadDocuments(&'a api::ReadDocumentsResult),
    EditDocuments(&'a api::EditDocumentsResult),
    CreateDocuments(&'a api::CreateDocumentsResult),
    InsertReviewComments(&'a api::InsertReviewCommentsResult),
    UseComputer(&'a api::UseComputerResult),
    RequestComputerUse(&'a api::RequestComputerUseResult),
    FetchConversation(&'a api::FetchConversationResult),
    StartAgent(&'a api::StartAgentResult),
    StartAgentV2(&'a api::StartAgentV2Result),
    SendMessageToAgent(&'a api::SendMessageToAgentResult),
    Cancelled,
    Other,
}

impl<'a> From<&'a api::message::tool_call_result::Result> for ResultView<'a> {
    fn from(r: &'a api::message::tool_call_result::Result) -> Self {
        use api::message::tool_call_result::Result as R;
        match r {
            R::ReadFiles(v) => ResultView::ReadFiles(v),
            R::ApplyFileDiffs(v) => ResultView::ApplyFileDiffs(v),
            R::RunShellCommand(v) => ResultView::RunShellCommand(v),
            R::Grep(v) => ResultView::Grep(v),
            R::FileGlobV2(v) => ResultView::FileGlobV2(v),
            #[allow(deprecated)]
            R::FileGlob(v) => ResultView::FileGlob(v),
            R::SearchCodebase(v) => ResultView::SearchCodebase(v),
            R::SuggestPlan(v) => ResultView::SuggestPlan(v),
            R::SuggestCreatePlan(v) => ResultView::SuggestCreatePlan(v),
            R::SuggestNewConversation(v) => ResultView::SuggestNewConversation(v),
            R::SuggestPrompt(v) => ResultView::SuggestPrompt(v),
            R::AskUserQuestion(v) => ResultView::AskUserQuestion(v),
            R::TransferShellCommandControlToUser(v) => ResultView::TransferShellCommandControl(v),
            R::OpenCodeReview(v) => ResultView::OpenCodeReview(v),
            R::InitProject(v) => ResultView::InitProject(v),
            R::ReadSkill(v) => ResultView::ReadSkill(v),
            R::UploadFileArtifact(v) => ResultView::UploadFileArtifact(v),
            R::ReadMcpResource(v) => ResultView::ReadMcpResource(v),
            R::CallMcpTool(v) => ResultView::CallMcpTool(v),
            R::WriteToLongRunningShellCommand(v) => ResultView::WriteToLongRunning(v),
            R::ReadShellCommandOutput(v) => ResultView::ReadShellCommandOutput(v),
            R::ReadDocuments(v) => ResultView::ReadDocuments(v),
            R::EditDocuments(v) => ResultView::EditDocuments(v),
            R::CreateDocuments(v) => ResultView::CreateDocuments(v),
            R::InsertReviewComments(v) => ResultView::InsertReviewComments(v),
            R::UseComputer(v) => ResultView::UseComputer(v),
            R::RequestComputerUseResult(v) => ResultView::RequestComputerUse(v),
            R::FetchConversation(v) => ResultView::FetchConversation(v),
            R::StartAgent(v) => ResultView::StartAgent(v),
            R::StartAgentV2(v) => ResultView::StartAgentV2(v),
            R::SendMessageToAgent(v) => ResultView::SendMessageToAgent(v),
            R::Cancel(_) => ResultView::Cancelled,
            // Server, Subagent are server-mediated; not surfaced as direct tool text.
            _ => ResultView::Other,
        }
    }
}

impl<'a> From<&'a api::request::input::tool_call_result::Result> for ResultView<'a> {
    fn from(r: &'a api::request::input::tool_call_result::Result) -> Self {
        use api::request::input::tool_call_result::Result as R;
        match r {
            R::ReadFiles(v) => ResultView::ReadFiles(v),
            R::ApplyFileDiffs(v) => ResultView::ApplyFileDiffs(v),
            R::RunShellCommand(v) => ResultView::RunShellCommand(v),
            R::Grep(v) => ResultView::Grep(v),
            R::FileGlobV2(v) => ResultView::FileGlobV2(v),
            #[allow(deprecated)]
            R::FileGlob(v) => ResultView::FileGlob(v),
            R::SearchCodebase(v) => ResultView::SearchCodebase(v),
            R::SuggestPlan(v) => ResultView::SuggestPlan(v),
            R::SuggestCreatePlan(v) => ResultView::SuggestCreatePlan(v),
            R::SuggestNewConversation(v) => ResultView::SuggestNewConversation(v),
            R::SuggestPrompt(v) => ResultView::SuggestPrompt(v),
            R::AskUserQuestion(v) => ResultView::AskUserQuestion(v),
            R::TransferShellCommandControlToUser(v) => ResultView::TransferShellCommandControl(v),
            R::OpenCodeReview(v) => ResultView::OpenCodeReview(v),
            R::InitProject(v) => ResultView::InitProject(v),
            R::ReadSkill(v) => ResultView::ReadSkill(v),
            R::UploadFileArtifact(v) => ResultView::UploadFileArtifact(v),
            R::ReadMcpResource(v) => ResultView::ReadMcpResource(v),
            R::CallMcpTool(v) => ResultView::CallMcpTool(v),
            R::WriteToLongRunningShellCommand(v) => ResultView::WriteToLongRunning(v),
            R::ReadShellCommandOutput(v) => ResultView::ReadShellCommandOutput(v),
            R::ReadDocuments(v) => ResultView::ReadDocuments(v),
            R::EditDocuments(v) => ResultView::EditDocuments(v),
            R::CreateDocuments(v) => ResultView::CreateDocuments(v),
            R::InsertReviewComments(v) => ResultView::InsertReviewComments(v),
            R::UseComputer(v) => ResultView::UseComputer(v),
            R::RequestComputerUse(v) => ResultView::RequestComputerUse(v),
            R::FetchConversation(v) => ResultView::FetchConversation(v),
            R::StartAgent(v) => ResultView::StartAgent(v),
            R::StartAgentV2(v) => ResultView::StartAgentV2(v),
            R::SendMessageToAgent(v) => ResultView::SendMessageToAgent(v),
        }
    }
}

pub fn render_result(view: ResultView<'_>) -> (String, bool) {
    match view {
        ResultView::ReadFiles(rf) => render_read_files(rf),
        ResultView::ApplyFileDiffs(ad) => render_apply_diffs(ad),
        ResultView::RunShellCommand(rs) => render_run_shell(rs),
        ResultView::Grep(gr) => render_grep(gr),
        ResultView::FileGlobV2(fg) => render_file_glob_v2(fg),
        #[allow(deprecated)]
        ResultView::FileGlob(fg) => match fg.result.as_ref() {
            Some(api::file_glob_result::Result::Success(s)) => (s.matched_files.clone(), false),
            Some(api::file_glob_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::SearchCodebase(sc) => match sc.result.as_ref() {
            Some(api::search_codebase_result::Result::Success(s)) => {
                let mut out = String::new();
                for f in &s.files {
                    out.push_str(&format!("=== {} ===\n{}\n", f.file_path, f.content));
                }
                if out.is_empty() {
                    ("(no matches)".into(), false)
                } else {
                    (out, false)
                }
            }
            Some(api::search_codebase_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::SuggestPlan(sp) => match sp.result.as_ref() {
            Some(api::suggest_plan_result::Result::Accepted(_)) => ("(plan accepted)".into(), false),
            Some(api::suggest_plan_result::Result::UserEditedPlan(p)) => (
                format!("(user-edited plan)\n{}", p.plan_text),
                false,
            ),
            None => ("(plan pending)".into(), false),
        },
        ResultView::SuggestCreatePlan(scp) => (
            if scp.accepted { "(plan creation accepted)".into() } else { "(plan creation rejected)".into() },
            false,
        ),
        ResultView::SuggestNewConversation(snc) => match snc.result.as_ref() {
            Some(api::suggest_new_conversation_result::Result::Accepted(_)) => ("(new conversation accepted)".into(), false),
            Some(api::suggest_new_conversation_result::Result::Rejected(_)) => ("(new conversation rejected)".into(), false),
            None => ("(pending)".into(), false),
        },
        ResultView::SuggestPrompt(sp) => match sp.result.as_ref() {
            Some(api::suggest_prompt_result::Result::Accepted(_)) => ("(prompt accepted by user)".into(), false),
            Some(api::suggest_prompt_result::Result::Rejected(_)) => ("(prompt rejected by user)".into(), false),
            None => ("(pending)".into(), false),
        },
        ResultView::AskUserQuestion(auq) => match auq.result.as_ref() {
            Some(api::ask_user_question_result::Result::Success(s)) => {
                let mut out = String::new();
                for a in &s.answers {
                    match a.answer.as_ref() {
                        Some(api::ask_user_question_result::answer_item::Answer::MultipleChoice(m)) => {
                            out.push_str(&format!(
                                "{}: {}{}\n",
                                a.question_id,
                                m.selected_options.join(", "),
                                if m.other_text.is_empty() { String::new() } else { format!(" (other: {})", m.other_text) },
                            ));
                        }
                        Some(api::ask_user_question_result::answer_item::Answer::Skipped(_)) => {
                            out.push_str(&format!("{}: (skipped)\n", a.question_id));
                        }
                        None => {}
                    }
                }
                if out.is_empty() {
                    ("(no answers)".into(), false)
                } else {
                    (out, false)
                }
            }
            Some(api::ask_user_question_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(pending)".into(), false),
        },
        ResultView::TransferShellCommandControl(_) => ("(control transferred to user)".into(), false),
        ResultView::OpenCodeReview(_) => ("(code review opened)".into(), false),
        ResultView::InitProject(_) => ("(project initialized)".into(), false),
        ResultView::ReadSkill(rs) => match rs.result.as_ref() {
            Some(api::read_skill_result::Result::Success(s)) => (
                s.content.as_ref().map(|c| c.content.clone()).unwrap_or_default(),
                false,
            ),
            Some(api::read_skill_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::UploadFileArtifact(ufa) => match ufa.result.as_ref() {
            Some(api::upload_file_artifact_result::Result::Success(s)) => (
                format!("uploaded artifact_uid={} mime={} size={}", s.artifact_uid, s.mime_type, s.size_bytes),
                false,
            ),
            Some(api::upload_file_artifact_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::ReadMcpResource(rmr) => match rmr.result.as_ref() {
            Some(api::read_mcp_resource_result::Result::Success(s)) => {
                let mut out = String::new();
                for c in &s.contents {
                    if let Some(api::mcp_resource_content::ContentType::Text(t)) = c.content_type.as_ref() {
                        out.push_str(&format!("=== {} ({}) ===\n{}\n", c.uri, t.mime_type, t.content));
                    } else {
                        out.push_str(&format!("=== {} (binary) ===\n", c.uri));
                    }
                }
                (out, false)
            }
            Some(api::read_mcp_resource_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::CallMcpTool(cmt) => match cmt.result.as_ref() {
            Some(api::call_mcp_tool_result::Result::Success(s)) => {
                let mut out = String::new();
                for r in &s.results {
                    match r.result.as_ref() {
                        Some(api::call_mcp_tool_result::success::result::Result::Text(t)) => {
                            out.push_str(&t.text);
                            out.push('\n');
                        }
                        Some(api::call_mcp_tool_result::success::result::Result::Image(i)) => {
                            out.push_str(&format!("(image: {} {} bytes)\n", i.mime_type, i.data.len()));
                        }
                        Some(api::call_mcp_tool_result::success::result::Result::Resource(_)) => {
                            out.push_str("(embedded resource)\n");
                        }
                        None => {}
                    }
                }
                (out, false)
            }
            Some(api::call_mcp_tool_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::WriteToLongRunning(w) => match w.result.as_ref() {
            Some(api::write_to_long_running_shell_command_result::Result::CommandFinished(f)) => (
                format!("exit_code={}\n{}", f.exit_code, f.output),
                f.exit_code != 0,
            ),
            Some(api::write_to_long_running_shell_command_result::Result::LongRunningCommandSnapshot(_)) => (
                "(still running)".into(), false,
            ),
            Some(api::write_to_long_running_shell_command_result::Result::Error(_)) => (
                "ERROR: shell command error".into(), true,
            ),
            None => ("(no result)".into(), true),
        },
        ResultView::ReadShellCommandOutput(r) => match r.result.as_ref() {
            Some(api::read_shell_command_output_result::Result::CommandFinished(f)) => (
                format!("exit_code={}\n{}", f.exit_code, f.output),
                f.exit_code != 0,
            ),
            Some(api::read_shell_command_output_result::Result::LongRunningCommandSnapshot(_)) => (
                "(still running)".into(), false,
            ),
            Some(api::read_shell_command_output_result::Result::Error(_)) => (
                "ERROR: shell command error".into(), true,
            ),
            None => ("(no result)".into(), true),
        },
        ResultView::ReadDocuments(rd) => match rd.result.as_ref() {
            Some(api::read_documents_result::Result::Success(s)) => {
                let mut out = String::new();
                for d in &s.documents {
                    out.push_str(&format!("=== document {} ===\n{}\n", d.document_id, d.content));
                }
                (out, false)
            }
            Some(api::read_documents_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::EditDocuments(ed) => match ed.result.as_ref() {
            Some(api::edit_documents_result::Result::Success(s)) => (
                format!("updated {} document(s)", s.updated_documents.len()),
                false,
            ),
            Some(api::edit_documents_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::CreateDocuments(cd) => match cd.result.as_ref() {
            Some(api::create_documents_result::Result::Success(s)) => (
                format!("created {} document(s)", s.created_documents.len()),
                false,
            ),
            Some(api::create_documents_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::InsertReviewComments(irc) => match irc.result.as_ref() {
            Some(api::insert_review_comments_result::Result::Success(_)) => (
                format!("inserted comments into {}", irc.repo_path),
                false,
            ),
            Some(api::insert_review_comments_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::UseComputer(uc) => match uc.result.as_ref() {
            Some(api::use_computer_result::Result::Success(s)) => {
                let cursor = s.cursor_position.as_ref().map(|c| format!("({}, {})", c.x, c.y)).unwrap_or_default();
                let shot = s.screenshot.as_ref().map(|i| format!("{}x{} {}", i.width, i.height, i.mime_type)).unwrap_or_default();
                (format!("ok; cursor={cursor}; screenshot={shot}"), false)
            }
            Some(api::use_computer_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::RequestComputerUse(rcu) => match rcu.result.as_ref() {
            Some(api::request_computer_use_result::Result::Approved(_)) => ("(computer use approved)".into(), false),
            Some(api::request_computer_use_result::Result::Rejected(_)) => ("(computer use rejected)".into(), false),
            Some(api::request_computer_use_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(pending)".into(), false),
        },
        ResultView::FetchConversation(fc) => match fc.result.as_ref() {
            Some(api::fetch_conversation_result::Result::Success(s)) => (
                format!("conversation materialized at {}", s.directory_path),
                false,
            ),
            Some(api::fetch_conversation_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::StartAgent(sa) => match sa.result.as_ref() {
            Some(api::start_agent_result::Result::Success(s)) => (
                format!("started agent {}", s.agent_id), false,
            ),
            Some(api::start_agent_result::Result::Error(e)) => (format!("ERROR: {}", e.error), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::StartAgentV2(sa) => match sa.result.as_ref() {
            Some(api::start_agent_v2_result::Result::Success(s)) => (
                format!("started agent {}", s.agent_id), false,
            ),
            Some(api::start_agent_v2_result::Result::Error(e)) => (format!("ERROR: {}", e.error), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::SendMessageToAgent(sma) => match sma.result.as_ref() {
            Some(api::send_message_to_agent_result::Result::Success(s)) => (
                format!("message sent (id={})", s.message_id), false,
            ),
            Some(api::send_message_to_agent_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
            None => ("(empty result)".into(), true),
        },
        ResultView::Cancelled => ("(tool call cancelled)".into(), false),
        ResultView::Other => ("(tool result type not yet wired)".into(), false),
    }
}

fn render_read_files(rf: &api::ReadFilesResult) -> (String, bool) {
    match rf.result.as_ref() {
        Some(api::read_files_result::Result::TextFilesSuccess(s)) => {
            let mut out = String::new();
            for f in &s.files {
                out.push_str(&format!("=== {} ===\n", f.file_path));
                out.push_str(&f.content);
                if !f.content.ends_with('\n') {
                    out.push('\n');
                }
            }
            (out, false)
        }
        Some(api::read_files_result::Result::AnyFilesSuccess(s)) => (
            format!("read {} non-text file(s)", s.files.len()),
            false,
        ),
        Some(api::read_files_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
        None => ("(empty result)".into(), true),
    }
}

fn render_apply_diffs(ad: &api::ApplyFileDiffsResult) -> (String, bool) {
    match ad.result.as_ref() {
        Some(api::apply_file_diffs_result::Result::Success(s)) => {
            let mut out = String::new();
            if !s.updated_files_v2.is_empty() {
                out.push_str(&format!("Updated {} file(s):\n", s.updated_files_v2.len()));
                for u in &s.updated_files_v2 {
                    if let Some(file) = &u.file {
                        out.push_str(&format!(
                            "  - {}{}\n",
                            file.file_path,
                            if u.was_edited_by_user { " (edited by user)" } else { "" }
                        ));
                    }
                }
            }
            #[allow(deprecated)]
            if !s.updated_files.is_empty() && s.updated_files_v2.is_empty() {
                out.push_str(&format!("Updated {} file(s):\n", s.updated_files.len()));
                for f in &s.updated_files {
                    out.push_str(&format!("  - {}\n", f.file_path));
                }
            }
            if !s.deleted_files.is_empty() {
                out.push_str(&format!("Deleted {} file(s):\n", s.deleted_files.len()));
                for f in &s.deleted_files {
                    out.push_str(&format!("  - {}\n", f.file_path));
                }
            }
            if out.is_empty() {
                out.push_str("(no changes)");
            }
            (out, false)
        }
        Some(api::apply_file_diffs_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
        None => ("(empty result)".into(), true),
    }
}

fn render_run_shell(rs: &api::RunShellCommandResult) -> (String, bool) {
    match rs.result.as_ref() {
        Some(api::run_shell_command_result::Result::CommandFinished(f)) => {
            let body = if f.output.is_empty() { "(no output)".to_string() } else { f.output.clone() };
            (format!("exit_code={}\n{}", f.exit_code, body), f.exit_code != 0)
        }
        Some(api::run_shell_command_result::Result::LongRunningCommandSnapshot(_)) => (
            "(command is still running; check back later)".into(), false,
        ),
        Some(api::run_shell_command_result::Result::PermissionDenied(_)) => (
            "ERROR: permission denied to run this command".into(), true,
        ),
        None => ("(no result)".into(), true),
    }
}

fn render_grep(gr: &api::GrepResult) -> (String, bool) {
    match gr.result.as_ref() {
        Some(api::grep_result::Result::Success(s)) => {
            if s.matched_files.is_empty() {
                ("(no matches)".into(), false)
            } else {
                let mut out = String::new();
                for f in &s.matched_files {
                    let lines: Vec<String> = f.matched_lines.iter().map(|l| l.line_number.to_string()).collect();
                    out.push_str(&format!("{}: lines {}\n", f.file_path, lines.join(", ")));
                }
                (out, false)
            }
        }
        Some(api::grep_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
        None => ("(empty result)".into(), true),
    }
}

fn render_file_glob_v2(fg: &api::FileGlobV2Result) -> (String, bool) {
    match fg.result.as_ref() {
        Some(api::file_glob_v2_result::Result::Success(s)) => {
            if s.matched_files.is_empty() {
                ("(no matches)".into(), false)
            } else {
                let body = s.matched_files.iter().map(|f| f.file_path.as_str()).collect::<Vec<_>>().join("\n");
                (body, false)
            }
        }
        Some(api::file_glob_v2_result::Result::Error(e)) => (format!("ERROR: {}", e.message), true),
        None => ("(empty result)".into(), true),
    }
}

pub fn render_message_result(r: &api::message::ToolCallResult) -> (String, bool) {
    match r.result.as_ref() {
        Some(inner) => render_result(inner.into()),
        None => ("(no tool result)".into(), true),
    }
}

pub fn render_input_result(r: &api::request::input::ToolCallResult) -> (String, bool) {
    match r.result.as_ref() {
        Some(inner) => render_result(inner.into()),
        None => ("(no tool result)".into(), true),
    }
}

// ───────────────────────── google.protobuf.Struct ↔ JSON ─────────────────────────

fn json_to_proto_struct(v: Value) -> ProtoStruct {
    let mut fields = std::collections::BTreeMap::new();
    if let Value::Object(map) = v {
        for (k, v) in map {
            fields.insert(k, json_to_proto_value(v));
        }
    }
    ProtoStruct { fields }
}

fn json_to_proto_value(v: Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match v {
        Value::Null => Kind::NullValue(0),
        Value::Bool(b) => Kind::BoolValue(b),
        Value::Number(n) => Kind::NumberValue(n.as_f64().unwrap_or(0.0)),
        Value::String(s) => Kind::StringValue(s),
        Value::Array(arr) => Kind::ListValue(prost_types::ListValue {
            values: arr.into_iter().map(json_to_proto_value).collect(),
        }),
        Value::Object(map) => {
            let mut fields = std::collections::BTreeMap::new();
            for (k, v) in map {
                fields.insert(k, json_to_proto_value(v));
            }
            Kind::StructValue(ProtoStruct { fields })
        }
    };
    prost_types::Value { kind: Some(kind) }
}

fn proto_struct_to_json(s: Option<&ProtoStruct>) -> Value {
    let Some(s) = s else { return Value::Null };
    let mut map = serde_json::Map::new();
    for (k, v) in &s.fields {
        map.insert(k.clone(), proto_value_to_json(v));
    }
    Value::Object(map)
}

fn proto_value_to_json(v: &prost_types::Value) -> Value {
    use prost_types::value::Kind;
    match v.kind.as_ref() {
        Some(Kind::NullValue(_)) | None => Value::Null,
        Some(Kind::BoolValue(b)) => Value::Bool(*b),
        Some(Kind::NumberValue(n)) => serde_json::Number::from_f64(*n).map(Value::Number).unwrap_or(Value::Null),
        Some(Kind::StringValue(s)) => Value::String(s.clone()),
        Some(Kind::ListValue(l)) => Value::Array(l.values.iter().map(proto_value_to_json).collect()),
        Some(Kind::StructValue(s)) => {
            let mut map = serde_json::Map::new();
            for (k, v) in &s.fields {
                map.insert(k.clone(), proto_value_to_json(v));
            }
            Value::Object(map)
        }
    }
}

pub fn system_prompt() -> String {
    "You are open-warp's coding assistant, a self-hosted clone of Warp's Agent Mode. \
     You have a full toolbox for inspecting, searching, running, editing, and orchestrating.\n\
     \n\
     Workflow guidelines:\n\
     - Use `file_glob` to locate files and `grep` to find references; chain into `read_files`.\n\
     - Use `apply_file_diffs` for ALL file edits — never write files via shell.\n\
     - Mark `run_shell_command` `is_read_only=true` for inspection commands so they auto-run.\n\
     - Use `ask_user_question` only when you genuinely cannot make progress without input.\n\
     - Use `start_agent` / `start_agent_v2` to spawn sub-agents for clearly-bounded \
     parallel work; otherwise stay in the main loop.\n\
     - Be concise. Don't narrate; just do the work."
        .to_string()
}
