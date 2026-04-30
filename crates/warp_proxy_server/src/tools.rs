//! Tool definitions: JSON schemas (for the provider's `tools` array) plus translators
//! between provider tool-use JSON ↔ Warp `Message::ToolCall` proto and Warp
//! `Message::ToolCallResult` proto → provider tool-result string.
//!
//! Phase 2.2 implements the 5 essential tools for a coding assistant:
//! `read_files`, `apply_file_diffs`, `run_shell_command`, `grep`, `file_glob`. Additional
//! tools land in subsequent phases.

use anyhow::{Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};
use warp_multi_agent_api as api;

use crate::providers::ToolSpec;

/// Build the list of provider-side tool specs from Warp's `supported_tools` list. Tools we
/// don't yet translate are silently skipped — the model just won't be told they exist.
pub fn build_tool_specs(supported: &[i32]) -> Vec<ToolSpec> {
    supported
        .iter()
        .filter_map(|id| api::ToolType::try_from(*id).ok())
        .filter_map(tool_spec)
        .collect()
}

fn tool_spec(t: api::ToolType) -> Option<ToolSpec> {
    use api::ToolType::*;
    let (name, description, schema) = match t {
        ReadFiles => (
            "read_files",
            "Read the contents of one or more text files in the workspace. Returns each \
             file's contents (optionally limited to specific line ranges). Use this whenever \
             you need to inspect source code or other text files before editing.",
            json!({
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Relative or absolute path to the file."},
                                "line_ranges": {
                                    "type": "array",
                                    "description": "Optional inclusive line ranges. If omitted, returns the entire file.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start": {"type": "integer", "minimum": 1},
                                            "end": {"type": "integer", "minimum": 1}
                                        },
                                        "required": ["start", "end"]
                                    }
                                }
                            },
                            "required": ["name"]
                        }
                    }
                },
                "required": ["files"]
            }),
        ),
        ApplyFileDiffs => (
            "apply_file_diffs",
            "Apply edits to existing files (exact-string search/replace), create new files, \
             or delete files. Always use this — never write files via shell. Each diff's \
             `search` must match the file content exactly once.",
            json!({
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "One-line summary of the change."},
                    "diffs": {
                        "type": "array",
                        "description": "Edits to existing files.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "search": {"type": "string", "description": "Exact content to replace. Must be unique."},
                                "replace": {"type": "string", "description": "Replacement content."}
                            },
                            "required": ["file_path", "search", "replace"]
                        }
                    },
                    "new_files": {
                        "type": "array",
                        "description": "Files to create.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["file_path", "content"]
                        }
                    },
                    "deleted_files": {
                        "type": "array",
                        "description": "Files to delete.",
                        "items": {
                            "type": "object",
                            "properties": {"file_path": {"type": "string"}},
                            "required": ["file_path"]
                        }
                    }
                },
                "required": ["summary"]
            }),
        ),
        RunShellCommand => (
            "run_shell_command",
            "Run a shell command in the user's terminal. Returns stdout, stderr, and exit \
             code. Set `is_read_only` to true for commands that don't change state (ls, cat, \
             git status, …) so they run without confirmation.",
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The full command to run."},
                    "is_read_only": {"type": "boolean", "description": "True if the command does not modify state."}
                },
                "required": ["command"]
            }),
        ),
        Grep => (
            "grep",
            "Search file contents for one or more regex patterns. Returns matching files \
             and line numbers. Pair with `read_files` to read the matched files for context.",
            json!({
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                        "description": "Patterns to search for."
                    },
                    "path": {"type": "string", "description": "Relative path to search in (defaults to working directory)."}
                },
                "required": ["queries"]
            }),
        ),
        FileGlobV2 => (
            "file_glob",
            "Find files matching glob patterns (e.g. '**/*.rs'). Use this to locate files \
             before reading.",
            json!({
                "type": "object",
                "properties": {
                    "patterns": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"}
                    },
                    "search_dir": {"type": "string", "description": "Directory to search in (default: working directory)."},
                    "max_matches": {"type": "integer", "description": "Maximum results. 0 = no limit."},
                    "max_depth": {"type": "integer", "description": "Maximum traversal depth. 0 = no limit."},
                    "min_depth": {"type": "integer", "description": "Minimum traversal depth. 0 = no limit."}
                },
                "required": ["patterns"]
            }),
        ),
        // Tools not yet translated — quietly skip so the model never tries to call them.
        _ => return None,
    };
    Some(ToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        input_schema: schema,
    })
}

/// Convert a provider tool-use into a Warp `ToolCall` message payload.
pub fn anthropic_tool_use_to_warp(
    name: &str,
    tool_call_id: String,
    input: Value,
) -> Result<api::message::ToolCall> {
    use api::message::tool_call as tc;

    let tool = match name {
        "read_files" => {
            #[derive(Deserialize)]
            struct Args {
                files: Vec<FileSpec>,
            }
            #[derive(Deserialize)]
            struct FileSpec {
                name: String,
                #[serde(default)]
                line_ranges: Vec<LineRange>,
            }
            #[derive(Deserialize)]
            struct LineRange {
                start: u32,
                end: u32,
            }
            let args: Args = serde_json::from_value(input)?;
            tc::Tool::ReadFiles(tc::ReadFiles {
                files: args
                    .files
                    .into_iter()
                    .map(|f| tc::read_files::File {
                        name: f.name,
                        line_ranges: f
                            .line_ranges
                            .into_iter()
                            .map(|r| api::FileContentLineRange {
                                start: r.start,
                                end: r.end,
                            })
                            .collect(),
                    })
                    .collect(),
            })
        }
        "apply_file_diffs" => {
            #[derive(Deserialize, Default)]
            struct Args {
                #[serde(default)]
                summary: String,
                #[serde(default)]
                diffs: Vec<DiffSpec>,
                #[serde(default)]
                new_files: Vec<NewFileSpec>,
                #[serde(default)]
                deleted_files: Vec<DeletedFileSpec>,
            }
            #[derive(Deserialize)]
            struct DiffSpec {
                file_path: String,
                search: String,
                replace: String,
            }
            #[derive(Deserialize)]
            struct NewFileSpec {
                file_path: String,
                #[serde(default)]
                content: String,
            }
            #[derive(Deserialize)]
            struct DeletedFileSpec {
                file_path: String,
            }
            let args: Args = serde_json::from_value(input)?;
            tc::Tool::ApplyFileDiffs(tc::ApplyFileDiffs {
                summary: args.summary,
                diffs: args
                    .diffs
                    .into_iter()
                    .map(|d| tc::apply_file_diffs::FileDiff {
                        file_path: d.file_path,
                        search: d.search,
                        replace: d.replace,
                    })
                    .collect(),
                new_files: args
                    .new_files
                    .into_iter()
                    .map(|f| tc::apply_file_diffs::NewFile {
                        file_path: f.file_path,
                        content: f.content,
                    })
                    .collect(),
                deleted_files: args
                    .deleted_files
                    .into_iter()
                    .map(|f| tc::apply_file_diffs::DeleteFile {
                        file_path: f.file_path,
                    })
                    .collect(),
                ..Default::default()
            })
        }
        "run_shell_command" => {
            #[derive(Deserialize)]
            struct Args {
                command: String,
                #[serde(default)]
                is_read_only: bool,
            }
            let args: Args = serde_json::from_value(input)?;
            tc::Tool::RunShellCommand(tc::RunShellCommand {
                command: args.command,
                #[allow(deprecated)]
                is_read_only: args.is_read_only,
                ..Default::default()
            })
        }
        "grep" => {
            #[derive(Deserialize)]
            struct Args {
                queries: Vec<String>,
                #[serde(default)]
                path: String,
            }
            let args: Args = serde_json::from_value(input)?;
            tc::Tool::Grep(tc::Grep {
                queries: args.queries,
                path: args.path,
            })
        }
        "file_glob" => {
            #[derive(Deserialize)]
            struct Args {
                patterns: Vec<String>,
                #[serde(default)]
                search_dir: String,
                #[serde(default)]
                max_matches: i32,
                #[serde(default)]
                max_depth: i32,
                #[serde(default)]
                min_depth: i32,
            }
            let args: Args = serde_json::from_value(input)?;
            tc::Tool::FileGlobV2(tc::FileGlobV2 {
                patterns: args.patterns,
                search_dir: args.search_dir,
                max_matches: args.max_matches,
                max_depth: args.max_depth,
                min_depth: args.min_depth,
            })
        }
        other => return Err(anyhow!("unknown tool: {other}")),
    };

    Ok(api::message::ToolCall {
        tool_call_id,
        tool: Some(tool),
    })
}

/// The provider-side `name` corresponding to the Warp tool variant inside `tool_call`.
/// Used when reconstructing tool-use blocks from history (so the next request can quote
/// the original `name` to the model).
pub fn tool_call_to_anthropic_name(tool: &api::message::tool_call::Tool) -> Option<&'static str> {
    use api::message::tool_call::Tool;
    Some(match tool {
        Tool::ReadFiles(_) => "read_files",
        Tool::ApplyFileDiffs(_) => "apply_file_diffs",
        Tool::RunShellCommand(_) => "run_shell_command",
        Tool::Grep(_) => "grep",
        Tool::FileGlobV2(_) => "file_glob",
        _ => return None,
    })
}

/// Reconstruct the original tool-use input JSON for a Warp `ToolCall`. Used when we feed
/// conversation history back to the provider on a subsequent request.
pub fn tool_call_to_anthropic_input(tool: &api::message::tool_call::Tool) -> Value {
    use api::message::tool_call::Tool;
    match tool {
        Tool::ReadFiles(t) => json!({
            "files": t.files.iter().map(|f| {
                json!({
                    "name": f.name,
                    "line_ranges": f.line_ranges.iter().map(|r| json!({
                        "start": r.start,
                        "end": r.end,
                    })).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>()
        }),
        Tool::ApplyFileDiffs(t) => json!({
            "summary": t.summary,
            "diffs": t.diffs.iter().map(|d| json!({
                "file_path": d.file_path,
                "search": d.search,
                "replace": d.replace,
            })).collect::<Vec<_>>(),
            "new_files": t.new_files.iter().map(|f| json!({
                "file_path": f.file_path,
                "content": f.content,
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
            "patterns": t.patterns,
            "search_dir": t.search_dir,
            "max_matches": t.max_matches,
            "max_depth": t.max_depth,
            "min_depth": t.min_depth,
        }),
        _ => json!({}),
    }
}

/// View into a tool-call result that bridges the two distinct proto types Warp uses for
/// the same data: one inside `Message::ToolCallResult` (history) and one inside
/// `Request::Input::UserInputs::UserInput::ToolCallResult` (inbound). Their `oneof result`
/// variants wrap identical top-level result types, so we can normalize them here.
pub enum ResultView<'a> {
    ReadFiles(&'a api::ReadFilesResult),
    ApplyFileDiffs(&'a api::ApplyFileDiffsResult),
    RunShellCommand(&'a api::RunShellCommandResult),
    Grep(&'a api::GrepResult),
    FileGlobV2(&'a api::FileGlobV2Result),
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
            _ => ResultView::Other,
        }
    }
}

/// Render the inner [`ResultView`] as text the provider can read back as a tool-result
/// content block. The boolean is `true` if the call should be marked `is_error` to the
/// model.
pub fn render_result(view: ResultView<'_>) -> (String, bool) {
    match view {
        ResultView::ReadFiles(rf) => match rf.result.as_ref() {
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
            Some(api::read_files_result::Result::Error(e)) => {
                (format!("ERROR: {}", e.message), true)
            }
            None => ("(empty result)".into(), true),
        },
        ResultView::ApplyFileDiffs(ad) => match ad.result.as_ref() {
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
            Some(api::apply_file_diffs_result::Result::Error(e)) => {
                (format!("ERROR: {}", e.message), true)
            }
            None => ("(empty result)".into(), true),
        },
        ResultView::RunShellCommand(rs) => match rs.result.as_ref() {
            Some(api::run_shell_command_result::Result::CommandFinished(f)) => {
                let body = if f.output.is_empty() {
                    "(no output)".to_string()
                } else {
                    f.output.clone()
                };
                (
                    format!("exit_code={}\n{}", f.exit_code, body),
                    f.exit_code != 0,
                )
            }
            Some(api::run_shell_command_result::Result::LongRunningCommandSnapshot(_)) => (
                "(command is still running; check back later)".into(),
                false,
            ),
            Some(api::run_shell_command_result::Result::PermissionDenied(_)) => {
                ("ERROR: permission denied to run this command".into(), true)
            }
            None => ("(no result)".into(), true),
        },
        ResultView::Grep(gr) => match gr.result.as_ref() {
            Some(api::grep_result::Result::Success(s)) => {
                if s.matched_files.is_empty() {
                    ("(no matches)".into(), false)
                } else {
                    let mut out = String::new();
                    for f in &s.matched_files {
                        let lines: Vec<String> = f
                            .matched_lines
                            .iter()
                            .map(|l| l.line_number.to_string())
                            .collect();
                        out.push_str(&format!("{}: lines {}\n", f.file_path, lines.join(", ")));
                    }
                    (out, false)
                }
            }
            Some(api::grep_result::Result::Error(e)) => {
                (format!("ERROR: {}", e.message), true)
            }
            None => ("(empty result)".into(), true),
        },
        ResultView::FileGlobV2(fg) => match fg.result.as_ref() {
            Some(api::file_glob_v2_result::Result::Success(s)) => {
                if s.matched_files.is_empty() {
                    ("(no matches)".into(), false)
                } else {
                    let body = s
                        .matched_files
                        .iter()
                        .map(|f| f.file_path.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    (body, false)
                }
            }
            Some(api::file_glob_v2_result::Result::Error(e)) => {
                (format!("ERROR: {}", e.message), true)
            }
            None => ("(empty result)".into(), true),
        },
        ResultView::Other => (
            "(tool result type not yet wired by open-warp-proxy)".into(),
            false,
        ),
    }
}

/// Convenience: render a `message::ToolCallResult` directly.
pub fn render_message_result(r: &api::message::ToolCallResult) -> (String, bool) {
    match r.result.as_ref() {
        Some(inner) => render_result(inner.into()),
        None => ("(no tool result)".into(), true),
    }
}

/// Convenience: render a `request::input::ToolCallResult` directly.
pub fn render_input_result(r: &api::request::input::ToolCallResult) -> (String, bool) {
    match r.result.as_ref() {
        Some(inner) => render_result(inner.into()),
        None => ("(no tool result)".into(), true),
    }
}

/// A short system prompt describing the tools and the expected behavior. Anthropic's tools
/// API also supplies the schemas, but a system prompt nudges the model toward the right
/// patterns (search → read → edit, prefer apply_file_diffs over shell write, etc.).
pub fn system_prompt() -> String {
    "You are open-warp's coding assistant, a self-hosted clone of Warp's Agent Mode. \
     You have access to a set of tools for reading, searching, running, and editing files.\n\
     \n\
     Workflow guidelines:\n\
     - Use `file_glob` to locate files by name and `grep` to find references; chain results \
     into `read_files` for context before suggesting changes.\n\
     - Use `apply_file_diffs` for ALL file edits — never write files via `run_shell_command`.\n\
     - Mark `run_shell_command` `is_read_only=true` for inspection commands (ls, cat, git \
     status, ...) so they run without confirmation.\n\
     - Be concise. Don't explain unless the user asks; just do the work."
        .to_string()
}
