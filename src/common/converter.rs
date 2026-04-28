//! 公共协议转换工具
//!
//! Anthropic 和 OpenAI 兼容层共享的转换逻辑

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use crate::kiro::model::requests::conversation::Message;
use crate::kiro::model::requests::tool::{InputSchema, Tool, ToolResult, ToolSpecification};

/// Kiro API 工具名称最大长度限制
pub const TOOL_NAME_MAX_LEN: usize = 63;

/// 追加到 Write 工具 description 末尾的内容
pub const WRITE_TOOL_DESCRIPTION_SUFFIX: &str = "- IMPORTANT: If the content to write exceeds 150 lines, you MUST only write the first 50 lines using this tool, then use `Edit` tool to append the remaining content in chunks of no more than 50 lines each. If needed, leave a unique placeholder to help append content. Do NOT attempt to write all content at once.";

/// 追加到 Edit 工具 description 末尾的内容
pub const EDIT_TOOL_DESCRIPTION_SUFFIX: &str = "- IMPORTANT: If the `new_string` content exceeds 50 lines, you MUST split it into multiple Edit calls, each replacing no more than 50 lines at a time. If used to append content, leave a unique placeholder to help append content. On the final chunk, do NOT include the placeholder.";

/// 追加到系统提示词的分块写入策略
pub const SYSTEM_CHUNKED_POLICY: &str = "\
When the Write or Edit tool has content size limits, always comply silently. \
Never suggest bypassing these limits via alternative tools. \
Never ask the user whether to switch approaches. \
Complete all chunked operations without commentary.";

/// 转换结果
#[derive(Debug)]
pub struct ConversionResult {
    /// 转换后的 Kiro 请求
    pub conversation_state: crate::kiro::model::requests::conversation::ConversationState,
    /// 工具名称映射（短名称 → 原始名称），仅当存在超长工具名时非空
    pub tool_name_map: HashMap<String, String>,
}

/// 转换错误
#[derive(Debug)]
pub enum ConversionError {
    UnsupportedModel(String),
    EmptyMessages,
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConversionError::UnsupportedModel(model) => write!(f, "模型不支持: {}", model),
            ConversionError::EmptyMessages => write!(f, "消息列表为空"),
        }
    }
}

impl std::error::Error for ConversionError {}

/// 模型映射：将 Anthropic/OpenAI 模型名映射到 Kiro 模型 ID
pub fn map_model(model: &str) -> Option<String> {
    let model_lower = model.to_lowercase();

    if model_lower.contains("sonnet") {
        if model_lower.contains("4-6") || model_lower.contains("4.6") {
            Some("claude-sonnet-4.6".to_string())
        } else {
            Some("claude-sonnet-4.5".to_string())
        }
    } else if model_lower.contains("opus") {
        if model_lower.contains("4-5") || model_lower.contains("4.5") {
            Some("claude-opus-4.5".to_string())
        } else {
            Some("claude-opus-4.6".to_string())
        }
    } else if model_lower.contains("haiku") {
        Some("claude-haiku-4.5".to_string())
    } else {
        None
    }
}

/// 根据模型名称返回对应的上下文窗口大小
pub fn get_context_window_size(model: &str) -> i32 {
    match map_model(model) {
        Some(mapped) if mapped == "claude-sonnet-4.6" || mapped == "claude-opus-4.6" => 1_000_000,
        _ => 200_000,
    }
}

/// 规范化 JSON Schema，修复 MCP 工具定义中常见的类型问题
pub fn normalize_json_schema(schema: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(mut obj) = schema else {
        return serde_json::json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": true
        });
    };

    if obj
        .get("type")
        .and_then(|v| v.as_str())
        .is_none_or(|s| s.is_empty())
    {
        obj.insert(
            "type".to_string(),
            serde_json::Value::String("object".to_string()),
        );
    }

    match obj.get("properties") {
        Some(serde_json::Value::Object(_)) => {}
        _ => {
            obj.insert(
                "properties".to_string(),
                serde_json::Value::Object(serde_json::Map::new()),
            );
        }
    }

    let required = match obj.remove("required") {
        Some(serde_json::Value::Array(arr)) => serde_json::Value::Array(
            arr.into_iter()
                .filter_map(|v| v.as_str().map(|s| serde_json::Value::String(s.to_string())))
                .collect(),
        ),
        _ => serde_json::Value::Array(Vec::new()),
    };
    obj.insert("required".to_string(), required);

    match obj.get("additionalProperties") {
        Some(serde_json::Value::Bool(_)) | Some(serde_json::Value::Object(_)) => {}
        _ => {
            obj.insert(
                "additionalProperties".to_string(),
                serde_json::Value::Bool(true),
            );
        }
    }

    serde_json::Value::Object(obj)
}

/// 生成确定性短名称：截断前缀 + "_" + 8 位 SHA256 hex
pub fn shorten_tool_name(name: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    let hash_hex = format!("{:x}", hasher.finalize());
    let hash_suffix = &hash_hex[..8];
    let prefix_max = TOOL_NAME_MAX_LEN - 1 - 8;
    let prefix = match name.char_indices().nth(prefix_max) {
        Some((idx, _)) => &name[..idx],
        None => name,
    };
    format!("{}_{}", prefix, hash_suffix)
}

/// 如果名称超长则缩短，并记录映射（short → original）
pub fn map_tool_name(name: &str, tool_name_map: &mut HashMap<String, String>) -> String {
    if name.len() <= TOOL_NAME_MAX_LEN {
        return name.to_string();
    }
    let short = shorten_tool_name(name);
    tool_name_map.insert(short.clone(), name.to_string());
    short
}

/// 收集历史消息中使用的所有工具名称
pub fn collect_history_tool_names(history: &[Message]) -> Vec<String> {
    let mut tool_names = Vec::new();

    for msg in history {
        if let Message::Assistant(assistant_msg) = msg
            && let Some(ref tool_uses) = assistant_msg.assistant_response_message.tool_uses
        {
            for tool_use in tool_uses {
                if !tool_names.contains(&tool_use.name) {
                    tool_names.push(tool_use.name.clone());
                }
            }
        }
    }

    tool_names
}

/// 为历史中使用但不在 tools 列表中的工具创建占位符定义
pub fn create_placeholder_tool(name: &str) -> Tool {
    Tool {
        tool_specification: ToolSpecification {
            name: name.to_string(),
            description: "Tool used in conversation history".to_string(),
            input_schema: InputSchema::from_json(serde_json::json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": true
            })),
        },
    }
}

/// 验证并过滤 tool_use/tool_result 配对
pub fn validate_tool_pairing(
    history: &[Message],
    tool_results: &[ToolResult],
) -> (Vec<ToolResult>, std::collections::HashSet<String>) {
    use std::collections::HashSet;

    let mut all_tool_use_ids: HashSet<String> = HashSet::new();
    let mut history_tool_result_ids: HashSet<String> = HashSet::new();

    for msg in history {
        match msg {
            Message::Assistant(assistant_msg) => {
                if let Some(ref tool_uses) = assistant_msg.assistant_response_message.tool_uses {
                    for tool_use in tool_uses {
                        all_tool_use_ids.insert(tool_use.tool_use_id.clone());
                    }
                }
            }
            Message::User(user_msg) => {
                for result in &user_msg
                    .user_input_message
                    .user_input_message_context
                    .tool_results
                {
                    history_tool_result_ids.insert(result.tool_use_id.clone());
                }
            }
        }
    }

    let mut unpaired_tool_use_ids: HashSet<String> = all_tool_use_ids
        .difference(&history_tool_result_ids)
        .cloned()
        .collect();

    let mut filtered_results = Vec::new();

    for result in tool_results {
        if unpaired_tool_use_ids.contains(&result.tool_use_id) {
            filtered_results.push(result.clone());
            unpaired_tool_use_ids.remove(&result.tool_use_id);
        } else if all_tool_use_ids.contains(&result.tool_use_id) {
            tracing::warn!(
                "跳过重复的 tool_result：该 tool_use 已在历史中配对，tool_use_id={}",
                result.tool_use_id
            );
        } else {
            tracing::warn!(
                "跳过孤立的 tool_result：找不到对应的 tool_use，tool_use_id={}",
                result.tool_use_id
            );
        }
    }

    for orphaned_id in &unpaired_tool_use_ids {
        tracing::warn!(
            "检测到孤立的 tool_use：找不到对应的 tool_result，将从历史中移除，tool_use_id={}",
            orphaned_id
        );
    }

    (filtered_results, unpaired_tool_use_ids)
}

/// 从历史消息中移除孤立的 tool_use
pub fn remove_orphaned_tool_uses(
    history: &mut [Message],
    orphaned_ids: &std::collections::HashSet<String>,
) {
    if orphaned_ids.is_empty() {
        return;
    }

    for msg in history.iter_mut() {
        if let Message::Assistant(assistant_msg) = msg
            && let Some(ref mut tool_uses) = assistant_msg.assistant_response_message.tool_uses
        {
            let original_len = tool_uses.len();
            tool_uses.retain(|tu| !orphaned_ids.contains(&tu.tool_use_id));

            if tool_uses.is_empty() {
                assistant_msg.assistant_response_message.tool_uses = None;
            } else if tool_uses.len() != original_len {
                tracing::debug!(
                    "从 assistant 消息中移除了 {} 个孤立的 tool_use",
                    original_len - tool_uses.len()
                );
            }
        }
    }
}

/// 确保历史中使用的工具都在 tools 列表中有定义
pub fn ensure_history_tools_in_list(history: &[Message], tools: &mut Vec<Tool>) {
    let history_tool_names = collect_history_tool_names(history);
    let existing_tool_names: std::collections::HashSet<_> = tools
        .iter()
        .map(|t| t.tool_specification.name.to_lowercase())
        .collect();

    for tool_name in history_tool_names {
        if !existing_tool_names.contains(&tool_name.to_lowercase()) {
            tools.push(create_placeholder_tool(&tool_name));
        }
    }
}
