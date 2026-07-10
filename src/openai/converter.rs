//! OpenAI → Kiro 协议转换器
//!
//! 负责将 OpenAI Chat Completions API 请求格式转换为 Kiro API 请求格式

use std::collections::HashMap;

use uuid::Uuid;

use crate::common::converter as shared;
pub use crate::common::converter::{ConversionError, ConversionResult};
use crate::kiro::model::requests::conversation::{
    AssistantMessage, ConversationState, CurrentMessage, HistoryAssistantMessage,
    HistoryUserMessage, KiroImage, Message, UserInputMessage, UserInputMessageContext, UserMessage,
};
use crate::kiro::model::requests::tool::{
    InputSchema, Tool, ToolResult, ToolSpecification, ToolUseEntry,
};

use super::types::{
    ChatCompletionRequest, ChatMessage, FunctionCall, ResponsesInput, ResponsesRequest, ToolCall,
};

/// 将 OpenAI Chat Completions 请求转换为 Kiro 请求
pub fn convert_request(req: &ChatCompletionRequest) -> Result<ConversionResult, ConversionError> {
    // 1. 映射模型
    let model_id = shared::map_model(&req.model)
        .ok_or_else(|| ConversionError::UnsupportedModel(req.model.clone()))?;

    // 2. 检查消息列表
    if req.messages.is_empty() {
        return Err(ConversionError::EmptyMessages);
    }

    // 3. 分离 system 消息和对话消息
    let (system_messages, conversation_messages) = split_messages(&req.messages);

    // 4. 检查对话消息不为空
    if conversation_messages.is_empty() {
        return Err(ConversionError::EmptyMessages);
    }

    // 5. 预处理：如果末尾不是 user/tool，截断到最后一条 user/tool
    let messages = if let Some(last) = conversation_messages.last() {
        if last.role != "user" && last.role != "tool" {
            tracing::info!("检测到末尾非 user/tool 消息，静默丢弃");
            let last_user_idx = conversation_messages
                .iter()
                .rposition(|m| m.role == "user" || m.role == "tool")
                .ok_or(ConversionError::EmptyMessages)?;
            &conversation_messages[..=last_user_idx]
        } else {
            &conversation_messages[..]
        }
    } else {
        return Err(ConversionError::EmptyMessages);
    };

    // 6. 生成会话 ID
    let conversation_id = req
        .user
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let agent_continuation_id = Uuid::new_v4().to_string();

    // 7. 处理最后一条消息作为 current_message
    let last_message = messages.last().unwrap();
    let (text_content, images, tool_results) = process_message_content(last_message);

    // 8. 转换工具定义
    let mut tool_name_map = HashMap::new();
    let mut tools = convert_tools(&req.tools, &mut tool_name_map);

    // 9. 构建历史消息
    let mut history =
        build_history(&system_messages, messages, &model_id, &req.model, &mut tool_name_map)?;

    // 10. 验证并过滤 tool_use/tool_result 配对
    let (validated_tool_results, orphaned_tool_use_ids) =
        shared::validate_tool_pairing(&history, &tool_results);

    // 11. 从历史中移除孤立的 tool_use
    shared::remove_orphaned_tool_uses(&mut history, &orphaned_tool_use_ids);

    // 12. 确保历史中使用的工具都在 tools 列表中
    shared::ensure_history_tools_in_list(&history, &mut tools);

    // 13. 构建 UserInputMessageContext
    let mut context = UserInputMessageContext::new();
    if !tools.is_empty() {
        context = context.with_tools(tools);
    }
    if !validated_tool_results.is_empty() {
        context = context.with_tool_results(validated_tool_results);
    }

    // 14. 构建当前消息
    let mut user_input = UserInputMessage::new(text_content, &model_id)
        .with_context(context)
        .with_origin("AI_EDITOR");

    if !images.is_empty() {
        user_input = user_input.with_images(images);
    }

    let current_message = CurrentMessage::new(user_input);

    // 15. 构建 ConversationState
    let conversation_state = ConversationState::new(conversation_id)
        .with_agent_continuation_id(agent_continuation_id)
        .with_agent_task_type("vibe")
        .with_chat_trigger_type("MANUAL")
        .with_current_message(current_message)
        .with_history(history);

    if !tool_name_map.is_empty() {
        tracing::info!(
            "工具名称映射: {} 个超长名称已缩短",
            tool_name_map.len()
        );
    }

    Ok(ConversionResult {
        conversation_state,
        tool_name_map,
    })
}

/// 将 OpenAI Responses 请求归一化为 Chat Completions 请求，再复用 `convert_request`
///
/// 目前尚未被 HTTP handler 调用（Responses 端点在 Task 7/8 中实现），暂时允许未使用。
#[allow(dead_code)]
pub fn convert_responses_request(
    req: &ResponsesRequest,
) -> Result<ConversionResult, ConversionError> {
    let chat = normalize_responses_to_chat(req)?;
    convert_request(&chat)
}

/// 将 Responses API 请求归一化为 Chat Completions 消息列表
///
/// 归一化规则：
/// - `instructions` → 前置一条 system 消息
/// - `input` 为纯字符串 → 一条 user 消息
/// - `input` 为条目数组：
///   - `message` 条目：保留 role（user/assistant），从 content parts
///     （`input_text`/`output_text`/`text`）中提取文本
///   - `function_call` 条目 → assistant 消息 + tool_calls
///   - `function_call_output` 条目 → tool 消息，`tool_call_id` 取自 `call_id`
///   - 未知类型条目：记录 warn 日志并跳过
#[allow(dead_code)]
pub fn normalize_responses_to_chat(
    req: &ResponsesRequest,
) -> Result<ChatCompletionRequest, ConversionError> {
    let mut messages = Vec::new();

    if let Some(instructions) = req.instructions.as_ref().filter(|s| !s.is_empty()) {
        messages.push(ChatMessage {
            role: "system".into(),
            content: Some(serde_json::json!(instructions)),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    match &req.input {
        ResponsesInput::Text(text) => {
            messages.push(ChatMessage {
                role: "user".into(),
                content: Some(serde_json::json!(text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                match convert_responses_item(item) {
                    Some(msg) => messages.push(msg),
                    None => tracing::warn!("跳过无法识别的 Responses input item: {}", item),
                }
            }
        }
    }

    Ok(ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        stream: req.stream,
        temperature: None,
        top_p: None,
        max_tokens: None,
        max_completion_tokens: None,
        stop: None,
        tools: req.tools.clone(),
        tool_choice: None,
        stream_options: None,
        user: req.user.clone(),
    })
}

/// 转换单个 Responses `input` 条目为 Chat 消息
///
/// 返回 `None` 表示条目类型未知或缺少必需字段，调用方负责记录日志并跳过。
fn convert_responses_item(item: &serde_json::Value) -> Option<ChatMessage> {
    let item_type = item
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("message");

    match item_type {
        "message" => {
            let role = item.get("role").and_then(|v| v.as_str())?.to_string();
            let text = extract_responses_content_text(item.get("content"));
            Some(ChatMessage {
                role,
                content: Some(serde_json::json!(text)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            })
        }
        "function_call" => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let arguments = item
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();

            Some(ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    index: Some(0),
                    id: Some(call_id),
                    call_type: Some("function".into()),
                    function: FunctionCall {
                        name: Some(name),
                        arguments,
                    },
                }]),
                tool_call_id: None,
                name: None,
            })
        }
        "function_call_output" => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let output = match item.get("output") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => String::new(),
            };
            Some(ChatMessage {
                role: "tool".into(),
                content: Some(serde_json::json!(output)),
                tool_calls: None,
                tool_call_id: Some(call_id),
                name: None,
            })
        }
        _ => None,
    }
}

/// 从 Responses content parts 中提取文本（支持 input_text/output_text/text）
fn extract_responses_content_text(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                let part_type = part.get("type").and_then(|v| v.as_str())?;
                if matches!(part_type, "input_text" | "output_text" | "text") {
                    part.get("text")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// 分离 system/developer 消息和对话消息
fn split_messages(messages: &[ChatMessage]) -> (Vec<&ChatMessage>, Vec<&ChatMessage>) {
    let mut system_msgs = Vec::new();
    let mut conversation_msgs = Vec::new();

    for msg in messages {
        if msg.role == "system" || msg.role == "developer" {
            system_msgs.push(msg);
        } else {
            conversation_msgs.push(msg);
        }
    }

    (system_msgs, conversation_msgs)
}

/// 处理消息内容，提取文本、图片和工具结果
fn process_message_content(msg: &ChatMessage) -> (String, Vec<KiroImage>, Vec<ToolResult>) {
    let images = Vec::new();

    // tool 角色消息 → tool_result
    if msg.role == "tool" {
        let tool_call_id = msg.tool_call_id.clone().unwrap_or_default();
        let content = msg.text_content();
        let result = ToolResult::success(&tool_call_id, content);
        return (String::new(), images, vec![result]);
    }

    // user 消息
    let text = msg.text_content();
    (text, images, Vec::new())
}

/// 转换工具定义（OpenAI function → Kiro ToolSpecification）
fn convert_tools(
    tools: &Option<Vec<super::types::ToolDefinition>>,
    tool_name_map: &mut HashMap<String, String>,
) -> Vec<Tool> {
    let Some(tools) = tools else {
        return Vec::new();
    };

    tools
        .iter()
        .map(|t| {
            let mut description = t.function.description.clone().unwrap_or_default();

            // 对 Write/Edit 工具追加自定义描述后缀
            let suffix = match t.function.name.as_str() {
                "Write" => shared::WRITE_TOOL_DESCRIPTION_SUFFIX,
                "Edit" => shared::EDIT_TOOL_DESCRIPTION_SUFFIX,
                _ => "",
            };
            if !suffix.is_empty() {
                description.push('\n');
                description.push_str(suffix);
            }

            // 限制描述长度
            let description = match description.char_indices().nth(10000) {
                Some((idx, _)) => description[..idx].to_string(),
                None => description,
            };

            let schema = t
                .function
                .parameters
                .clone()
                .unwrap_or(serde_json::json!({"type": "object", "properties": {}}));

            Tool {
                tool_specification: ToolSpecification {
                    name: shared::map_tool_name(&t.function.name, tool_name_map),
                    description,
                    input_schema: InputSchema::from_json(shared::normalize_json_schema(schema)),
                },
            }
        })
        .collect()
}

/// 生成 thinking 标签前缀（当模型名包含 `-thinking` 时）
///
/// 与 Anthropic 侧不同：OpenAI Chat Completions 协议没有显式的 `thinking` 请求字段，
/// 因此改用模型名后缀 `-thinking`（大小写不敏感）作为开关。
fn generate_thinking_prefix_for_model(model: &str) -> Option<String> {
    if super::thinking::is_thinking_model(model) {
        Some(
            "<thinking_mode>enabled</thinking_mode><max_thinking_length>20000</max_thinking_length>"
                .to_string(),
        )
    } else {
        None
    }
}

/// 检查内容是否已包含 thinking 标签配置
fn has_thinking_tags(content: &str) -> bool {
    content.contains("<thinking_mode>") || content.contains("<max_thinking_length>")
}

/// 构建历史消息
fn build_history(
    system_messages: &[&ChatMessage],
    messages: &[&ChatMessage],
    model_id: &str,
    raw_model: &str,
    tool_name_map: &mut HashMap<String, String>,
) -> Result<Vec<Message>, ConversionError> {
    let mut history = Vec::new();

    // 生成 thinking 前缀（如果模型名包含 -thinking）
    let thinking_prefix = generate_thinking_prefix_for_model(raw_model);

    // 1. 处理系统消息
    if !system_messages.is_empty() {
        let system_content: String = system_messages
            .iter()
            .map(|m| m.text_content())
            .collect::<Vec<_>>()
            .join("\n");

        if !system_content.is_empty() {
            let system_content = format!("{}\n{}", system_content, shared::SYSTEM_CHUNKED_POLICY);

            // 注入 thinking 标签到系统消息最前面（如果需要且不存在）
            let final_content = if let Some(ref prefix) = thinking_prefix {
                if !has_thinking_tags(&system_content) {
                    format!("{}\n{}", prefix, system_content)
                } else {
                    system_content
                }
            } else {
                system_content
            };

            let user_msg = HistoryUserMessage::new(final_content, model_id);
            history.push(Message::User(user_msg));

            let assistant_msg = HistoryAssistantMessage::new("I will follow these instructions.");
            history.push(Message::Assistant(assistant_msg));
        }
    } else if let Some(ref prefix) = thinking_prefix {
        // 没有系统消息但需要注入 thinking 配置，插入新的系统消息
        let user_msg = HistoryUserMessage::new(prefix.clone(), model_id);
        history.push(Message::User(user_msg));

        let assistant_msg = HistoryAssistantMessage::new("I will follow these instructions.");
        history.push(Message::Assistant(assistant_msg));
    }

    // 2. 处理对话历史（最后一条作为 currentMessage，不加入历史）
    let history_end = messages.len().saturating_sub(1);

    let mut user_buffer: Vec<&ChatMessage> = Vec::new();
    let mut assistant_buffer: Vec<&ChatMessage> = Vec::new();
    // tool 消息紧跟在 assistant 之后，需要和下一个 user 消息合并
    let mut tool_buffer: Vec<&ChatMessage> = Vec::new();

    for msg in &messages[..history_end] {

        match msg.role.as_str() {
            "user" => {
                // 先处理累积的 assistant 消息
                if !assistant_buffer.is_empty() {
                    let merged = merge_assistant_messages(&assistant_buffer, tool_name_map)?;
                    history.push(Message::Assistant(merged));
                    assistant_buffer.clear();
                }

                // 如果有 tool 结果缓冲，和 user 消息一起处理
                if !tool_buffer.is_empty() {
                    let merged_user =
                        merge_user_with_tool_results(&tool_buffer, msg, model_id)?;
                    history.push(Message::User(merged_user));
                    tool_buffer.clear();
                } else {
                    user_buffer.push(msg);
                }
            }
            "assistant" => {
                // 先处理累积的 user 消息
                if !user_buffer.is_empty() {
                    let merged_user = merge_user_messages(&user_buffer, model_id)?;
                    history.push(Message::User(merged_user));
                    user_buffer.clear();
                }
                // 先处理累积的 tool 消息（作为独立的 user 消息）
                if !tool_buffer.is_empty() {
                    let tool_user = tool_results_to_user_message(&tool_buffer, model_id)?;
                    history.push(Message::User(tool_user));
                    tool_buffer.clear();
                }
                assistant_buffer.push(msg);
            }
            "tool" => {
                // 先处理累积的 user 消息
                if !user_buffer.is_empty() {
                    let merged_user = merge_user_messages(&user_buffer, model_id)?;
                    history.push(Message::User(merged_user));
                    user_buffer.clear();
                }
                // 先处理累积的 assistant 消息
                if !assistant_buffer.is_empty() {
                    let merged = merge_assistant_messages(&assistant_buffer, tool_name_map)?;
                    history.push(Message::Assistant(merged));
                    assistant_buffer.clear();
                }
                tool_buffer.push(msg);
            }
            _ => {}
        }
    }

    // 处理末尾累积的 assistant 消息
    if !assistant_buffer.is_empty() {
        let merged = merge_assistant_messages(&assistant_buffer, tool_name_map)?;
        history.push(Message::Assistant(merged));
    }

    // 处理末尾累积的 tool 消息
    if !tool_buffer.is_empty() {
        let tool_user = tool_results_to_user_message(&tool_buffer, model_id)?;
        history.push(Message::User(tool_user));
    }

    // 处理结尾的孤立 user 消息
    if !user_buffer.is_empty() {
        let merged_user = merge_user_messages(&user_buffer, model_id)?;
        history.push(Message::User(merged_user));

        let auto_assistant = HistoryAssistantMessage::new("OK");
        history.push(Message::Assistant(auto_assistant));
    }

    Ok(history)
}

/// 合并多个 user 消息
fn merge_user_messages(
    messages: &[&ChatMessage],
    model_id: &str,
) -> Result<HistoryUserMessage, ConversionError> {
    let content: String = messages
        .iter()
        .map(|m| m.text_content())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    Ok(HistoryUserMessage {
        user_input_message: UserMessage::new(&content, model_id),
    })
}

/// 将 tool 结果消息转换为 Kiro user 消息
fn tool_results_to_user_message(
    tool_messages: &[&ChatMessage],
    model_id: &str,
) -> Result<HistoryUserMessage, ConversionError> {
    let mut tool_results = Vec::new();
    for msg in tool_messages {
        let tool_call_id = msg.tool_call_id.clone().unwrap_or_default();
        let content = msg.text_content();
        tool_results.push(ToolResult::success(&tool_call_id, content));
    }

    let mut user_msg = UserMessage::new("", model_id);
    if !tool_results.is_empty() {
        let ctx = UserInputMessageContext::new().with_tool_results(tool_results);
        user_msg = user_msg.with_context(ctx);
    }

    Ok(HistoryUserMessage {
        user_input_message: user_msg,
    })
}

/// 合并 tool 结果和 user 消息
fn merge_user_with_tool_results(
    tool_messages: &[&ChatMessage],
    user_msg: &ChatMessage,
    model_id: &str,
) -> Result<HistoryUserMessage, ConversionError> {
    let mut tool_results = Vec::new();
    for msg in tool_messages {
        let tool_call_id = msg.tool_call_id.clone().unwrap_or_default();
        let content = msg.text_content();
        tool_results.push(ToolResult::success(&tool_call_id, content));
    }

    let text = user_msg.text_content();
    let mut user = UserMessage::new(&text, model_id);
    if !tool_results.is_empty() {
        let ctx = UserInputMessageContext::new().with_tool_results(tool_results);
        user = user.with_context(ctx);
    }

    Ok(HistoryUserMessage {
        user_input_message: user,
    })
}

/// 转换 assistant 消息
fn convert_assistant_message(
    msg: &ChatMessage,
    tool_name_map: &mut HashMap<String, String>,
) -> Result<HistoryAssistantMessage, ConversionError> {
    let text_content = msg.text_content();
    let mut tool_uses = Vec::new();

    // 处理 tool_calls
    if let Some(ref calls) = msg.tool_calls {
        for call in calls {
            let id = call.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
            let name = call.function.name.clone().unwrap_or_default();
            let mapped_name = shared::map_tool_name(&name, tool_name_map);

            let input: serde_json::Value = if call.function.arguments.is_empty() {
                serde_json::json!({})
            } else {
                serde_json::from_str(&call.function.arguments).unwrap_or_else(|e| {
                    tracing::warn!("工具输入 JSON 解析失败: {}, id: {}", e, id);
                    serde_json::json!({})
                })
            };

            tool_uses.push(ToolUseEntry::new(id, mapped_name).with_input(input));
        }
    }

    // Kiro API 要求 content 不能为空
    let final_content = if text_content.is_empty() && !tool_uses.is_empty() {
        " ".to_string()
    } else {
        text_content
    };

    let mut assistant = AssistantMessage::new(final_content);
    if !tool_uses.is_empty() {
        assistant = assistant.with_tool_uses(tool_uses);
    }

    Ok(HistoryAssistantMessage {
        assistant_response_message: assistant,
    })
}

/// 合并多个连续的 assistant 消息
fn merge_assistant_messages(
    messages: &[&ChatMessage],
    tool_name_map: &mut HashMap<String, String>,
) -> Result<HistoryAssistantMessage, ConversionError> {
    assert!(!messages.is_empty());
    if messages.len() == 1 {
        return convert_assistant_message(messages[0], tool_name_map);
    }

    let mut all_tool_uses: Vec<ToolUseEntry> = Vec::new();
    let mut content_parts: Vec<String> = Vec::new();

    for msg in messages {
        let converted = convert_assistant_message(msg, tool_name_map)?;
        let am = converted.assistant_response_message;
        if !am.content.trim().is_empty() {
            content_parts.push(am.content);
        }
        if let Some(tus) = am.tool_uses {
            all_tool_uses.extend(tus);
        }
    }

    let content = if content_parts.is_empty() && !all_tool_uses.is_empty() {
        " ".to_string()
    } else {
        content_parts.join("\n\n")
    };

    let mut assistant = AssistantMessage::new(content);
    if !all_tool_uses.is_empty() {
        assistant = assistant.with_tool_uses(all_tool_uses);
    }
    Ok(HistoryAssistantMessage {
        assistant_response_message: assistant,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::types::{
        ChatMessage, FunctionCall, FunctionDefinition, ResponsesInput, ResponsesRequest, ToolCall,
        ToolDefinition,
    };

    fn make_request(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "claude-sonnet-4-6".into(),
            messages,
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: Some(1024),
            max_completion_tokens: None,
            stop: None,
            tools: None,
            tool_choice: None,
            stream_options: None,
            user: None,
        }
    }

    #[test]
    fn test_basic_user_message() {
        let req = make_request(vec![ChatMessage {
            role: "user".into(),
            content: Some(serde_json::json!("Hello!")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]);

        let result = convert_request(&req).unwrap();
        assert_eq!(
            result.conversation_state.current_message.user_input_message.content,
            "Hello!"
        );
    }

    #[test]
    fn test_system_message_becomes_history_pair() {
        let req = make_request(vec![
            ChatMessage {
                role: "system".into(),
                content: Some(serde_json::json!("You are helpful.")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some(serde_json::json!("Hi")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ]);

        let result = convert_request(&req).unwrap();
        // system → user + assistant pair in history
        assert!(result.conversation_state.history.len() >= 2);
    }

    #[test]
    fn test_developer_role_treated_as_system() {
        let req = make_request(vec![
            ChatMessage {
                role: "developer".into(),
                content: Some(serde_json::json!("Instructions")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some(serde_json::json!("Hi")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ]);

        let result = convert_request(&req).unwrap();
        assert!(result.conversation_state.history.len() >= 2);
    }

    #[test]
    fn test_tool_calls_and_tool_results() {
        let mut req = make_request(vec![
            ChatMessage {
                role: "user".into(),
                content: Some(serde_json::json!("What's the weather?")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: None,
                tool_calls: Some(vec![ToolCall {
                    index: Some(0),
                    id: Some("call_abc".into()),
                    call_type: Some("function".into()),
                    function: FunctionCall {
                        name: Some("get_weather".into()),
                        arguments: r#"{"location":"Tokyo"}"#.into(),
                    },
                }]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(serde_json::json!("Sunny, 25°C")),
                tool_calls: None,
                tool_call_id: Some("call_abc".into()),
                name: None,
            },
        ]);

        req.tools = Some(vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: Some("Get weather".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}}
                })),
            },
        }]);

        let result = convert_request(&req).unwrap();
        // The tool result should be in the current message context
        let ctx = &result
            .conversation_state
            .current_message
            .user_input_message
            .user_input_message_context;
        assert!(!ctx.tool_results.is_empty());
    }

    #[test]
    fn test_empty_messages_error() {
        let req = make_request(vec![]);
        assert!(convert_request(&req).is_err());
    }

    #[test]
    fn test_unsupported_model_error() {
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(serde_json::json!("Hi")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            stream: false,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            stop: None,
            tools: None,
            tool_choice: None,
            stream_options: None,
            user: None,
        };
        assert!(matches!(
            convert_request(&req),
            Err(ConversionError::UnsupportedModel(_))
        ));
    }

    #[test]
    fn test_multi_turn_conversation() {
        let req = make_request(vec![
            ChatMessage {
                role: "user".into(),
                content: Some(serde_json::json!("Hello")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: Some(serde_json::json!("Hi there!")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some(serde_json::json!("How are you?")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ]);

        let result = convert_request(&req).unwrap();
        // History should have user + assistant pair
        assert!(result.conversation_state.history.len() >= 2);
        assert_eq!(
            result.conversation_state.current_message.user_input_message.content,
            "How are you?"
        );
    }

    #[test]
    fn test_thinking_model_injects_prefix_into_system_message() {
        let mut req = make_request(vec![
            ChatMessage {
                role: "system".into(),
                content: Some(serde_json::json!("You are helpful.")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some(serde_json::json!("Hi")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ]);
        req.model = "claude-sonnet-4-6-thinking".into();

        let result = convert_request(&req).unwrap();
        let history = &result.conversation_state.history;
        let first_user = history
            .iter()
            .find_map(|m| match m {
                crate::kiro::model::requests::conversation::Message::User(u) => {
                    Some(u.user_input_message.content.clone())
                }
                _ => None,
            })
            .expect("history should contain a user message");

        assert!(first_user.contains("<thinking_mode>enabled</thinking_mode>"));
        assert!(first_user.contains("<max_thinking_length>20000</max_thinking_length>"));
        assert!(first_user.contains("You are helpful."));
    }

    #[test]
    fn test_thinking_model_without_system_message_inserts_prefix_pair() {
        let mut req = make_request(vec![ChatMessage {
            role: "user".into(),
            content: Some(serde_json::json!("Hi")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]);
        req.model = "claude-opus-4-6-Thinking".into();

        let result = convert_request(&req).unwrap();
        let history = &result.conversation_state.history;
        let first_user = history
            .iter()
            .find_map(|m| match m {
                crate::kiro::model::requests::conversation::Message::User(u) => {
                    Some(u.user_input_message.content.clone())
                }
                _ => None,
            })
            .expect("history should contain a user message");

        assert!(first_user.contains("<thinking_mode>enabled</thinking_mode>"));
    }

    #[test]
    fn test_non_thinking_model_does_not_inject_prefix() {
        let req = make_request(vec![
            ChatMessage {
                role: "system".into(),
                content: Some(serde_json::json!("You are helpful.")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some(serde_json::json!("Hi")),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ]);

        let result = convert_request(&req).unwrap();
        let history = &result.conversation_state.history;
        let first_user = history
            .iter()
            .find_map(|m| match m {
                crate::kiro::model::requests::conversation::Message::User(u) => {
                    Some(u.user_input_message.content.clone())
                }
                _ => None,
            })
            .expect("history should contain a user message");

        assert!(!first_user.contains("<thinking_mode>"));
    }

    #[test]
    fn test_model_mapping() {
        let req = make_request(vec![ChatMessage {
            role: "user".into(),
            content: Some(serde_json::json!("Hi")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }]);

        let result = convert_request(&req).unwrap();
        // claude-sonnet-4-6 should map to claude-sonnet-4.6
        let model = &result
            .conversation_state
            .current_message
            .user_input_message
            .model_id;
        assert_eq!(model, "claude-sonnet-4.6");
    }

    // === Responses API 归一化测试 ===

    fn make_responses_request(input: ResponsesInput) -> ResponsesRequest {
        ResponsesRequest {
            model: "claude-sonnet-4-6".into(),
            input,
            instructions: None,
            stream: false,
            tools: None,
            user: None,
        }
    }

    #[test]
    fn test_normalize_text_input_with_instructions() {
        let mut req = make_responses_request(ResponsesInput::Text("Hi there".into()));
        req.instructions = Some("You are helpful.".into());

        let chat = normalize_responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(chat.messages[0].text_content(), "You are helpful.");
        assert_eq!(chat.messages[1].role, "user");
        assert_eq!(chat.messages[1].text_content(), "Hi there");
    }

    #[test]
    fn test_normalize_text_input_without_instructions() {
        let req = make_responses_request(ResponsesInput::Text("Hello!".into()));

        let chat = normalize_responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.messages[0].text_content(), "Hello!");
    }

    #[test]
    fn test_normalize_message_item_with_content_parts() {
        let items = vec![serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What's the weather?"}]
        })];
        let req = make_responses_request(ResponsesInput::Items(items));

        let chat = normalize_responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.messages[0].text_content(), "What's the weather?");
    }

    #[test]
    fn test_normalize_function_call_item_becomes_assistant_tool_calls() {
        let items = vec![serde_json::json!({
            "type": "function_call",
            "call_id": "call_abc",
            "name": "get_weather",
            "arguments": "{\"location\":\"Tokyo\"}"
        })];
        let req = make_responses_request(ResponsesInput::Items(items));

        let chat = normalize_responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        let tool_calls = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_abc"));
        assert_eq!(tool_calls[0].function.name.as_deref(), Some("get_weather"));
        assert_eq!(tool_calls[0].function.arguments, "{\"location\":\"Tokyo\"}");
    }

    #[test]
    fn test_normalize_function_call_output_item_becomes_tool_message() {
        let items = vec![serde_json::json!({
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": "Sunny, 25°C"
        })];
        let req = make_responses_request(ResponsesInput::Items(items));

        let chat = normalize_responses_to_chat(&req).unwrap();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "tool");
        assert_eq!(chat.messages[0].tool_call_id.as_deref(), Some("call_abc"));
        assert_eq!(chat.messages[0].text_content(), "Sunny, 25°C");
    }

    #[test]
    fn test_normalize_unknown_item_is_skipped() {
        let items = vec![
            serde_json::json!({"type": "reasoning", "id": "r1"}),
            serde_json::json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            }),
        ];
        let req = make_responses_request(ResponsesInput::Items(items));

        let chat = normalize_responses_to_chat(&req).unwrap();
        // 未知条目被跳过，只保留有效的 message 条目
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
    }

    #[test]
    fn test_normalize_full_conversation_with_tool_round_trip() {
        let items = vec![
            serde_json::json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What's the weather in Tokyo?"}]
            }),
            serde_json::json!({
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": "{\"location\":\"Tokyo\"}"
            }),
            serde_json::json!({
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "Sunny, 25°C"
            }),
        ];
        let mut req = make_responses_request(ResponsesInput::Items(items));
        req.tools = Some(vec![ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: Some("Get weather".into()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}}
                })),
            },
        }]);

        let result = convert_responses_request(&req).unwrap();
        let ctx = &result
            .conversation_state
            .current_message
            .user_input_message
            .user_input_message_context;
        assert!(!ctx.tool_results.is_empty());
    }

    #[test]
    fn test_convert_responses_request_maps_model_tools_user_stream() {
        let mut req = make_responses_request(ResponsesInput::Text("Hi".into()));
        req.user = Some("session-123".into());
        req.stream = true;

        let chat = normalize_responses_to_chat(&req).unwrap();
        assert_eq!(chat.model, "claude-sonnet-4-6");
        assert_eq!(chat.user.as_deref(), Some("session-123"));
        assert!(chat.stream);
    }
}
