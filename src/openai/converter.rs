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

use super::types::{ChatCompletionRequest, ChatMessage};

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
    let mut history = build_history(&system_messages, messages, &model_id, &mut tool_name_map)?;

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

/// 构建历史消息
fn build_history(
    system_messages: &[&ChatMessage],
    messages: &[&ChatMessage],
    model_id: &str,
    tool_name_map: &mut HashMap<String, String>,
) -> Result<Vec<Message>, ConversionError> {
    let mut history = Vec::new();

    // 1. 处理系统消息
    if !system_messages.is_empty() {
        let system_content: String = system_messages
            .iter()
            .map(|m| m.text_content())
            .collect::<Vec<_>>()
            .join("\n");

        if !system_content.is_empty() {
            let system_content = format!("{}\n{}", system_content, shared::SYSTEM_CHUNKED_POLICY);

            let user_msg = HistoryUserMessage::new(system_content, model_id);
            history.push(Message::User(user_msg));

            let assistant_msg = HistoryAssistantMessage::new("I will follow these instructions.");
            history.push(Message::Assistant(assistant_msg));
        }
    }

    // 2. 处理对话历史（最后一条作为 currentMessage，不加入历史）
    let history_end = messages.len().saturating_sub(1);

    let mut user_buffer: Vec<&ChatMessage> = Vec::new();
    let mut assistant_buffer: Vec<&ChatMessage> = Vec::new();
    // tool 消息紧跟在 assistant 之后，需要和下一个 user 消息合并
    let mut tool_buffer: Vec<&ChatMessage> = Vec::new();

    for i in 0..history_end {
        let msg = messages[i];

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
        ChatMessage, FunctionCall, FunctionDefinition, ToolCall, ToolDefinition,
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
}
