//! OpenAI SSE 流式响应转换
//!
//! 将 Kiro 事件流转换为 OpenAI Chat Completions 的 SSE 格式

use std::collections::HashMap;

use uuid::Uuid;

use crate::common::converter::get_context_window_size;
use crate::kiro::model::events::Event;

use super::types::{
    ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, Delta, FunctionCall,
    ResponseMessage, ToolCall, Usage,
};

/// OpenAI 流式响应上下文
pub struct OpenAIStreamContext {
    /// 响应 ID
    id: String,
    /// 模型名
    model: String,
    /// 创建时间
    created: i64,
    /// 估算的输入 tokens
    input_tokens: i32,
    /// 输出 tokens 计数
    output_tokens: i32,
    /// 是否已发送 role delta
    role_sent: bool,
    /// 当前工具调用索引
    tool_call_index: i32,
    /// 工具调用 JSON 缓冲
    tool_json_buffers: HashMap<String, String>,
    /// 工具名称映射（短名称 → 原始名称）
    tool_name_map: HashMap<String, String>,
    /// 是否有工具调用
    has_tool_use: bool,
    /// 从 contextUsageEvent 计算的实际 input_tokens
    context_input_tokens: Option<i32>,
    /// 是否包含 usage
    include_usage: bool,
    /// stop_reason
    stop_reason: String,
}

impl OpenAIStreamContext {
    pub fn new(
        model: &str,
        input_tokens: i32,
        tool_name_map: HashMap<String, String>,
        include_usage: bool,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
            model: model.to_string(),
            created: chrono::Utc::now().timestamp(),
            input_tokens,
            output_tokens: 0,
            role_sent: false,
            tool_call_index: 0,
            tool_json_buffers: HashMap::new(),
            tool_name_map,
            has_tool_use: false,
            context_input_tokens: None,
            include_usage,
            stop_reason: "stop".to_string(),
        }
    }

    /// 处理 Kiro 事件，返回 SSE 字符串列表
    pub fn process_event(&mut self, event: &Event) -> Vec<String> {
        match event {
            Event::AssistantResponse(resp) => {
                let mut results = Vec::new();

                // 首次发送 role delta
                if !self.role_sent {
                    self.role_sent = true;
                    let chunk = self.make_chunk(
                        Delta {
                            role: Some("assistant".into()),
                            content: None,
                            tool_calls: None,
                        },
                        None,
                    );
                    results.push(format_sse(&chunk));
                }

                // 发送内容 delta
                if !resp.content.is_empty() {
                    self.output_tokens += estimate_tokens(&resp.content);
                    let chunk = self.make_chunk(
                        Delta {
                            role: None,
                            content: Some(resp.content.clone()),
                            tool_calls: None,
                        },
                        None,
                    );
                    results.push(format_sse(&chunk));
                }

                results
            }
            Event::ToolUse(tool_use) => {
                let mut results = Vec::new();
                self.has_tool_use = true;

                // 首次发送 role delta
                if !self.role_sent {
                    self.role_sent = true;
                    let chunk = self.make_chunk(
                        Delta {
                            role: Some("assistant".into()),
                            content: None,
                            tool_calls: None,
                        },
                        None,
                    );
                    results.push(format_sse(&chunk));
                }

                // 累积工具 JSON 并提取状态
                let is_first;
                let buffer_len;
                {
                    let buffer = self
                        .tool_json_buffers
                        .entry(tool_use.tool_use_id.clone())
                        .or_insert_with(String::new);
                    is_first = buffer.is_empty();
                    buffer.push_str(&tool_use.input);
                    buffer_len = buffer.len();
                }

                // 恢复原始工具名
                let original_name = self
                    .tool_name_map
                    .get(&tool_use.name)
                    .cloned()
                    .unwrap_or_else(|| tool_use.name.clone());

                let tc = if is_first {
                    ToolCall {
                        index: Some(self.tool_call_index),
                        id: Some(tool_use.tool_use_id.clone()),
                        call_type: Some("function".into()),
                        function: FunctionCall {
                            name: Some(original_name),
                            arguments: tool_use.input.clone(),
                        },
                    }
                } else {
                    ToolCall {
                        index: Some(self.tool_call_index),
                        id: None,
                        call_type: None,
                        function: FunctionCall {
                            name: None,
                            arguments: tool_use.input.clone(),
                        },
                    }
                };

                let chunk = self.make_chunk(
                    Delta {
                        role: None,
                        content: None,
                        tool_calls: Some(vec![tc]),
                    },
                    None,
                );
                results.push(format_sse(&chunk));

                if tool_use.stop {
                    self.tool_call_index += 1;
                    self.output_tokens += (buffer_len as f64 / 4.0).ceil() as i32;
                }

                results
            }
            Event::ContextUsage(ctx) => {
                let window_size = get_context_window_size(&self.model);
                let actual = (ctx.context_usage_percentage * (window_size as f64) / 100.0) as i32;
                self.context_input_tokens = Some(actual);
                if ctx.context_usage_percentage >= 100.0 {
                    self.stop_reason = "length".to_string();
                }
                Vec::new()
            }
            Event::Exception {
                exception_type, ..
            } => {
                if exception_type == "ContentLengthExceededException" {
                    self.stop_reason = "length".to_string();
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// 生成最终事件
    pub fn generate_final_events(&self) -> Vec<String> {
        let mut results = Vec::new();

        let finish_reason = if self.has_tool_use && self.stop_reason == "stop" {
            "tool_calls".to_string()
        } else {
            self.stop_reason.clone()
        };

        // 发送带 finish_reason 的最终 chunk
        let chunk = ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".into(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason),
            }],
            usage: if self.include_usage {
                let input = self.context_input_tokens.unwrap_or(self.input_tokens);
                Some(Usage {
                    prompt_tokens: input,
                    completion_tokens: self.output_tokens,
                    total_tokens: input + self.output_tokens,
                })
            } else {
                None
            },
        };
        results.push(format_sse(&chunk));

        // 发送 [DONE]
        results.push("data: [DONE]\n\n".to_string());

        results
    }

    fn make_chunk(&self, delta: Delta, finish_reason: Option<String>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".into(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage: None,
        }
    }
}

/// 非流式响应：从收集的事件构建完整的 ChatCompletionResponse
pub fn build_non_stream_response(
    model: &str,
    input_tokens: i32,
    events: &[Event],
    tool_name_map: &HashMap<String, String>,
) -> ChatCompletionResponse {
    let mut text_content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut has_tool_use = false;
    let mut stop_reason = "stop".to_string();
    let mut context_input_tokens: Option<i32> = None;
    let mut tool_json_buffers: HashMap<String, String> = HashMap::new();
    let mut tool_call_index: i32 = 0;

    for event in events {
        match event {
            Event::AssistantResponse(resp) => {
                text_content.push_str(&resp.content);
            }
            Event::ToolUse(tool_use) => {
                has_tool_use = true;

                let buffer = tool_json_buffers
                    .entry(tool_use.tool_use_id.clone())
                    .or_insert_with(String::new);
                buffer.push_str(&tool_use.input);

                if tool_use.stop {
                    let input: serde_json::Value = if buffer.is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(buffer).unwrap_or_else(|e| {
                            tracing::warn!(
                                "工具输入 JSON 解析失败: {}, tool_use_id: {}",
                                e,
                                tool_use.tool_use_id
                            );
                            serde_json::json!({})
                        })
                    };

                    let original_name = tool_name_map
                        .get(&tool_use.name)
                        .cloned()
                        .unwrap_or_else(|| tool_use.name.clone());

                    tool_calls.push(ToolCall {
                        index: Some(tool_call_index),
                        id: Some(tool_use.tool_use_id.clone()),
                        call_type: Some("function".into()),
                        function: FunctionCall {
                            name: Some(original_name),
                            arguments: input.to_string(),
                        },
                    });
                    tool_call_index += 1;
                }
            }
            Event::ContextUsage(ctx) => {
                let window_size = get_context_window_size(model);
                let actual = (ctx.context_usage_percentage * (window_size as f64) / 100.0) as i32;
                context_input_tokens = Some(actual);
                if ctx.context_usage_percentage >= 100.0 {
                    stop_reason = "length".to_string();
                }
            }
            Event::Exception {
                exception_type, ..
            } => {
                if exception_type == "ContentLengthExceededException" {
                    stop_reason = "length".to_string();
                }
            }
            _ => {}
        }
    }

    if has_tool_use && stop_reason == "stop" {
        stop_reason = "tool_calls".to_string();
    }

    let content = if text_content.is_empty() {
        None
    } else {
        Some(text_content.clone())
    };

    let tool_calls_field = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    let output_tokens = estimate_tokens(&text_content)
        + tool_json_buffers
            .values()
            .map(|v| estimate_tokens(v))
            .sum::<i32>();

    let final_input_tokens = context_input_tokens.unwrap_or(input_tokens);

    ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
        object: "chat.completion".into(),
        created: chrono::Utc::now().timestamp(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".into(),
                content,
                tool_calls: tool_calls_field,
            },
            finish_reason: Some(stop_reason),
        }],
        usage: Usage {
            prompt_tokens: final_input_tokens,
            completion_tokens: output_tokens,
            total_tokens: final_input_tokens + output_tokens,
        },
    }
}

/// 格式化为 SSE 字符串
fn format_sse(chunk: &ChatCompletionChunk) -> String {
    let json = serde_json::to_string(chunk).unwrap_or_default();
    format!("data: {}\n\n", json)
}

/// 简单的 token 估算（每 4 字符约 1 token）
fn estimate_tokens(text: &str) -> i32 {
    (text.len() as f64 / 4.0).ceil() as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::model::events::{AssistantResponseEvent, ToolUseEvent};

    fn make_assistant_event(content: &str) -> Event {
        let mut e = AssistantResponseEvent::default();
        e.content = content.to_string();
        Event::AssistantResponse(e)
    }

    #[test]
    fn test_stream_text_response() {
        let mut ctx = OpenAIStreamContext::new("claude-sonnet-4-6", 100, HashMap::new(), false);

        let event = make_assistant_event("Hello");

        let results = ctx.process_event(&event);
        // Should have role delta + content delta
        assert_eq!(results.len(), 2);
        assert!(results[0].contains("\"role\":\"assistant\""));
        assert!(results[1].contains("\"content\":\"Hello\""));
    }

    #[test]
    fn test_stream_final_events() {
        let ctx = OpenAIStreamContext::new("claude-sonnet-4-6", 100, HashMap::new(), false);
        let finals = ctx.generate_final_events();
        assert_eq!(finals.len(), 2);
        assert!(finals[0].contains("\"finish_reason\":\"stop\""));
        assert_eq!(finals[1], "data: [DONE]\n\n");
    }

    #[test]
    fn test_stream_tool_calls_finish_reason() {
        let mut ctx = OpenAIStreamContext::new("claude-sonnet-4-6", 100, HashMap::new(), false);

        let event = Event::ToolUse(ToolUseEvent {
            tool_use_id: "call_1".into(),
            name: "test_tool".into(),
            input: "{}".into(),
            stop: true,
        });
        ctx.process_event(&event);

        let finals = ctx.generate_final_events();
        assert!(finals[0].contains("\"finish_reason\":\"tool_calls\""));
    }

    #[test]
    fn test_non_stream_response() {
        let events = vec![make_assistant_event("Hello world")];

        let resp = build_non_stream_response("claude-sonnet-4-6", 50, &events, &HashMap::new());
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello world"));
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.prompt_tokens, 50);
    }

    #[test]
    fn test_non_stream_tool_calls() {
        let events = vec![
            Event::ToolUse(ToolUseEvent {
                tool_use_id: "call_1".into(),
                name: "get_weather".into(),
                input: r#"{"location":"Tokyo"}"#.into(),
                stop: true,
            }),
        ];

        let resp = build_non_stream_response("claude-sonnet-4-6", 50, &events, &HashMap::new());
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
        assert!(resp.choices[0].message.tool_calls.is_some());
        let tc = &resp.choices[0].message.tool_calls.as_ref().unwrap()[0];
        assert_eq!(tc.function.name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn test_tool_name_mapping_in_stream() {
        let mut name_map = HashMap::new();
        name_map.insert("short_name".to_string(), "original_long_name".to_string());

        let mut ctx = OpenAIStreamContext::new("claude-sonnet-4-6", 100, name_map, false);

        let event = Event::ToolUse(ToolUseEvent {
            tool_use_id: "call_1".into(),
            name: "short_name".into(),
            input: "{}".into(),
            stop: true,
        });

        let results = ctx.process_event(&event);
        let combined: String = results.join("");
        assert!(combined.contains("original_long_name"));
    }
}
