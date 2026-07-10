//! OpenAI API 类型定义

use serde::{Deserialize, Deserializer, Serialize};

// === 错误响应 ===

/// OpenAI 格式的错误响应
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                code: None,
            },
        }
    }

    pub fn authentication_error() -> Self {
        Self {
            error: ErrorDetail {
                message: "Invalid API key".into(),
                error_type: "invalid_request_error".into(),
                code: Some("invalid_api_key".into()),
            },
        }
    }
}

// === Chat Completions 请求类型 ===

/// Chat Completions 请求体
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<i32>,
    #[serde(default)]
    pub max_completion_tokens: Option<i32>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default, deserialize_with = "deserialize_tools_lenient")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// 用于传递额外信息（如 session_id）
    #[serde(default)]
    pub user: Option<String>,
}

impl ChatCompletionRequest {
    /// 获取有效的 max_tokens，优先使用 max_completion_tokens
    #[allow(dead_code)]
    pub fn effective_max_tokens(&self) -> i32 {
        self.max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(4096)
    }
}

/// stop 字段支持 string 或 string 数组
#[derive(Debug, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

/// 流式选项
#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

// === 消息类型 ===

/// Chat 消息
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    /// 消息内容，可以是 string、null 或 content parts 数组
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    /// 工具调用（assistant 消息）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// 工具调用 ID（tool 角色消息必需）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// 函数名（tool 角色消息可选）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// 提取消息的文本内容
    pub fn text_content(&self) -> String {
        match &self.content {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(arr)) => {
                let mut parts = Vec::new();
                for item in arr {
                    if let Some(text) = item.get("text").and_then(|v| v.as_str())
                        && item.get("type").and_then(|v| v.as_str()) == Some("text")
                    {
                        parts.push(text.to_string());
                    }
                }
                parts.join("\n")
            }
            _ => String::new(),
        }
    }
}

// === 工具类型 ===

/// 工具定义（内部统一为 Chat Completions 的 `function` 嵌套结构）
///
/// 反序列化同时接受：
/// - Chat Completions：`{"type":"function","function":{"name":...}}`
/// - Responses API：`{"type":"function","name":...,"description":...,"parameters":...}`
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

impl<'de> Deserialize<'de> for ToolDefinition {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::try_from_value(&value).ok_or_else(|| {
            serde::de::Error::custom(
                "expected function tool with nested `function` or flat `name` fields",
            )
        })
    }
}

impl ToolDefinition {
    /// 从 JSON 值解析 function tool；非 function / 无法识别时返回 None
    pub fn try_from_value(value: &serde_json::Value) -> Option<Self> {
        let tool_type = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("function");
        if tool_type != "function" {
            return None;
        }

        if let Some(func) = value.get("function") {
            let name = func.get("name")?.as_str()?.to_string();
            return Some(Self {
                tool_type: tool_type.to_string(),
                function: FunctionDefinition {
                    name,
                    description: func
                        .get("description")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    parameters: func.get("parameters").cloned(),
                },
            });
        }

        // Responses / Codex 扁平格式
        let name = value.get("name")?.as_str()?.to_string();
        Some(Self {
            tool_type: tool_type.to_string(),
            function: FunctionDefinition {
                name,
                description: value
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                parameters: value.get("parameters").cloned(),
            },
        })
    }
}

/// 宽松解析 tools 列表：跳过非 function / 无法识别的条目（Codex 会附带 local_shell 等）
fn deserialize_tools_lenient<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<ToolDefinition>>, D::Error>
where
    D: Deserializer<'de>,
{
    let opt: Option<Vec<serde_json::Value>> = Option::deserialize(deserializer)?;
    let Some(values) = opt else {
        return Ok(None);
    };
    let tools: Vec<ToolDefinition> = values
        .iter()
        .filter_map(ToolDefinition::try_from_value)
        .collect();
    Ok(Some(tools))
}

/// 函数定义
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// 工具调用（在 assistant 消息和流式 delta 中使用）
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: FunctionCall,
}

/// 函数调用
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: String,
}

// === 非流式响应类型 ===

/// Chat Completion 响应
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// 选项
#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: i32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

/// 响应消息
#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// 思考内容（thinking 模型），非标准字段，兼容主流 OpenAI 客户端约定
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// Token 使用统计
#[derive(Debug, Serialize, Clone)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credits: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metering_unit: Option<String>,
}

// === Responses API 类型 ===

/// Responses API 请求体（最小实现，归一化后复用 Chat Completions 转换逻辑）
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, deserialize_with = "deserialize_tools_lenient")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub user: Option<String>,
}

/// Responses API `input` 字段：纯文本或结构化条目数组
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<serde_json::Value>),
}

/// Responses API 响应体
#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub model: String,
    pub status: String,
    pub output: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,
}

/// Responses API token 使用统计（字段命名与 Chat Completions 的 `Usage` 不同）
#[derive(Debug, Serialize, Clone)]
pub struct ResponsesUsage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub total_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credits: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metering_unit: Option<String>,
}

impl From<Usage> for ResponsesUsage {
    fn from(usage: Usage) -> Self {
        Self {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            credits: usage.credits,
            metering_unit: usage.metering_unit,
        }
    }
}

// === 流式响应类型 ===

/// Chat Completion Chunk（流式）
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// 流式选项
#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: i32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// 增量内容
#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// 思考内容增量（thinking 模型）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_chat_completion_request_basic() {
        let json = r#"{
            "model": "claude-sonnet-4-6",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-sonnet-4-6");
        assert_eq!(req.messages.len(), 1);
        assert!(!req.stream);
    }

    #[test]
    fn test_deserialize_request_with_tools() {
        let json = r#"{
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert_eq!(req.tools.as_ref().unwrap()[0].function.name, "get_weather");
    }

    #[test]
    fn test_serialize_chat_completion_response() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-abc123".into(),
            object: "chat.completion".into(),
            created: 1677858242,
            model: "claude-sonnet-4-6".into(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".into(),
                    content: Some("Hello!".into()),
                    tool_calls: None,
                    reasoning_content: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                credits: None,
                metering_unit: None,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_serialize_chunk() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-abc".into(),
            object: "chat.completion.chunk".into(),
            created: 1677858242,
            model: "claude-sonnet-4-6".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some("Hello".into()),
                    tool_calls: None,
                    reasoning_content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["object"], "chat.completion.chunk");
        assert!(json["choices"][0]["delta"].get("role").is_none());
        assert_eq!(json["choices"][0]["delta"]["content"], "Hello");
    }

    #[test]
    fn test_deserialize_null_content_with_tool_calls() {
        let json = r#"{
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"Tokyo\"}"
                }
            }]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert!(msg.content.is_none() || msg.content.as_ref().unwrap().is_null());
        assert!(msg.tool_calls.is_some());
        assert_eq!(msg.tool_calls.as_ref().unwrap()[0].function.arguments, "{\"location\":\"Tokyo\"}");
    }

    #[test]
    fn test_deserialize_tool_message() {
        let json = r#"{
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "Sunny, 25°C"
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_abc"));
        assert_eq!(msg.text_content(), "Sunny, 25°C");
    }

    #[test]
    fn test_text_content_extraction() {
        // String content
        let msg = ChatMessage {
            role: "user".into(),
            content: Some(serde_json::json!("Hello")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(msg.text_content(), "Hello");

        // Null content
        let msg = ChatMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        assert_eq!(msg.text_content(), "");
    }

    #[test]
    fn test_effective_max_tokens() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 8192,
            "max_tokens": 1024
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        // max_completion_tokens takes priority
        assert_eq!(req.effective_max_tokens(), 8192);

        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2048
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.effective_max_tokens(), 2048);

        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.effective_max_tokens(), 4096);
    }

    #[test]
    fn test_deserialize_responses_request_text_input() {
        let json = r#"{
            "model": "claude-sonnet-4-6",
            "input": "Hello!"
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-sonnet-4-6");
        assert!(matches!(req.input, ResponsesInput::Text(ref s) if s == "Hello!"));
        assert!(!req.stream);
        assert!(req.instructions.is_none());
    }

    #[test]
    fn test_deserialize_responses_flat_function_tools() {
        let json = r#"{
            "model": "claude-haiku-4-5-20251001",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type":"object","properties":{"city":{"type":"string"}}}
                },
                {"type": "local_shell"},
                {
                    "type": "function",
                    "function": {
                        "name": "nested_ok",
                        "parameters": {"type":"object"}
                    }
                }
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        let tools = req.tools.expect("tools");
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(tools[1].function.name, "nested_ok");
    }

    #[test]
    fn test_deserialize_responses_request_items_input() {
        let json = r#"{
            "model": "claude-sonnet-4-6",
            "instructions": "You are helpful.",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
            ]
        }"#;
        let req: ResponsesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.instructions.as_deref(), Some("You are helpful."));
        match req.input {
            ResponsesInput::Items(ref items) => assert_eq!(items.len(), 1),
            _ => panic!("expected Items variant"),
        }
    }

    #[test]
    fn test_stop_sequence_variants() {
        // Single string
        let json = r#"{"model":"t","messages":[],"stop":"END"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.stop, Some(StopSequence::Single(ref s)) if s == "END"));

        // Array
        let json = r#"{"model":"t","messages":[],"stop":["END","STOP"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.stop, Some(StopSequence::Multiple(ref v)) if v.len() == 2));
    }
}
