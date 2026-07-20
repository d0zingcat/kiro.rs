//! Responses API 流式（SSE）/非流式视图适配
//!
//! 将 Kiro 事件流转换为 OpenAI Responses API 的输出格式。
//!
//! # SSE 事件命名与整体顺序
//!
//! 每个 SSE 事件使用 `data: {"type":"...","..."}\n\n` 的形式，`type` 字段可供
//! 主流 SDK（`event.type`）区分事件种类，与 Chat Completions 侧的 `stream.rs`
//! 完全独立、不复用其 SSE 格式。整体事件顺序：
//!
//! ```text
//! response.created
//!   ├─ (可选，thinking 开启且检测到思考内容时)
//!   │    response.output_item.added        (item.type = "reasoning")
//!   │    response.reasoning_text.delta *N
//!   │    response.reasoning_text.done
//!   │    response.output_item.done
//!   ├─ (正文文本)
//!   │    response.output_item.added        (item.type = "message")
//!   │    response.content_part.added       (part.type = "output_text")
//!   │    response.output_text.delta *N
//!   │    response.output_text.done
//!   │    response.content_part.done
//!   │    response.output_item.done
//!   └─ (每个工具调用)
//!        response.output_item.added        (item.type = "function_call")
//!        response.function_call_arguments.delta *N
//!        response.function_call_arguments.done
//!        response.output_item.done
//! response.completed                        (response.output = 已完成的 items, 含 usage)
//! data: [DONE]
//! ```
//!
//! # 输出 item JSON 形状（流式 done / 非流式共用，已通过测试锁定）
//!
//! - reasoning：`{"id","type":"reasoning","status":"completed","content":[{"type":"reasoning_text","text"}]}`
//! - message：`{"id","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text","annotations":[]}]}`
//! - function_call：`{"id","type":"function_call","status":"completed","call_id","name","arguments"}`
//!
//! 该形状力求贴近 OpenAI Responses API 的真实结构，但为兼容层自定义精简版本
//! （例如 reasoning 使用完整原文而非官方的加密 summary）。

use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};
use uuid::Uuid;

use crate::common::converter::get_context_window_size;
use crate::kiro::model::events::{Event, MeteringEvent};

use super::stream::{build_non_stream_response, estimate_tokens};
use super::thinking::{ThinkingEvent, ThinkingStreamParser, is_gpt_hidden_cot_model};
use super::types::{ResponsesResponse, ResponsesUsage};
use super::usage::build_usage;

/// 当前正在流式输出的 Responses output item
#[derive(Debug)]
enum OpenItem {
    Reasoning {
        id: String,
        index: i32,
        text: String,
    },
    Message {
        id: String,
        index: i32,
        text: String,
    },
    FunctionCall {
        id: String,
        index: i32,
        call_id: String,
        name: String,
        arguments: String,
    },
    /// Codex freeform / OpenAI custom tool（如 `exec`）
    CustomToolCall {
        id: String,
        index: i32,
        call_id: String,
        name: String,
        /// 累积的上游 JSON arguments（关闭时再 unwrap 为 freeform input）
        arguments: String,
    },
}

/// Responses API 流式响应上下文
pub struct ResponsesStreamContext {
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
    /// 从 contextUsageEvent 计算的实际 input_tokens
    context_input_tokens: Option<i32>,
    /// 内部记录的结束原因（"stop" / "length"），用于决定最终 status
    stop_reason: String,
    /// 计费事件（来自 meteringEvent）
    metering: Option<MeteringEvent>,
    /// 是否启用 thinking（模型名包含 `-thinking`）
    thinking_enabled: bool,
    /// thinking 标签流式解析器
    thinking_parser: ThinkingStreamParser,
    /// 工具名称映射（短名称 → 原始名称）
    tool_name_map: HashMap<String, String>,
    /// 需按 `custom_tool_call` 回传的工具名（Codex freeform）
    custom_tool_names: HashSet<String>,
    /// 是否已发送 response.created
    created_sent: bool,
    /// 下一个 output_index
    output_index: i32,
    /// 当前正在流式输出的 item（同一时刻最多一个）
    current_item: Option<OpenItem>,
    /// 已完成的 output items（按顺序），用于 response.completed
    finished_items: Vec<Value>,
}

impl ResponsesStreamContext {
    /// 当前累积的 metering 事件（若有）
    pub fn metering(&self) -> Option<&MeteringEvent> {
        self.metering.as_ref()
    }

    /// 创建不启用 thinking 解析的流上下文
    #[allow(dead_code)] // 保留作为测试/公共 API 的便捷构造函数
    pub fn new(model: &str, input_tokens: i32, tool_name_map: HashMap<String, String>) -> Self {
        Self::new_with_thinking(model, input_tokens, tool_name_map, false, HashSet::new())
    }

    /// 创建启用 thinking 解析的流上下文
    pub fn new_with_thinking(
        model: &str,
        input_tokens: i32,
        tool_name_map: HashMap<String, String>,
        thinking_enabled: bool,
        custom_tool_names: HashSet<String>,
    ) -> Self {
        Self {
            id: format!("resp_{}", Uuid::new_v4().simple()),
            model: model.to_string(),
            created: chrono::Utc::now().timestamp(),
            input_tokens,
            output_tokens: 0,
            context_input_tokens: None,
            stop_reason: "stop".to_string(),
            metering: None,
            thinking_enabled,
            thinking_parser: ThinkingStreamParser::new(),
            tool_name_map,
            custom_tool_names,
            created_sent: false,
            output_index: 0,
            current_item: None,
            finished_items: Vec::new(),
        }
    }

    /// 处理 Kiro 事件，返回 SSE 字符串列表
    pub fn process_event(&mut self, event: &Event) -> Vec<String> {
        match event {
            Event::AssistantResponse(resp) => {
                let mut results = self.ensure_created();

                if !resp.content.is_empty() {
                    self.output_tokens += estimate_tokens(&resp.content);

                    if self.thinking_enabled {
                        for ev in self.thinking_parser.push(&resp.content) {
                            results.extend(self.dispatch_thinking_event(ev));
                        }
                    } else {
                        results.extend(self.push_message_delta(&resp.content));
                    }
                }

                results
            }
            Event::ToolUse(tool_use) => {
                let mut results = self.ensure_created();

                // thinking 模式下，tool_use 到来前需要 flush 掉解析器中滞留的探测缓冲
                if self.thinking_enabled {
                    for ev in self.thinking_parser.finish() {
                        results.extend(self.dispatch_thinking_event(ev));
                    }
                }

                let original_name = self
                    .tool_name_map
                    .get(&tool_use.name)
                    .cloned()
                    .unwrap_or_else(|| tool_use.name.clone());
                let is_custom = self.custom_tool_names.contains(&original_name);

                if is_custom {
                    let same_call = matches!(
                        &self.current_item,
                        Some(OpenItem::CustomToolCall { call_id, .. })
                            if call_id == &tool_use.tool_use_id
                    );
                    if !same_call {
                        results.extend(
                            self.open_custom_tool_call_item(&tool_use.tool_use_id, &original_name),
                        );
                    }
                    // 先缓冲 JSON arguments；关闭时 unwrap 成 freeform input 再发给 Codex
                    if let Some(OpenItem::CustomToolCall { arguments, .. }) = &mut self.current_item
                    {
                        arguments.push_str(&tool_use.input);
                    }
                    self.output_tokens += estimate_tokens(&tool_use.input);
                } else {
                    let same_call = matches!(
                        &self.current_item,
                        Some(OpenItem::FunctionCall { call_id, .. })
                            if call_id == &tool_use.tool_use_id
                    );
                    if !same_call {
                        results
                            .extend(self.open_function_call_item(&tool_use.tool_use_id, &original_name));
                    }
                    results.extend(self.push_function_call_delta(&tool_use.input));
                    self.output_tokens += estimate_tokens(&tool_use.input);
                }

                if tool_use.stop {
                    results.extend(self.close_current_item());
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
            Event::Metering(m) => {
                self.metering = Some(m.clone());
                Vec::new()
            }
            Event::ReasoningContent(reasoning) => {
                // GPT-5.6 hidden CoT：不向 Codex 转发 reasoning SSE，避免
                // `ReasoningRawContentDelta without active item`
                if is_gpt_hidden_cot_model(&self.model) {
                    return Vec::new();
                }
                let mut results = self.ensure_created();
                // 原生 reasoning 事件始终转发到 Responses reasoning item，
                // 不依赖模型名 `-thinking` 后缀（Opus 4.8 等会默认推送）。
                if let Some(delta) = reasoning.text_delta() {
                    self.output_tokens += estimate_tokens(delta);
                    results.extend(self.push_reasoning_delta(delta));
                }
                // signature 收尾：关闭当前 reasoning item，便于后续 message/tool 接上
                if reasoning.has_signature() {
                    if matches!(self.current_item, Some(OpenItem::Reasoning { .. })) {
                        results.extend(self.close_current_item());
                    }
                }
                results
            }
            _ => Vec::new(),
        }
    }

    /// 生成最终事件（response.completed + [DONE]）
    pub fn generate_final_events(&mut self) -> Vec<String> {
        let mut results = self.ensure_created();

        if self.thinking_enabled {
            for ev in self.thinking_parser.finish() {
                results.extend(self.dispatch_thinking_event(ev));
            }
        }

        results.extend(self.close_current_item());

        let input = self.context_input_tokens.unwrap_or(self.input_tokens);
        let usage: ResponsesUsage =
            build_usage(input, self.output_tokens, self.metering.as_ref()).into();
        let usage_value = serde_json::to_value(&usage).unwrap_or_default();

        let status = if self.stop_reason == "length" {
            "incomplete"
        } else {
            "completed"
        };

        results.push(sse(json!({
            "type": "response.completed",
            "response": {
                "id": self.id,
                "object": "response",
                "created_at": self.created,
                "model": self.model,
                "status": status,
                "output": self.finished_items,
                "usage": usage_value
            }
        })));
        results.push("data: [DONE]\n\n".to_string());

        results
    }

    /// 首次调用时发送 response.created，之后为空操作
    fn ensure_created(&mut self) -> Vec<String> {
        if self.created_sent {
            return Vec::new();
        }
        self.created_sent = true;
        vec![sse(json!({
            "type": "response.created",
            "response": {
                "id": self.id,
                "object": "response",
                "created_at": self.created,
                "model": self.model,
                "status": "in_progress",
                "output": []
            }
        }))]
    }

    /// 将 thinking 解析事件路由到 reasoning / message 增量
    fn dispatch_thinking_event(&mut self, event: ThinkingEvent) -> Vec<String> {
        match event {
            ThinkingEvent::Reasoning(text) => self.push_reasoning_delta(&text),
            ThinkingEvent::Content(text) => self.push_message_delta(&text),
        }
    }

    fn next_output_index(&mut self) -> i32 {
        let idx = self.output_index;
        self.output_index += 1;
        idx
    }

    /// 打开一个 reasoning item（若已打开则不重复发送 added 事件）
    fn open_reasoning_item(&mut self) -> Vec<String> {
        if matches!(self.current_item, Some(OpenItem::Reasoning { .. })) {
            return Vec::new();
        }
        let mut events = self.close_current_item();
        let id = format!("rs_{}", Uuid::new_v4().simple());
        let index = self.next_output_index();
        events.push(sse(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": {
                "id": id,
                "type": "reasoning",
                "status": "in_progress",
                "content": [{"type": "reasoning_text", "text": ""}]
            }
        })));
        self.current_item = Some(OpenItem::Reasoning {
            id,
            index,
            text: String::new(),
        });
        events
    }

    fn push_reasoning_delta(&mut self, delta: &str) -> Vec<String> {
        if delta.is_empty() {
            return Vec::new();
        }
        let mut events = self.open_reasoning_item();
        let (id, index) = match &mut self.current_item {
            Some(OpenItem::Reasoning { id, index, text }) => {
                text.push_str(delta);
                (id.clone(), *index)
            }
            _ => unreachable!("open_reasoning_item guarantees Reasoning variant"),
        };
        events.push(sse(json!({
            "type": "response.reasoning_text.delta",
            "item_id": id,
            "output_index": index,
            "content_index": 0,
            "delta": delta
        })));
        events
    }

    /// 打开一个 message item（若已打开则不重复发送 added 事件）
    fn open_message_item(&mut self) -> Vec<String> {
        if matches!(self.current_item, Some(OpenItem::Message { .. })) {
            return Vec::new();
        }
        let mut events = self.close_current_item();
        let id = format!("msg_{}", Uuid::new_v4().simple());
        let index = self.next_output_index();
        events.push(sse(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": []
            }
        })));
        events.push(sse(json!({
            "type": "response.content_part.added",
            "item_id": id,
            "output_index": index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}
        })));
        self.current_item = Some(OpenItem::Message {
            id,
            index,
            text: String::new(),
        });
        events
    }

    fn push_message_delta(&mut self, delta: &str) -> Vec<String> {
        if delta.is_empty() {
            return Vec::new();
        }
        let mut events = self.open_message_item();
        let (id, index) = match &mut self.current_item {
            Some(OpenItem::Message { id, index, text }) => {
                text.push_str(delta);
                (id.clone(), *index)
            }
            _ => unreachable!("open_message_item guarantees Message variant"),
        };
        events.push(sse(json!({
            "type": "response.output_text.delta",
            "item_id": id,
            "output_index": index,
            "content_index": 0,
            "delta": delta
        })));
        events
    }

    /// 打开一个 function_call item（关闭之前打开的其他 item）
    fn open_function_call_item(&mut self, call_id: &str, name: &str) -> Vec<String> {
        let mut events = self.close_current_item();
        let id = format!("fc_{}", Uuid::new_v4().simple());
        let index = self.next_output_index();
        events.push(sse(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": {
                "id": id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": name,
                "arguments": ""
            }
        })));
        self.current_item = Some(OpenItem::FunctionCall {
            id,
            index,
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments: String::new(),
        });
        events
    }

    fn push_function_call_delta(&mut self, delta: &str) -> Vec<String> {
        if delta.is_empty() {
            return Vec::new();
        }
        let (id, index) = match &mut self.current_item {
            Some(OpenItem::FunctionCall {
                id,
                index,
                arguments,
                ..
            }) => {
                arguments.push_str(delta);
                (id.clone(), *index)
            }
            _ => return Vec::new(),
        };
        vec![sse(json!({
            "type": "response.function_call_arguments.delta",
            "item_id": id,
            "output_index": index,
            "delta": delta
        }))]
    }

    /// 打开一个 custom_tool_call item（Codex freeform tools）
    fn open_custom_tool_call_item(&mut self, call_id: &str, name: &str) -> Vec<String> {
        let mut events = self.close_current_item();
        let id = format!("ctc_{}", Uuid::new_v4().simple());
        let index = self.next_output_index();
        events.push(sse(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": {
                "id": id,
                "type": "custom_tool_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": name,
                "input": ""
            }
        })));
        self.current_item = Some(OpenItem::CustomToolCall {
            id,
            index,
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments: String::new(),
        });
        events
    }

    /// 关闭当前打开的 item（若有），发送对应的 done 事件序列并归档到 finished_items
    fn close_current_item(&mut self) -> Vec<String> {
        let item = match self.current_item.take() {
            Some(item) => item,
            None => return Vec::new(),
        };

        let mut events = Vec::new();
        match item {
            OpenItem::Reasoning { id, index, text } => {
                events.push(sse(json!({
                    "type": "response.reasoning_text.done",
                    "item_id": id,
                    "output_index": index,
                    "content_index": 0,
                    "text": text
                })));
                let item_json = reasoning_item_json(&id, &text);
                events.push(sse(json!({
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": item_json.clone()
                })));
                self.finished_items.push(item_json);
            }
            OpenItem::Message { id, index, text } => {
                events.push(sse(json!({
                    "type": "response.output_text.done",
                    "item_id": id,
                    "output_index": index,
                    "content_index": 0,
                    "text": text
                })));
                events.push(sse(json!({
                    "type": "response.content_part.done",
                    "item_id": id,
                    "output_index": index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []}
                })));
                let item_json = message_item_json(&id, &text);
                events.push(sse(json!({
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": item_json.clone()
                })));
                self.finished_items.push(item_json);
            }
            OpenItem::FunctionCall {
                id,
                index,
                call_id,
                name,
                arguments,
            } => {
                events.push(sse(json!({
                    "type": "response.function_call_arguments.done",
                    "item_id": id,
                    "output_index": index,
                    "arguments": arguments
                })));
                let item_json = function_call_item_json(&id, &call_id, &name, &arguments);
                events.push(sse(json!({
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": item_json.clone()
                })));
                self.finished_items.push(item_json);
            }
            OpenItem::CustomToolCall {
                id,
                index,
                call_id,
                name,
                arguments,
            } => {
                let input = unwrap_custom_tool_input(&arguments);
                // 一次推送完整 freeform input（上游按 JSON arguments 流式，难以边解析边 unwrap）
                if !input.is_empty() {
                    events.push(sse(json!({
                        "type": "response.custom_tool_call_input.delta",
                        "item_id": id,
                        "output_index": index,
                        "delta": input
                    })));
                }
                events.push(sse(json!({
                    "type": "response.custom_tool_call_input.done",
                    "item_id": id,
                    "output_index": index,
                    "input": input
                })));
                let item_json = custom_tool_call_item_json(&id, &call_id, &name, &input);
                events.push(sse(json!({
                    "type": "response.output_item.done",
                    "output_index": index,
                    "item": item_json.clone()
                })));
                self.finished_items.push(item_json);
            }
        }
        events
    }
}

/// 非流式响应：从收集的事件构建完整的 ResponsesResponse
///
/// 内部复用 `build_non_stream_response`（Chat Completions 侧的事件聚合逻辑，
/// 已经处理了 thinking 提取 / tool_name_map 还原 / metering→usage），
/// 再将其消息内容映射为 Responses `output[]` 的 item 形状。
///
pub fn build_responses_non_stream(
    model: &str,
    input_tokens: i32,
    events: &[Event],
    tool_name_map: &HashMap<String, String>,
    extract_thinking: bool,
    custom_tool_names: &HashSet<String>,
) -> ResponsesResponse {
    // 原生 reasoningContentEvent 文本（与 <thinking> 标签提取互补）
    // GPT-5.6 hidden CoT：不写入 output，避免 Codex 解析异常
    let mut native_reasoning = String::new();
    if !is_gpt_hidden_cot_model(model) {
        for event in events {
            if let Event::ReasoningContent(r) = event {
                if let Some(delta) = r.text_delta() {
                    native_reasoning.push_str(delta);
                }
            }
        }
    }

    let chat = build_non_stream_response(model, input_tokens, events, tool_name_map, extract_thinking);
    let message = &chat.choices[0].message;

    let mut output = Vec::new();

    let reasoning_text = if !native_reasoning.is_empty() {
        Some(native_reasoning)
    } else {
        message.reasoning_content.clone()
    };

    if let Some(reasoning) = reasoning_text.as_deref().filter(|s| !s.is_empty()) {
        output.push(reasoning_item_json(
            &format!("rs_{}", Uuid::new_v4().simple()),
            reasoning,
        ));
    }

    if let Some(content) = message.content.as_deref() {
        output.push(message_item_json(
            &format!("msg_{}", Uuid::new_v4().simple()),
            content,
        ));
    }

    if let Some(tool_calls) = message.tool_calls.as_ref() {
        for tc in tool_calls {
            let name = tc.function.name.as_deref().unwrap_or_default();
            let call_id = tc.id.as_deref().unwrap_or_default();
            if custom_tool_names.contains(name) {
                let input = unwrap_custom_tool_input(&tc.function.arguments);
                output.push(custom_tool_call_item_json(
                    &format!("ctc_{}", Uuid::new_v4().simple()),
                    call_id,
                    name,
                    &input,
                ));
            } else {
                output.push(function_call_item_json(
                    &format!("fc_{}", Uuid::new_v4().simple()),
                    call_id,
                    name,
                    &tc.function.arguments,
                ));
            }
        }
    }

    let status = match chat.choices[0].finish_reason.as_deref() {
        Some("length") => "incomplete",
        _ => "completed",
    };

    ResponsesResponse {
        id: format!("resp_{}", Uuid::new_v4().simple()),
        object: "response".into(),
        created_at: chat.created,
        model: model.to_string(),
        status: status.into(),
        output,
        usage: Some(chat.usage.into()),
    }
}

/// 将 Kiro JSON function arguments 还原为 Codex freeform custom tool input
fn unwrap_custom_tool_input(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        match v {
            Value::String(s) => return s,
            Value::Object(map) => {
                for key in ["input", "code", "source", "script", "content"] {
                    if let Some(Value::String(s)) = map.get(key) {
                        return s.clone();
                    }
                }
            }
            _ => {}
        }
    }
    raw.to_string()
}

/// reasoning output item 的 JSON 形状（流式 done 事件与非流式共用，已通过测试锁定）
fn reasoning_item_json(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": text}]
    })
}

/// message output item 的 JSON 形状
fn message_item_json(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}]
    })
}

/// function_call output item 的 JSON 形状
fn function_call_item_json(id: &str, call_id: &str, name: &str, arguments: &str) -> Value {
    json!({
        "id": id,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments
    })
}

/// custom_tool_call output item 的 JSON 形状（Codex freeform）
fn custom_tool_call_item_json(id: &str, call_id: &str, name: &str, input: &str) -> Value {
    json!({
        "id": id,
        "type": "custom_tool_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "input": input
    })
}

/// 格式化为 SSE 字符串
fn sse(value: Value) -> String {
    format!("data: {}\n\n", serde_json::to_string(&value).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::model::events::{AssistantResponseEvent, MeteringEvent, ToolUseEvent};

    fn make_assistant_event(content: &str) -> Event {
        let mut e = AssistantResponseEvent::default();
        e.content = content.to_string();
        Event::AssistantResponse(e)
    }

    /// 解析 SSE 字符串列表为 (type, json) 序列，跳过 [DONE]
    fn parse_events(strs: &[String]) -> Vec<serde_json::Value> {
        strs.iter()
            .filter(|s| s.starts_with("data: {"))
            .filter_map(|s| {
                let json_str = s.trim_start_matches("data: ").trim_end();
                serde_json::from_str::<serde_json::Value>(json_str).ok()
            })
            .collect()
    }

    fn types_of(events: &[serde_json::Value]) -> Vec<String> {
        events
            .iter()
            .map(|v| v["type"].as_str().unwrap_or_default().to_string())
            .collect()
    }

    // === Step 1: 纯文本事件序列 ===

    #[test]
    fn test_plain_text_event_sequence() {
        let mut ctx = ResponsesStreamContext::new("claude-sonnet-4-6", 100, HashMap::new());

        let mut all = ctx.process_event(&make_assistant_event("Hello"));
        all.extend(ctx.generate_final_events());

        let parsed = parse_events(&all);
        let types = types_of(&parsed);

        assert_eq!(
            types,
            vec![
                "response.created",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );

        // delta 内容正确
        let delta_event = &parsed[3];
        assert_eq!(delta_event["delta"], "Hello");

        // 终止符
        assert_eq!(all.last().unwrap(), "data: [DONE]\n\n");

        // response.completed 中 output 包含已完成的 message item
        let completed = &parsed[7];
        assert_eq!(completed["response"]["status"], "completed");
        assert_eq!(completed["response"]["output"][0]["type"], "message");
        assert_eq!(
            completed["response"]["output"][0]["content"][0]["text"],
            "Hello"
        );
        assert!(completed["response"]["usage"].is_object());
    }

    #[test]
    fn test_multiple_text_deltas_accumulate() {
        let mut ctx = ResponsesStreamContext::new("claude-sonnet-4-6", 100, HashMap::new());

        let mut all = ctx.process_event(&make_assistant_event("Hello "));
        all.extend(ctx.process_event(&make_assistant_event("world")));
        all.extend(ctx.generate_final_events());

        let parsed = parse_events(&all);
        let deltas: Vec<&str> = parsed
            .iter()
            .filter(|v| v["type"] == "response.output_text.delta")
            .map(|v| v["delta"].as_str().unwrap())
            .collect();
        assert_eq!(deltas, vec!["Hello ", "world"]);

        let completed = parsed.last().unwrap();
        assert_eq!(
            completed["response"]["output"][0]["content"][0]["text"],
            "Hello world"
        );
    }

    #[test]
    fn test_empty_stream_still_emits_created_and_completed() {
        let mut ctx = ResponsesStreamContext::new("claude-sonnet-4-6", 100, HashMap::new());
        let all = ctx.generate_final_events();
        let parsed = parse_events(&all);
        let types = types_of(&parsed);
        assert_eq!(types, vec!["response.created", "response.completed"]);
        assert_eq!(parsed[1]["response"]["output"].as_array().unwrap().len(), 0);
    }

    // === Step 3: 工具调用事件序列 ===

    #[test]
    fn test_tool_call_event_sequence() {
        let mut ctx = ResponsesStreamContext::new("claude-sonnet-4-6", 100, HashMap::new());

        let mut all = ctx.process_event(&Event::ToolUse(ToolUseEvent {
            tool_use_id: "call_1".into(),
            name: "get_weather".into(),
            input: r#"{"location":"Tokyo"}"#.into(),
            stop: true,
        }));
        all.extend(ctx.generate_final_events());

        let parsed = parse_events(&all);
        let types = types_of(&parsed);

        assert_eq!(
            types,
            vec![
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );

        let added = &parsed[1];
        assert_eq!(added["item"]["type"], "function_call");
        assert_eq!(added["item"]["call_id"], "call_1");
        assert_eq!(added["item"]["name"], "get_weather");

        let completed = parsed.last().unwrap();
        let item = &completed["response"]["output"][0];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["call_id"], "call_1");
        assert_eq!(item["name"], "get_weather");
        assert_eq!(item["arguments"], r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn test_tool_call_streamed_in_multiple_chunks() {
        let mut ctx = ResponsesStreamContext::new("claude-sonnet-4-6", 100, HashMap::new());

        let mut all = ctx.process_event(&Event::ToolUse(ToolUseEvent {
            tool_use_id: "call_1".into(),
            name: "get_weather".into(),
            input: r#"{"location":"#.into(),
            stop: false,
        }));
        all.extend(ctx.process_event(&Event::ToolUse(ToolUseEvent {
            tool_use_id: "call_1".into(),
            name: "get_weather".into(),
            input: r#""Tokyo"}"#.into(),
            stop: true,
        })));
        all.extend(ctx.generate_final_events());

        let parsed = parse_events(&all);
        // 只应该有一个 output_item.added（同一 tool_use_id 不应重复打开 item）
        let added_count = parsed
            .iter()
            .filter(|v| v["type"] == "response.output_item.added")
            .count();
        assert_eq!(added_count, 1);

        let completed = parsed.last().unwrap();
        let item = &completed["response"]["output"][0];
        assert_eq!(item["arguments"], r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn test_tool_name_mapping_restored_in_stream() {
        let mut name_map = HashMap::new();
        name_map.insert("short_name".to_string(), "original_long_name".to_string());

        let mut ctx = ResponsesStreamContext::new("claude-sonnet-4-6", 100, name_map);

        let all = ctx.process_event(&Event::ToolUse(ToolUseEvent {
            tool_use_id: "call_1".into(),
            name: "short_name".into(),
            input: "{}".into(),
            stop: true,
        }));

        let combined: String = all.join("");
        assert!(combined.contains("original_long_name"));
    }

    // === Step 3: reasoning 事件序列 ===

    #[test]
    fn test_reasoning_then_message_event_sequence() {
        let mut ctx = ResponsesStreamContext::new_with_thinking(
            "claude-sonnet-4-6-thinking",
            100,
            HashMap::new(),
            true,
            HashSet::new(),
        );

        let mut all = ctx.process_event(&make_assistant_event("<thinking>\nabc</thinking>\n\nhello"));
        all.extend(ctx.generate_final_events());

        let parsed = parse_events(&all);
        let types = types_of(&parsed);

        assert_eq!(
            types,
            vec![
                "response.created",
                "response.output_item.added",  // reasoning
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.output_item.done",   // reasoning done
                "response.output_item.added",  // message
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",   // message done
                "response.completed",
            ]
        );

        let completed = parsed.last().unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["type"], "reasoning_text");
        assert_eq!(output[0]["content"][0]["text"], "abc");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "hello");
    }

    #[test]
    fn test_reasoning_disabled_keeps_raw_tags_in_message() {
        let mut ctx = ResponsesStreamContext::new("claude-sonnet-4-6", 100, HashMap::new());

        let mut all = ctx.process_event(&make_assistant_event("<thinking>abc</thinking>\n\nhello"));
        all.extend(ctx.generate_final_events());

        let parsed = parse_events(&all);
        let completed = parsed.last().unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "message");
        assert!(
            output[0]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("<thinking>")
        );
    }

    #[test]
    fn test_native_reasoning_content_event_forwarded_without_thinking_suffix() {
        use crate::kiro::model::events::ReasoningContentEvent;

        // Opus 4.8 等即使模型名无 -thinking 也会推送原生 reasoningContentEvent
        let mut ctx = ResponsesStreamContext::new("claude-opus-4-8", 100, HashMap::new());

        let mut all = ctx.process_event(&Event::ReasoningContent(ReasoningContentEvent::text(
            "I've",
        )));
        all.extend(ctx.process_event(&Event::ReasoningContent(
            ReasoningContentEvent::text(" verified"),
        )));
        all.extend(ctx.process_event(&Event::ReasoningContent(
            ReasoningContentEvent::signature("sig_abc"),
        )));
        all.extend(ctx.process_event(&make_assistant_event("STATUS=OK")));
        all.extend(ctx.generate_final_events());

        let parsed = parse_events(&all);
        let types = types_of(&parsed);

        assert!(types.contains(&"response.reasoning_text.delta".to_string()));
        assert!(types.contains(&"response.output_text.delta".to_string()));

        let completed = parsed.last().unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "I've verified");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "STATUS=OK");
    }

    #[test]
    fn test_non_stream_native_reasoning_content_event() {
        use crate::kiro::model::events::ReasoningContentEvent;

        let events = vec![
            Event::ReasoningContent(ReasoningContentEvent::text("plan")),
            Event::ReasoningContent(ReasoningContentEvent::text(" A")),
            make_assistant_event("done"),
        ];
        let resp =
            build_responses_non_stream("claude-opus-4-8", 50, &events, &HashMap::new(), false, &HashSet::new());
        assert_eq!(resp.output.len(), 2);
        assert_eq!(resp.output[0]["type"], "reasoning");
        assert_eq!(resp.output[0]["content"][0]["text"], "plan A");
        assert_eq!(resp.output[1]["type"], "message");
        assert_eq!(resp.output[1]["content"][0]["text"], "done");
    }

    #[test]
    fn test_usage_includes_credits() {
        let mut ctx = ResponsesStreamContext::new("claude-sonnet-4-6", 100, HashMap::new());
        ctx.process_event(&make_assistant_event("Hi"));
        ctx.process_event(&Event::Metering(MeteringEvent {
            unit: Some("credit".into()),
            unit_plural: Some("credits".into()),
            usage: 0.02,
        }));

        let all = ctx.generate_final_events();
        let parsed = parse_events(&all);
        let completed = parsed.last().unwrap();
        assert_eq!(completed["response"]["usage"]["credits"], 0.02);
        assert_eq!(completed["response"]["usage"]["metering_unit"], "credit");
    }

    // === 非流式视图 ===

    #[test]
    fn test_non_stream_plain_text() {
        let events = vec![make_assistant_event("Hello world")];
        let resp = build_responses_non_stream("claude-sonnet-4-6", 50, &events, &HashMap::new(), false, &HashSet::new());

        assert_eq!(resp.object, "response");
        assert_eq!(resp.status, "completed");
        assert_eq!(resp.output.len(), 1);
        assert_eq!(resp.output[0]["type"], "message");
        assert_eq!(resp.output[0]["content"][0]["text"], "Hello world");
        assert_eq!(resp.usage.as_ref().unwrap().input_tokens, 50);
    }

    #[test]
    fn test_non_stream_tool_calls() {
        let events = vec![Event::ToolUse(ToolUseEvent {
            tool_use_id: "call_1".into(),
            name: "get_weather".into(),
            input: r#"{"location":"Tokyo"}"#.into(),
            stop: true,
        })];

        let resp = build_responses_non_stream("claude-sonnet-4-6", 50, &events, &HashMap::new(), false, &HashSet::new());

        assert_eq!(resp.output.len(), 1);
        assert_eq!(resp.output[0]["type"], "function_call");
        assert_eq!(resp.output[0]["call_id"], "call_1");
        assert_eq!(resp.output[0]["name"], "get_weather");
        assert_eq!(resp.output[0]["arguments"], r#"{"location":"Tokyo"}"#);
    }

    #[test]
    fn test_non_stream_reasoning_and_message() {
        let events = vec![make_assistant_event("<thinking>\nabc</thinking>\n\nhello")];
        let resp =
            build_responses_non_stream("claude-sonnet-4-6-thinking", 50, &events, &HashMap::new(), true, &HashSet::new());

        assert_eq!(resp.output.len(), 2);
        assert_eq!(resp.output[0]["type"], "reasoning");
        assert_eq!(resp.output[0]["content"][0]["type"], "reasoning_text");
        assert_eq!(resp.output[0]["content"][0]["text"], "abc");
        assert_eq!(resp.output[1]["type"], "message");
        assert_eq!(resp.output[1]["content"][0]["text"], "hello");
    }

    #[test]
    fn test_non_stream_tool_name_mapping_restored() {
        let mut name_map = HashMap::new();
        name_map.insert("short_name".to_string(), "original_long_name".to_string());

        let events = vec![Event::ToolUse(ToolUseEvent {
            tool_use_id: "call_1".into(),
            name: "short_name".into(),
            input: "{}".into(),
            stop: true,
        })];

        let resp = build_responses_non_stream("claude-sonnet-4-6", 50, &events, &name_map, false, &HashSet::new());
        assert_eq!(resp.output[0]["name"], "original_long_name");
    }

    #[test]
    fn test_non_stream_usage_includes_credits() {
        let events = vec![
            make_assistant_event("Hello"),
            Event::Metering(MeteringEvent {
                unit: Some("credit".into()),
                unit_plural: Some("credits".into()),
                usage: 0.03,
            }),
        ];
        let resp = build_responses_non_stream("claude-sonnet-4-6", 50, &events, &HashMap::new(), false, &HashSet::new());
        let usage = resp.usage.as_ref().unwrap();
        assert_eq!(usage.credits, Some(0.03));
        assert_eq!(usage.metering_unit.as_deref(), Some("credit"));
    }

    // === 锁定 reasoning / message / function_call 的 output item JSON 形状 ===

    #[test]
    fn test_reasoning_item_json_shape_is_locked() {
        let v = reasoning_item_json("rs_abc", "some thought");
        assert_eq!(v["id"], "rs_abc");
        assert_eq!(v["type"], "reasoning");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["content"][0]["type"], "reasoning_text");
        assert_eq!(v["content"][0]["text"], "some thought");
        assert_eq!(v.as_object().unwrap().len(), 4);
    }

    #[test]
    fn test_message_item_json_shape_is_locked() {
        let v = message_item_json("msg_abc", "hello");
        assert_eq!(v["id"], "msg_abc");
        assert_eq!(v["type"], "message");
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["content"][0]["type"], "output_text");
        assert_eq!(v["content"][0]["text"], "hello");
        assert_eq!(v["content"][0]["annotations"], serde_json::json!([]));
    }

    #[test]
    fn test_function_call_item_json_shape_is_locked() {
        let v = function_call_item_json("fc_abc", "call_1", "get_weather", "{}");
        assert_eq!(v["id"], "fc_abc");
        assert_eq!(v["type"], "function_call");
        assert_eq!(v["status"], "completed");
        assert_eq!(v["call_id"], "call_1");
        assert_eq!(v["name"], "get_weather");
        assert_eq!(v["arguments"], "{}");
    }
}
