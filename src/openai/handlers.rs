//! OpenAI API Handler 函数

use std::convert::Infallible;

use crate::kiro::model::events::Event;
use crate::kiro::model::requests::kiro::KiroRequest;
use crate::kiro::parser::decoder::EventStreamDecoder;
use axum::{
    Json as JsonExtractor,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
};
use bytes::Bytes;
use futures::{StreamExt, stream};
use std::time::Duration;
use tokio::time::interval;

use super::converter::{ConversionError, convert_request, convert_responses_request};
use super::middleware::AppState;
use super::responses_stream::{ResponsesStreamContext, build_responses_non_stream};
use super::stream::{OpenAIStreamContext, build_non_stream_response};
use super::thinking::is_thinking_model;
use super::types::{ChatCompletionRequest, ErrorResponse, ResponsesInput, ResponsesRequest};

/// 将 KiroProvider 错误映射为 OpenAI 格式的 HTTP 响应
fn map_provider_error(err: anyhow::Error) -> Response {
    let err_str = err.to_string();

    if err_str.contains("CONTENT_LENGTH_EXCEEDS_THRESHOLD") {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "invalid_request_error",
                "Context window is full. Reduce conversation history, system prompt, or tools.",
            )),
        )
            .into_response();
    }

    if err_str.contains("Input is too long") {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "invalid_request_error",
                "Input is too long. Reduce the size of your messages.",
            )),
        )
            .into_response();
    }

    tracing::error!("Kiro API 调用失败: {}", err);
    (
        StatusCode::BAD_GATEWAY,
        Json(ErrorResponse::new(
            "server_error",
            format!("Upstream API error: {}", err),
        )),
    )
        .into_response()
}

/// POST /v1/chat/completions
pub async fn post_chat_completions(
    State(state): State<AppState>,
    JsonExtractor(payload): JsonExtractor<ChatCompletionRequest>,
) -> Response {
    tracing::info!(
        model = %payload.model,
        stream = %payload.stream,
        message_count = %payload.messages.len(),
        "Received POST /v1/chat/completions request"
    );

    // 检查 KiroProvider 是否可用
    let provider = match &state.kiro_provider {
        Some(p) => p.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::new(
                    "server_error",
                    "Kiro API provider not configured",
                )),
            )
                .into_response();
        }
    };

    // 转换请求
    let conversion_result = match convert_request(&payload) {
        Ok(result) => result,
        Err(e) => {
            let message = match &e {
                ConversionError::UnsupportedModel(model) => {
                    format!("Model not supported: {}", model)
                }
                ConversionError::EmptyMessages => "Messages list is empty".to_string(),
            };
            tracing::warn!("请求转换失败: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("invalid_request_error", message)),
            )
                .into_response();
        }
    };

    // 构建 Kiro 请求
    let kiro_request = KiroRequest {
        conversation_state: conversion_result.conversation_state,
        profile_arn: None,
    };

    let request_body = match serde_json::to_string(&kiro_request) {
        Ok(body) => body,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    "server_error",
                    format!("Failed to serialize request: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 估算输入 tokens（简化版，使用 Anthropic 类型的 token 计数）
    let input_tokens = estimate_input_tokens(&payload);

    let tool_name_map = conversion_result.tool_name_map;
    let include_usage = payload
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    // 模型名包含 `-thinking`（大小写不敏感）即视为启用 thinking
    let thinking_enabled = is_thinking_model(&payload.model);

    if payload.stream {
        handle_stream_request(
            provider,
            &request_body,
            &payload.model,
            input_tokens,
            tool_name_map,
            include_usage,
            thinking_enabled,
        )
        .await
    } else {
        // 非流式响应：仅在配置开启且模型启用 thinking 时提取 thinking 块
        let extract_thinking = state.extract_thinking && thinking_enabled;
        handle_non_stream_request(
            provider,
            &request_body,
            &payload.model,
            input_tokens,
            tool_name_map,
            extract_thinking,
        )
        .await
    }
}

/// 估算输入 tokens
fn estimate_input_tokens(payload: &ChatCompletionRequest) -> i32 {
    let mut total = 0;
    for msg in &payload.messages {
        total += (msg.text_content().len() as f64 / 4.0).ceil() as i32;
    }
    total.max(1)
}

/// POST /v1/responses
pub async fn post_responses(
    State(state): State<AppState>,
    JsonExtractor(payload): JsonExtractor<ResponsesRequest>,
) -> Response {
    tracing::info!(
        model = %payload.model,
        stream = %payload.stream,
        "Received POST /v1/responses request"
    );

    // 检查 KiroProvider 是否可用
    let provider = match &state.kiro_provider {
        Some(p) => p.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::new(
                    "server_error",
                    "Kiro API provider not configured",
                )),
            )
                .into_response();
        }
    };

    // 转换请求（内部归一化为 Chat Completions 格式后复用转换逻辑）
    let conversion_result = match convert_responses_request(&payload) {
        Ok(result) => result,
        Err(e) => {
            let message = match &e {
                ConversionError::UnsupportedModel(model) => {
                    format!("Model not supported: {}", model)
                }
                ConversionError::EmptyMessages => "Messages list is empty".to_string(),
            };
            tracing::warn!("请求转换失败: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("invalid_request_error", message)),
            )
                .into_response();
        }
    };

    // 构建 Kiro 请求
    let kiro_request = KiroRequest {
        conversation_state: conversion_result.conversation_state,
        profile_arn: None,
    };

    let request_body = match serde_json::to_string(&kiro_request) {
        Ok(body) => body,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    "server_error",
                    format!("Failed to serialize request: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 估算输入 tokens
    let input_tokens = estimate_responses_input_tokens(&payload);

    let tool_name_map = conversion_result.tool_name_map;

    // 模型名包含 `-thinking`（大小写不敏感）即视为启用 thinking
    let thinking_enabled = is_thinking_model(&payload.model);

    if payload.stream {
        handle_responses_stream_request(
            provider,
            &request_body,
            &payload.model,
            input_tokens,
            tool_name_map,
            thinking_enabled,
        )
        .await
    } else {
        // 非流式响应：仅在配置开启且模型启用 thinking 时提取 thinking 块
        let extract_thinking = state.extract_thinking && thinking_enabled;
        handle_responses_non_stream_request(
            provider,
            &request_body,
            &payload.model,
            input_tokens,
            tool_name_map,
            extract_thinking,
        )
        .await
    }
}

/// 估算 Responses 请求的输入 tokens（简化版启发式，与 Chat Completions 侧一致）
fn estimate_responses_input_tokens(payload: &ResponsesRequest) -> i32 {
    let mut total_chars = 0usize;

    if let Some(instructions) = &payload.instructions {
        total_chars += instructions.len();
    }

    total_chars += match &payload.input {
        ResponsesInput::Text(text) => text.len(),
        ResponsesInput::Items(items) => serde_json::to_string(items).map(|s| s.len()).unwrap_or(0),
    };

    ((total_chars as f64 / 4.0).ceil() as i32).max(1)
}

/// Ping 事件间隔（25秒）
const PING_INTERVAL_SECS: u64 = 25;

/// 处理流式请求
async fn handle_stream_request(
    provider: std::sync::Arc<crate::kiro::provider::KiroProvider>,
    request_body: &str,
    model: &str,
    input_tokens: i32,
    tool_name_map: std::collections::HashMap<String, String>,
    include_usage: bool,
    thinking_enabled: bool,
) -> Response {
    let api_result = match provider.call_api_stream(request_body).await {
        Ok(resp) => resp,
        Err(e) => return map_provider_error(e),
    };
    let credential_id = api_result.credential_id;
    let response = api_result.response;

    let ctx = OpenAIStreamContext::new_with_thinking(
        model,
        input_tokens,
        tool_name_map,
        include_usage,
        thinking_enabled,
    );

    // 创建 SSE 流
    let body_stream = response.bytes_stream();

    let stream = stream::unfold(
        (
            body_stream,
            ctx,
            EventStreamDecoder::new(),
            false,
            interval(Duration::from_secs(PING_INTERVAL_SECS)),
            provider,
            credential_id,
        ),
        |(mut body_stream, mut ctx, mut decoder, finished, mut ping_interval, provider, credential_id)| async move {
            if finished {
                return None;
            }

            tokio::select! {
                chunk_result = body_stream.next() => {
                    match chunk_result {
                        Some(Ok(chunk)) => {
                            if let Err(e) = decoder.feed(&chunk) {
                                tracing::warn!("缓冲区溢出: {}", e);
                            }

                            let mut events = Vec::new();
                            for result in decoder.decode_iter() {
                                match result {
                                    Ok(frame) => {
                                        if let Ok(event) = Event::from_frame(frame) {
                                            let sse_strings = ctx.process_event(&event);
                                            events.extend(sse_strings);
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!("解码事件失败: {}", e);
                                    }
                                }
                            }

                            let bytes: Vec<Result<Bytes, Infallible>> = events
                                .into_iter()
                                .map(|s| Ok(Bytes::from(s)))
                                .collect();

                            Some((stream::iter(bytes), (body_stream, ctx, decoder, false, ping_interval, provider, credential_id)))
                        }
                        Some(Err(e)) => {
                            tracing::error!("读取响应流失败: {}", e);
                            if let Some(metering) = ctx.metering() {
                                provider.record_credits_used(credential_id, metering);
                            }
                            let final_events = ctx.generate_final_events();
                            let bytes: Vec<Result<Bytes, Infallible>> = final_events
                                .into_iter()
                                .map(|s| Ok(Bytes::from(s)))
                                .collect();
                            Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval, provider, credential_id)))
                        }
                        None => {
                            if let Some(metering) = ctx.metering() {
                                provider.record_credits_used(credential_id, metering);
                            }
                            let final_events = ctx.generate_final_events();
                            let bytes: Vec<Result<Bytes, Infallible>> = final_events
                                .into_iter()
                                .map(|s| Ok(Bytes::from(s)))
                                .collect();
                            Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval, provider, credential_id)))
                        }
                    }
                }
                _ = ping_interval.tick() => {
                    let bytes: Vec<Result<Bytes, Infallible>> = vec![Ok(Bytes::from(": ping\n\n"))];
                    Some((stream::iter(bytes), (body_stream, ctx, decoder, false, ping_interval, provider, credential_id)))
                }
            }
        },
    )
    .flatten();

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// 处理非流式请求
async fn handle_non_stream_request(
    provider: std::sync::Arc<crate::kiro::provider::KiroProvider>,
    request_body: &str,
    model: &str,
    input_tokens: i32,
    tool_name_map: std::collections::HashMap<String, String>,
    extract_thinking: bool,
) -> Response {
    let api_result = match provider.call_api(request_body).await {
        Ok(resp) => resp,
        Err(e) => return map_provider_error(e),
    };
    let credential_id = api_result.credential_id;
    let response = api_result.response;

    let body_bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::new(
                    "server_error",
                    format!("Failed to read response: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 解析事件流
    let mut decoder = EventStreamDecoder::new();
    if let Err(e) = decoder.feed(&body_bytes) {
        tracing::warn!("缓冲区溢出: {}", e);
    }

    let mut events = Vec::new();
    for result in decoder.decode_iter() {
        match result {
            Ok(frame) => {
                if let Ok(event) = Event::from_frame(frame) {
                    events.push(event);
                }
            }
            Err(e) => {
                tracing::warn!("解码事件失败: {}", e);
            }
        }
    }

    if let Some(m) = events.iter().find_map(|e| match e {
        Event::Metering(m) => Some(m),
        _ => None,
    }) {
        provider.record_credits_used(credential_id, m);
    }

    let resp = build_non_stream_response(
        model,
        input_tokens,
        &events,
        &tool_name_map,
        extract_thinking,
    );
    (StatusCode::OK, Json(resp)).into_response()
}

/// 处理 Responses API 流式请求
async fn handle_responses_stream_request(
    provider: std::sync::Arc<crate::kiro::provider::KiroProvider>,
    request_body: &str,
    model: &str,
    input_tokens: i32,
    tool_name_map: std::collections::HashMap<String, String>,
    thinking_enabled: bool,
) -> Response {
    let api_result = match provider.call_api_stream(request_body).await {
        Ok(resp) => resp,
        Err(e) => return map_provider_error(e),
    };
    let credential_id = api_result.credential_id;
    let response = api_result.response;

    let ctx = ResponsesStreamContext::new_with_thinking(
        model,
        input_tokens,
        tool_name_map,
        thinking_enabled,
    );

    // 创建 SSE 流
    let body_stream = response.bytes_stream();

    let stream = stream::unfold(
        (
            body_stream,
            ctx,
            EventStreamDecoder::new(),
            false,
            interval(Duration::from_secs(PING_INTERVAL_SECS)),
            provider,
            credential_id,
        ),
        |(mut body_stream, mut ctx, mut decoder, finished, mut ping_interval, provider, credential_id)| async move {
            if finished {
                return None;
            }

            tokio::select! {
                chunk_result = body_stream.next() => {
                    match chunk_result {
                        Some(Ok(chunk)) => {
                            if let Err(e) = decoder.feed(&chunk) {
                                tracing::warn!("缓冲区溢出: {}", e);
                            }

                            let mut events = Vec::new();
                            for result in decoder.decode_iter() {
                                match result {
                                    Ok(frame) => {
                                        if let Ok(event) = Event::from_frame(frame) {
                                            let sse_strings = ctx.process_event(&event);
                                            events.extend(sse_strings);
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!("解码事件失败: {}", e);
                                    }
                                }
                            }

                            let bytes: Vec<Result<Bytes, Infallible>> = events
                                .into_iter()
                                .map(|s| Ok(Bytes::from(s)))
                                .collect();

                            Some((stream::iter(bytes), (body_stream, ctx, decoder, false, ping_interval, provider, credential_id)))
                        }
                        Some(Err(e)) => {
                            tracing::error!("读取响应流失败: {}", e);
                            if let Some(metering) = ctx.metering() {
                                provider.record_credits_used(credential_id, metering);
                            }
                            let final_events = ctx.generate_final_events();
                            let bytes: Vec<Result<Bytes, Infallible>> = final_events
                                .into_iter()
                                .map(|s| Ok(Bytes::from(s)))
                                .collect();
                            Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval, provider, credential_id)))
                        }
                        None => {
                            if let Some(metering) = ctx.metering() {
                                provider.record_credits_used(credential_id, metering);
                            }
                            let final_events = ctx.generate_final_events();
                            let bytes: Vec<Result<Bytes, Infallible>> = final_events
                                .into_iter()
                                .map(|s| Ok(Bytes::from(s)))
                                .collect();
                            Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval, provider, credential_id)))
                        }
                    }
                }
                _ = ping_interval.tick() => {
                    let bytes: Vec<Result<Bytes, Infallible>> = vec![Ok(Bytes::from(": ping\n\n"))];
                    Some((stream::iter(bytes), (body_stream, ctx, decoder, false, ping_interval, provider, credential_id)))
                }
            }
        },
    )
    .flatten();

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// 处理 Responses API 非流式请求
async fn handle_responses_non_stream_request(
    provider: std::sync::Arc<crate::kiro::provider::KiroProvider>,
    request_body: &str,
    model: &str,
    input_tokens: i32,
    tool_name_map: std::collections::HashMap<String, String>,
    extract_thinking: bool,
) -> Response {
    let api_result = match provider.call_api(request_body).await {
        Ok(resp) => resp,
        Err(e) => return map_provider_error(e),
    };
    let credential_id = api_result.credential_id;
    let response = api_result.response;

    let body_bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::new(
                    "server_error",
                    format!("Failed to read response: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 解析事件流
    let mut decoder = EventStreamDecoder::new();
    if let Err(e) = decoder.feed(&body_bytes) {
        tracing::warn!("缓冲区溢出: {}", e);
    }

    let mut events = Vec::new();
    for result in decoder.decode_iter() {
        match result {
            Ok(frame) => {
                if let Ok(event) = Event::from_frame(frame) {
                    events.push(event);
                }
            }
            Err(e) => {
                tracing::warn!("解码事件失败: {}", e);
            }
        }
    }

    if let Some(m) = events.iter().find_map(|e| match e {
        Event::Metering(m) => Some(m),
        _ => None,
    }) {
        provider.record_credits_used(credential_id, m);
    }

    let resp = build_responses_non_stream(
        model,
        input_tokens,
        &events,
        &tool_name_map,
        extract_thinking,
    );

    // 空补全（无文本、无 function_call、无 reasoning）视为上游异常
    if resp.output.is_empty() {
        tracing::error!("上游返回空补全（无文本/function_call/reasoning）");
        return (
            StatusCode::BAD_GATEWAY,
            Json(ErrorResponse::new(
                "api_error",
                "Upstream returned an empty completion",
            )),
        )
            .into_response();
    }

    (StatusCode::OK, Json(resp)).into_response()
}
