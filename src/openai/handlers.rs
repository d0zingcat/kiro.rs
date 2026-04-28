//! OpenAI API Handler 函数

use std::convert::Infallible;

use crate::kiro::model::events::Event;
use crate::kiro::model::requests::kiro::KiroRequest;
use crate::kiro::parser::decoder::EventStreamDecoder;
use crate::token;
use axum::{
    Json as JsonExtractor,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
};
use bytes::Bytes;
use futures::{Stream, StreamExt, stream};
use std::time::Duration;
use tokio::time::interval;

use crate::anthropic::middleware::AppState;

use super::converter::{ConversionError, convert_request};
use super::stream::{OpenAIStreamContext, build_non_stream_response};
use super::types::{
    ChatCompletionRequest, ErrorResponse, Model, ModelsResponse,
};

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

/// GET /openai/v1/models
pub async fn get_models() -> impl IntoResponse {
    tracing::info!("Received GET /openai/v1/models request");

    let models = vec![
        Model {
            id: "claude-opus-4-6".into(),
            object: "model".into(),
            created: 1770163200,
            owned_by: "anthropic".into(),
        },
        Model {
            id: "claude-opus-4-6-thinking".into(),
            object: "model".into(),
            created: 1770163200,
            owned_by: "anthropic".into(),
        },
        Model {
            id: "claude-sonnet-4-6".into(),
            object: "model".into(),
            created: 1771286400,
            owned_by: "anthropic".into(),
        },
        Model {
            id: "claude-sonnet-4-6-thinking".into(),
            object: "model".into(),
            created: 1771286400,
            owned_by: "anthropic".into(),
        },
        Model {
            id: "claude-opus-4-5-20251101".into(),
            object: "model".into(),
            created: 1763942400,
            owned_by: "anthropic".into(),
        },
        Model {
            id: "claude-sonnet-4-5-20250929".into(),
            object: "model".into(),
            created: 1759104000,
            owned_by: "anthropic".into(),
        },
        Model {
            id: "claude-haiku-4-5-20251001".into(),
            object: "model".into(),
            created: 1760486400,
            owned_by: "anthropic".into(),
        },
    ];

    Json(ModelsResponse {
        object: "list".into(),
        data: models,
    })
}

/// POST /openai/v1/chat/completions
pub async fn post_chat_completions(
    State(state): State<AppState>,
    JsonExtractor(payload): JsonExtractor<ChatCompletionRequest>,
) -> Response {
    tracing::info!(
        model = %payload.model,
        stream = %payload.stream,
        message_count = %payload.messages.len(),
        "Received POST /openai/v1/chat/completions request"
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

    if payload.stream {
        handle_stream_request(
            provider,
            &request_body,
            &payload.model,
            input_tokens,
            tool_name_map,
            include_usage,
        )
        .await
    } else {
        handle_non_stream_request(
            provider,
            &request_body,
            &payload.model,
            input_tokens,
            tool_name_map,
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
) -> Response {
    let response = match provider.call_api_stream(request_body).await {
        Ok(resp) => resp,
        Err(e) => return map_provider_error(e),
    };

    let mut ctx = OpenAIStreamContext::new(model, input_tokens, tool_name_map, include_usage);

    // 创建 SSE 流
    let body_stream = response.bytes_stream();

    let stream = stream::unfold(
        (
            body_stream,
            ctx,
            EventStreamDecoder::new(),
            false,
            interval(Duration::from_secs(PING_INTERVAL_SECS)),
        ),
        |(mut body_stream, mut ctx, mut decoder, finished, mut ping_interval)| async move {
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

                            Some((stream::iter(bytes), (body_stream, ctx, decoder, false, ping_interval)))
                        }
                        Some(Err(e)) => {
                            tracing::error!("读取响应流失败: {}", e);
                            let final_events = ctx.generate_final_events();
                            let bytes: Vec<Result<Bytes, Infallible>> = final_events
                                .into_iter()
                                .map(|s| Ok(Bytes::from(s)))
                                .collect();
                            Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval)))
                        }
                        None => {
                            let final_events = ctx.generate_final_events();
                            let bytes: Vec<Result<Bytes, Infallible>> = final_events
                                .into_iter()
                                .map(|s| Ok(Bytes::from(s)))
                                .collect();
                            Some((stream::iter(bytes), (body_stream, ctx, decoder, true, ping_interval)))
                        }
                    }
                }
                _ = ping_interval.tick() => {
                    let bytes: Vec<Result<Bytes, Infallible>> = vec![Ok(Bytes::from(": ping\n\n"))];
                    Some((stream::iter(bytes), (body_stream, ctx, decoder, false, ping_interval)))
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
) -> Response {
    let response = match provider.call_api(request_body).await {
        Ok(resp) => resp,
        Err(e) => return map_provider_error(e),
    };

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

    let resp = build_non_stream_response(model, input_tokens, &events, &tool_name_map);
    (StatusCode::OK, Json(resp)).into_response()
}
