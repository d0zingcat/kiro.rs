//! Anthropic API 兼容服务模块
//!
//! 提供与 Anthropic Claude API 兼容的 HTTP 服务端点。
//!
//! # 支持的端点
//!
//! ## 标准端点 (/v1)
//! - `GET /v1/models` - 获取可用模型列表
//! - `POST /v1/messages` - 创建消息（对话）
//! - `POST /v1/messages/count_tokens` - 计算 token 数量
//!
//! ## Claude Code 兼容端点 (/cc/v1)
//! - `POST /cc/v1/messages` - 创建消息（流式响应会等待 contextUsageEvent 后再发送 message_start，确保 input_tokens 准确）
//! - `POST /cc/v1/messages/count_tokens` - 计算 token 数量（与 /v1 相同）
//!
//! ## 响应 usage 扩展字段
//!
//! 除 Anthropic 标准的 `input_tokens` / `output_tokens` 外，若上游返回 `meteringEvent`，
//! 响应 `usage` 还会包含：
//! - `credits` (`number`) - 本次请求消耗的 Kiro credits
//! - `metering_unit` (`string`) - 计费单位，通常为 `"credit"`
//!
//! credits 与 token 是不同计量单位，不会互相替换。详见 README「用量字段 (usage) 与 Credits」。
//!
//! # 使用示例
//! ```rust,ignore
//! use kiro_rs::anthropic;
//!
//! let app = anthropic::create_router("your-api-key");
//! let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
//! axum::serve(listener, app).await?;
//! ```

mod converter;
mod handlers;
mod middleware;
mod router;
mod stream;
pub mod types;
mod usage;
mod websearch;

pub use router::create_router_with_provider;
