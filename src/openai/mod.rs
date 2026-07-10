//! OpenAI API 兼容服务模块
//!
//! 提供与 OpenAI Chat Completions API 兼容的 HTTP 服务端点。
//!
//! # 支持的端点
//!
//! - `POST /v1/chat/completions` - Chat Completions（对话）
//!
//! 模型列表由 Anthropic 兼容端点 `GET /v1/models` 提供，不在此模块重复注册。

mod converter;
mod handlers;
pub mod middleware;
mod responses_stream;
mod router;
mod stream;
mod thinking;
pub mod types;
mod usage;

pub use router::create_router;

#[cfg(test)]
#[path = "isolation_tests.rs"]
mod isolation_tests;
