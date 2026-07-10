//! OpenAI API 兼容服务模块
//!
//! 提供与 OpenAI Chat Completions API 兼容的 HTTP 服务端点。
//!
//! # 支持的端点
//!
//! - `GET /openai/v1/models` - 获取可用模型列表
//! - `POST /openai/v1/chat/completions` - Chat Completions（对话）

mod converter;
mod handlers;
pub mod middleware;
mod router;
mod stream;
pub mod types;

pub use router::create_router;

#[cfg(test)]
#[path = "isolation_tests.rs"]
mod isolation_tests;
