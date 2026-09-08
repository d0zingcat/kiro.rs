//! OpenAI API 路由配置

use axum::{
    Router,
    extract::DefaultBodyLimit,
    middleware,
    routing::post,
};

use super::handlers::{post_chat_completions, post_responses};
use super::middleware::{AppState, auth_middleware};

/// 请求体最大大小限制 (50MB)，与 Anthropic 路由对齐；Codex 长上下文会超过 Axum 默认 2MB
const MAX_BODY_SIZE: usize = 50 * 1024 * 1024;

/// 创建 OpenAI API 路由（接受已构建的 AppState）
pub fn create_router(state: AppState) -> Router {
    let v1 = Router::new()
        .route("/chat/completions", post(post_chat_completions))
        .route("/responses", post(post_responses))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .layer(DefaultBodyLimit::max(MAX_BODY_SIZE));

    Router::new().nest("/v1", v1).with_state(state)
}
