//! OpenAI API 路由配置

use axum::{Router, middleware, routing::post};

use super::handlers::{post_chat_completions, post_responses};
use super::middleware::{AppState, auth_middleware};

/// 创建 OpenAI API 路由（接受已构建的 AppState）
pub fn create_router(state: AppState) -> Router {
    let v1 = Router::new()
        .route("/chat/completions", post(post_chat_completions))
        .route("/responses", post(post_responses))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    Router::new().nest("/v1", v1).with_state(state)
}
