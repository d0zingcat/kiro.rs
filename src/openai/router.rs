//! OpenAI API 路由配置

use axum::{
    Router,
    middleware,
    routing::{get, post},
};

use crate::anthropic::middleware::{AppState, auth_middleware};

use super::handlers::{get_models, post_chat_completions};

/// 创建 OpenAI API 路由（接受已构建的 AppState）
pub fn create_router(state: AppState) -> Router {
    let openai_v1_routes = Router::new()
        .route("/models", get(get_models))
        .route("/chat/completions", post(post_chat_completions))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    Router::new()
        .nest("/openai/v1", openai_v1_routes)
        .with_state(state)
}
