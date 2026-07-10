# OpenAI Compatible API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close gaps on `feat/openai-compatible-api` so OpenAI clients can use `POST /v1/chat/completions` and `POST /v1/responses` (stream + non-stream) via an independent OpenAI→Kiro pipeline with tools, thinking (`reasoning_content` / `reasoning`), and usage credits — without importing `crate::anthropic`.

**Architecture:** Keep existing `src/openai/{converter,stream,handlers,types}` as the Chat Completions core. Add openai-owned middleware/state, remount under `/v1`, restore `MeteringEvent` parsing from `origin/main`, add thinking + credits to the Chat path, then add Responses as a second view over the same converter/event accumulator. Never call Anthropic converter/stream.

**Tech Stack:** Rust, axum, serde/serde_json, futures streams, existing `KiroProvider` + `EventStreamDecoder`, `crate::common::{auth,converter}`.

**Spec:** [`docs/superpowers/specs/2026-07-10-openai-compatible-api-design.md`](../specs/2026-07-10-openai-compatible-api-design.md)

---

## File map

| File | Action |
|------|--------|
| `src/openai/middleware.rs` | **Create** — openai `AppState` + OpenAI-shaped auth |
| `src/openai/router.rs` | **Modify** — `/v1/chat/completions`, `/v1/responses`; drop `/openai/v1` and openai `/models` |
| `src/openai/handlers.rs` | **Modify** — use openai state; add Responses handlers; drop `get_models` |
| `src/openai/types.rs` | **Modify** — `reasoning_content`, credits on `Usage`; add Responses types |
| `src/openai/converter.rs` | **Modify** — thinking prefix for `-thinking`; Responses→Chat normalize helper |
| `src/openai/stream.rs` | **Modify** — thinking split; credits; Responses SSE/JSON builders |
| `src/openai/thinking.rs` | **Create** — `<thinking>` parse helpers (copied/adapted, not imported from anthropic) |
| `src/openai/usage.rs` | **Create** — build OpenAI usage JSON (+ optional credits) |
| `src/openai/mod.rs` | **Modify** — declare new modules; update docs |
| `src/main.rs` | **Modify** — dual AppState; logs for `/v1/chat/completions` + `/v1/responses` |
| `src/kiro/model/events/metering.rs` | **Create** — port from `origin/main` |
| `src/kiro/model/events/base.rs` | **Modify** — `Event::Metering(MeteringEvent)` |
| `src/kiro/model/events/mod.rs` | **Modify** — export `MeteringEvent` |
| `src/anthropic/middleware.rs` | **Modify** — accept `Arc<KiroProvider>` so main can share one provider |
| `README.md` | **Modify** — document OpenAI endpoints |

---

### Task 1: Decouple OpenAI from `anthropic::middleware`

**Files:**
- Create: `src/openai/middleware.rs`
- Modify: `src/openai/router.rs`, `src/openai/handlers.rs`, `src/openai/mod.rs`
- Modify: `src/anthropic/middleware.rs` (add `with_kiro_provider_arc`)
- Modify: `src/main.rs`

- [ ] **Step 1: Write failing isolation test**

Add to `src/openai/mod.rs` (or a tiny `src/openai/isolation_tests.rs` included from mod):

```rust
#[cfg(test)]
mod isolation_tests {
    #[test]
    fn openai_sources_must_not_import_anthropic() {
        let openai_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/openai");
        for entry in std::fs::read_dir(&openai_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).unwrap();
            assert!(
                !src.contains("crate::anthropic"),
                "{} must not import crate::anthropic",
                path.display()
            );
        }
    }
}
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cargo test --no-default-features openai::isolation_tests -- --nocapture
```

Expected: FAIL mentioning `handlers.rs` or `router.rs` importing `crate::anthropic`.

- [ ] **Step 3: Add openai middleware**

Create `src/openai/middleware.rs`:

```rust
use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Json, Response},
};

use crate::common::auth;
use crate::kiro::provider::KiroProvider;

use super::types::ErrorResponse;

#[derive(Clone)]
pub struct AppState {
    pub api_key: String,
    pub kiro_provider: Option<Arc<KiroProvider>>,
    pub extract_thinking: bool,
}

impl AppState {
    pub fn new(api_key: impl Into<String>, extract_thinking: bool) -> Self {
        Self {
            api_key: api_key.into(),
            kiro_provider: None,
            extract_thinking,
        }
    }

    pub fn with_kiro_provider(mut self, provider: Arc<KiroProvider>) -> Self {
        self.kiro_provider = Some(provider);
        self
    }
}

pub async fn auth_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    match auth::extract_api_key(&request) {
        Some(key) if auth::constant_time_eq(&key, &state.api_key) => next.run(request).await,
        _ => {
            let error = ErrorResponse::authentication_error();
            (StatusCode::UNAUTHORIZED, Json(error)).into_response()
        }
    }
}
```

Ensure `ErrorResponse::authentication_error` is public/usable (already `#[allow(dead_code)]` on branch — remove allow once used).

- [ ] **Step 4: Teach Anthropic AppState to accept `Arc<KiroProvider>`**

In `src/anthropic/middleware.rs`, change `with_kiro_provider` to take `Arc<KiroProvider>` (or add `with_kiro_provider_arc` and keep old method wrapping `Arc::new`). Prefer single method:

```rust
pub fn with_kiro_provider(mut self, provider: Arc<KiroProvider>) -> Self {
    self.kiro_provider = Some(provider);
    self
}
```

Update any Anthropic call sites that passed owned `KiroProvider` to `Arc::new(...)`.

- [ ] **Step 5: Wire router/handlers/main**

- `openai/mod.rs`: `mod middleware;`
- `openai/router.rs` + `handlers.rs`: `use super::middleware::AppState` and `auth_middleware` (delete anthropic imports)
- `main.rs`:

```rust
let kiro_provider = Arc::new(KiroProvider::with_proxy(/* existing args */));

let anthropic_state = anthropic::middleware::AppState::new(&api_key, config.extract_thinking)
    .with_kiro_provider(kiro_provider.clone());
let openai_state = openai::middleware::AppState::new(&api_key, config.extract_thinking)
    .with_kiro_provider(kiro_provider);

let anthropic_app = anthropic::create_router_with_state(anthropic_state);
let openai_app = openai::create_router(openai_state);
```

- [ ] **Step 6: Re-run isolation test + build**

```bash
cargo test --no-default-features openai::isolation_tests
cargo build --no-default-features
```

Expected: PASS / compile OK.

- [ ] **Step 7: Commit**

```bash
git add src/openai/middleware.rs src/openai/router.rs src/openai/handlers.rs src/openai/mod.rs src/openai/types.rs src/anthropic/middleware.rs src/anthropic/router.rs src/main.rs
git commit -m "$(cat <<'EOF'
refactor(openai): 使用独立 AppState，解除对 anthropic middleware 的依赖

EOF
)"
```

---

### Task 2: Remount Chat Completions on `/v1`

**Files:**
- Modify: `src/openai/router.rs`, `src/openai/handlers.rs`, `src/openai/mod.rs`, `src/main.rs`, `README.md`

- [ ] **Step 1: Change router nest path and drop models route**

`src/openai/router.rs` target:

```rust
use axum::{Router, middleware, routing::post};
use super::handlers::post_chat_completions;
use super::middleware::{AppState, auth_middleware};

pub fn create_router(state: AppState) -> Router {
    let v1 = Router::new()
        .route("/chat/completions", post(post_chat_completions))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    Router::new().nest("/v1", v1).with_state(state)
}
```

Remove `get_models` from `handlers.rs` and module docs that mention `/openai/v1/models`.

- [ ] **Step 2: Update main logs**

```rust
tracing::info!("OpenAI 兼容 API:");
tracing::info!("  POST /v1/chat/completions");
```

Do **not** register a second `/v1/models` — Anthropic already owns it.

- [ ] **Step 3: Update README API table**

Document `POST /v1/chat/completions` under OpenAI-compatible section; note shared `GET /v1/models`.

- [ ] **Step 4: Build + existing openai tests**

```bash
cargo test --no-default-features openai::
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(openai): 将 Chat Completions 挂到 /v1/chat/completions

EOF
)"
```

---

### Task 3: Restore `MeteringEvent` parsing (credits prerequisite)

This branch stubs `Event::Metering(())`. Port from `origin/main`.

**Files:**
- Create: `src/kiro/model/events/metering.rs` (copy from `git show origin/main:src/kiro/model/events/metering.rs`)
- Modify: `src/kiro/model/events/mod.rs`, `src/kiro/model/events/base.rs`
- Fix any match arms that assumed `Metering(())` (grep the repo)

- [ ] **Step 1: Add unit test for metering deserialize** (comes with ported file)

- [ ] **Step 2: Wire `Event::Metering(MeteringEvent)`**

In `base.rs` parse arm:

```rust
EventType::Metering => {
    let payload = super::MeteringEvent::from_frame(&frame)?;
    Ok(Self::Metering(payload))
}
```

Export from `mod.rs`: `pub use metering::MeteringEvent;`

- [ ] **Step 3: Fix compile errors from enum change**

```bash
rg 'Event::Metering' -n src
cargo build --no-default-features
```

Update match arms to `Event::Metering(m)` / `_`.

- [ ] **Step 4: Run metering tests**

```bash
cargo test --no-default-features metering -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(kiro): 恢复 meteringEvent 解析以支持 credits

EOF
)"
```

---

### Task 4: OpenAI usage helper + credits fields

**Files:**
- Create: `src/openai/usage.rs`
- Modify: `src/openai/types.rs` (`Usage` add optional `credits` / `metering_unit`)
- Modify: `src/openai/mod.rs`

- [ ] **Step 1: Failing tests in `usage.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::model::events::MeteringEvent;

    #[test]
    fn usage_without_metering() {
        let u = build_usage(10, 5, None);
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 15);
        assert!(u.credits.is_none());
    }

    #[test]
    fn usage_with_credits() {
        let m = MeteringEvent {
            unit: Some("credit".into()),
            unit_plural: Some("credits".into()),
            usage: 0.01,
        };
        let u = build_usage(10, 5, Some(&m));
        assert_eq!(u.credits, Some(0.01));
        assert_eq!(u.metering_unit.as_deref(), Some("credit"));
    }
}
```

- [ ] **Step 2: Implement**

```rust
use crate::kiro::model::events::MeteringEvent;
use super::types::Usage;

pub fn build_usage(
    prompt_tokens: i32,
    completion_tokens: i32,
    metering: Option<&MeteringEvent>,
) -> Usage {
    let mut usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        credits: None,
        metering_unit: None,
    };
    if let Some(m) = metering {
        if m.is_credit() {
            usage.credits = Some(m.usage);
        }
        usage.metering_unit = m.unit.clone();
    }
    usage
}
```

Update `Usage` in `types.rs`:

```rust
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
```

- [ ] **Step 3: Wire stream + non-stream to capture `Event::Metering` and call `build_usage`**

In `OpenAIStreamContext` and `build_non_stream_response`, store `Option<MeteringEvent>` and pass into `build_usage` for final usage chunks / JSON.

- [ ] **Step 4: Test**

```bash
cargo test --no-default-features openai::usage
cargo test --no-default-features openai::stream
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(openai): usage 支持 credits / metering_unit

EOF
)"
```

---

### Task 5: Thinking → `reasoning_content` (Chat Completions)

**Files:**
- Create: `src/openai/thinking.rs` (adapt logic from `src/anthropic/stream.rs`: `find_real_thinking_start_tag`, `extract_thinking_from_complete_text`, streaming buffer — **copy code, do not import anthropic**)
- Modify: `src/openai/types.rs` — add `reasoning_content` to `ResponseMessage` and `Delta`
- Modify: `src/openai/stream.rs` — split assistant text when thinking enabled
- Modify: `src/openai/converter.rs` — inject thinking mode prefix when model ends with `-thinking`
- Modify: `src/openai/handlers.rs` — pass `thinking_enabled` / `extract_thinking` into stream/non-stream

- [ ] **Step 1: Failing tests for thinking helpers**

```rust
#[test]
fn extract_thinking_basic() {
    let (t, rest) = extract_thinking_from_complete_text("<thinking>\nabc</thinking>\n\nhello");
    assert_eq!(t.as_deref(), Some("abc"));
    assert_eq!(rest, "hello");
}

#[test]
fn quoted_fake_tag_ignored() {
    assert!(find_real_thinking_start_tag("use `<thinking>` then<thinking>x").is_some());
}
```

- [ ] **Step 2: Implement `thinking.rs` by adapting Anthropic helpers**

Keep the same tag rules (`<thinking>` / `</thinking>`, skip backtick/quote wrapped fakes). Export:

- `find_real_thinking_start_tag`
- `extract_thinking_from_complete_text`
- `ThinkingStreamParser` with `push(&str) -> Vec<ThinkingDelta>` where `ThinkingDelta` is `Reasoning(String)` or `Content(String)`

- [ ] **Step 3: Converter — enable thinking for `-thinking` models**

In `convert_request`, after mapping model:

```rust
let thinking_enabled = req.model.to_lowercase().contains("-thinking");
// When building first system/user history content, if thinking_enabled and no existing
// <thinking_mode> tag, prepend:
// <thinking_mode>enabled</thinking_mode><max_thinking_length>20000</max_thinking_length>
```

Mirror Anthropic’s `generate_thinking_prefix` + `has_thinking_tags` behavior for the system message path already used in openai converter.

Add test:

```rust
#[test]
fn thinking_model_injects_prefix() {
    let req = /* user message, model claude-sonnet-4-6-thinking */;
    let result = convert_request(&req).unwrap();
    let history_text = /* stringify first history user/system content */;
    assert!(history_text.contains("<thinking_mode>enabled</thinking_mode>"));
}
```

- [ ] **Step 4: Stream/non-stream emit `reasoning_content`**

- Extend `Delta` / `ResponseMessage` with `reasoning_content: Option<String>`
- When `thinking_enabled`, run parser on `AssistantResponse` text; emit separate chunks for reasoning vs content
- Non-stream: `extract_thinking_from_complete_text` when `state.extract_thinking && thinking_enabled` (same gate as Anthropic non-stream)

- [ ] **Step 5: Tests**

```bash
cargo test --no-default-features openai::thinking
cargo test --no-default-features openai::stream
cargo test --no-default-features openai::converter
```

- [ ] **Step 6: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(openai): Chat Completions 支持 thinking / reasoning_content

EOF
)"
```

---

### Task 6: Responses API types + request normalization

**Files:**
- Modify: `src/openai/types.rs`
- Modify: `src/openai/converter.rs` — `responses_to_chat_request` or `convert_responses_request`

- [ ] **Step 1: Add Responses request/response types (minimal)**

```rust
#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<serde_json::Value>),
}
```

Non-stream response skeleton:

```rust
#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String, // "response"
    pub created_at: i64,
    pub status: String, // "completed"
    pub model: String,
    pub output: Vec<serde_json::Value>,
    pub usage: ResponsesUsage,
}
```

`ResponsesUsage` mirrors token fields + optional credits (can reuse `Usage` or a thin alias).

- [ ] **Step 2: Failing normalize tests**

```rust
#[test]
fn responses_text_input_becomes_user_message() {
    let req = ResponsesRequest {
        model: "claude-sonnet-4-6".into(),
        input: ResponsesInput::Text("hi".into()),
        instructions: Some("be brief".into()),
        stream: false,
        tools: None,
        user: None,
    };
    let chat = normalize_responses_to_chat(&req).unwrap();
    assert_eq!(chat.messages[0].role, "system");
    assert_eq!(chat.messages[1].role, "user");
    assert_eq!(chat.messages[1].text_content(), "hi");
}

#[test]
fn responses_function_call_output_becomes_tool_message() {
    // input item: {"type":"function_call_output","call_id":"call_1","output":"{}"}
    // plus prior function_call assistant item if present in array
}
```

Normalization rules (v1):

| Responses input | Chat message |
|-----------------|--------------|
| `instructions` | leading `system` |
| `input` string | one `user` |
| item `message` role user/assistant | same role; text from `content` parts `input_text`/`output_text`/`text` |
| item `function_call` | assistant with `tool_calls` |
| item `function_call_output` | `tool` with `tool_call_id` = `call_id` |
| ignore unknown item types with `tracing::warn` |

Then `convert_request(&chat)` reuses existing Chat→Kiro converter.

- [ ] **Step 3: Implement normalize + `convert_responses_request`**

```rust
pub fn convert_responses_request(req: &ResponsesRequest) -> Result<ConversionResult, ConversionError> {
    let chat = normalize_responses_to_chat(req)?;
    convert_request(&chat)
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test --no-default-features openai::converter::tests -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(openai): 添加 Responses 请求类型与 Chat 归一化

EOF
)"
```

---

### Task 7: Responses stream + non-stream views

**Files:**
- Modify: `src/openai/stream.rs` (or create `src/openai/responses_stream.rs` if `stream.rs` exceeds ~600 lines — prefer split)

**Minimal SSE event set (emit in order):**

1. `response.created`
2. `response.output_item.added` (reasoning item if any)
3. `response.reasoning_text.delta` / `response.reasoning_text.done` (if thinking; if official name differs in current OpenAI docs, match docs — fallback: put reasoning in a `reasoning` output item and stream text deltas on that item)
4. `response.output_item.added` (message)
5. `response.content_part.added`
6. `response.output_text.delta` (many)
7. `response.output_text.done` / `response.content_part.done` / `response.output_item.done`
8. For tools: `response.output_item.added` (`function_call`) + `response.function_call_arguments.delta` + `...done`
9. `response.completed` (include final `response` object with `usage`)
10. Terminal `data: [DONE]`

Each SSE frame: `event: <type>\ndata: <json>\n\n` **or** OpenAI’s common form `data: {"type":"...","delta":...}\n\n` — match what the official Node/Python SDK expects (prefer `data:` JSON with `"type"` field, as SDK iterates `event.type`).

- [ ] **Step 1: Unit-test event sequence for plain text**

Feed synthetic `Event::AssistantResponse` chunks into `ResponsesStreamContext`; assert joined SSE contains `response.created`, `response.output_text.delta`, `response.completed`.

- [ ] **Step 2: Implement `ResponsesStreamContext` + `build_responses_non_stream`**

Reuse the same thinking parser and tool_name_map / metering capture as Chat.

Non-stream `output` array example:

```json
[
  {"type":"reasoning","id":"rs_...","summary":[{"type":"summary_text","text":"..."}]},
  {"type":"message","id":"msg_...","role":"assistant","content":[{"type":"output_text","text":"..."}]},
  {"type":"function_call","id":"fc_...","call_id":"call_...","name":"...","arguments":"{...}"}
]
```

(If `reasoning.summary` shape is awkward, use a simpler `{"type":"reasoning","content":[{"type":"reasoning_text","text":"..."}]}` consistent with current public Responses schemas — verify against OpenAI docs once while implementing and lock the chosen shape in tests.)

- [ ] **Step 3: Tests for tool_calls + reasoning paths**

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(openai): Responses 流式/非流式视图适配

EOF
)"
```

---

### Task 8: Responses handler + route

**Files:**
- Modify: `src/openai/handlers.rs`, `src/openai/router.rs`, `src/main.rs`, `README.md`

- [ ] **Step 1: Add `post_responses` mirroring `post_chat_completions`**

```rust
pub async fn post_responses(
    State(state): State<AppState>,
    JsonExtractor(payload): JsonExtractor<ResponsesRequest>,
) -> Response {
    let provider = match &state.kiro_provider {
        Some(p) => p.clone(),
        None => { /* 503 OpenAI error */ }
    };
    let conversion = match convert_responses_request(&payload) { ... };
    // serialize KiroRequest, estimate tokens, branch stream/non-stream
    // stream → ResponsesStreamContext SSE
    // non-stream → build_responses_non_stream
}
```

Empty completion (no text, no function_call, no reasoning) → 502 `api_error`.

- [ ] **Step 2: Register route**

```rust
.route("/responses", post(post_responses))
```

- [ ] **Step 3: Update logs + README**

```rust
tracing::info!("  POST /v1/responses");
```

- [ ] **Step 4: Build + test**

```bash
cargo test --no-default-features openai::
cargo build --no-default-features
```

- [ ] **Step 5: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(openai): 添加 POST /v1/responses 端点

EOF
)"
```

---

### Task 9: Final verification + docs polish

- [ ] **Step 1: Isolation + full test suite**

```bash
cargo test --no-default-features
rg 'crate::anthropic' src/openai && echo 'FAIL: anthropic import found' || echo 'OK'
```

Expected: all tests pass; `rg` finds no matches under `src/openai`.

- [ ] **Step 2: Manual smoke (optional if credentials available)**

```bash
# Chat non-stream
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"ping"}]}'

# Responses stream
curl -N http://127.0.0.1:$PORT/v1/responses \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","input":"ping","stream":true}'
```

- [ ] **Step 3: README success-criteria checklist**

Confirm documented: stream/non-stream both APIs, tools, `-thinking` / `reasoning_content`, credits, no Anthropic coupling.

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
docs: 完善 OpenAI 兼容 API 说明并完成验收核对

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Independent openai pipeline / no anthropic import | 1, 9 |
| `POST /v1/chat/completions` stream+non-stream | 2 (mount), existing handlers + 4–5 |
| `POST /v1/responses` stream+non-stream stateless | 6–8 |
| Tools / function calling | existing converter/stream; covered in 7–8 |
| Thinking → `reasoning_content` / Responses reasoning | 5, 7 |
| Usage + credits | 3, 4 |
| Shared `/v1/models` (no openai duplicate) | 2 |
| Own middleware / OpenAI errors | 1 |
| README | 2, 8, 9 |

## Out of scope (do not implement)

Images, stateful Responses, Anthropic code reuse via `use crate::anthropic`, embeddings/audio/assistants, `/cc` OpenAI variant.
