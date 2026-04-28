---
title: "feat: 添加 OpenAI 兼容 API 支持"
type: feat
status: active
date: 2026-04-28
---

# feat: 添加 OpenAI 兼容 API 支持

## Overview

kiro-rs 目前仅暴露 Anthropic API 兼容端点（`/v1/messages`），不支持 OpenAI Chat Completions 格式（`/v1/chat/completions`）。许多工具和框架（如 OpenCode、Cursor、Continue、LiteLLM 等）原生使用 OpenAI 格式。本计划为 kiro-rs 添加 OpenAI 兼容层，使其能同时服务两种 API 格式的客户端。

---

## Problem Frame

用户希望通过 OpenAI 兼容的 `/v1/chat/completions` 端点访问 Kiro 后端的 Claude 模型。当前项目只实现了 Anthropic → Kiro 的转换链路，缺少 OpenAI → Kiro 的转换能力。需要新增一套 OpenAI 兼容的请求/响应类型定义、协议转换器和路由处理器。

---

## Requirements Trace

- R1. 支持 `POST /v1/chat/completions` 端点，接受 OpenAI Chat Completions 格式请求
- R2. 支持流式（SSE）和非流式两种响应模式
- R3. 支持 `tools` / `tool_choice` (function calling) 的请求和响应转换
- R4. 支持 `GET /v1/models` 返回 OpenAI 格式的模型列表
- R5. 复用现有的 Kiro Provider 和凭据管理基础设施
- R6. 复用现有的认证中间件（`x-api-key` / `Bearer`）
- R7. 支持 `system` / `developer` / `user` / `assistant` / `tool` 角色映射
- R8. 支持 thinking 模式（通过模型名后缀 `-thinking` 触发）

---

## Scope Boundaries

- 不实现 OpenAI Assistants API、Embeddings、Images、Audio 等非 Chat Completions 端点
- 不实现 `n > 1`（多候选回复）— Kiro 后端不支持
- 不实现 `logprobs` — Kiro 后端不支持
- 不实现 OpenAI Responses API（`/v1/responses`）
- WebSearch 工具的 OpenAI 格式适配暂不实现（可后续扩展）

### Deferred to Follow-Up Work

- OpenAI Responses API 支持：未来迭代
- `stream_options.include_usage` 的精确 usage 统计：可后续优化

---

## Context & Research

### Relevant Code and Patterns

- `src/anthropic/` — 现有的 Anthropic 兼容层，是本次工作的直接参考模板
  - `types.rs` — 请求/响应类型定义（手写 serde 结构体）
  - `router.rs` — Axum 路由定义，使用 `Router::new().nest("/v1", ...)` 模式
  - `handlers.rs` — 请求处理器，包含流式/非流式两种处理路径
  - `converter.rs` — Anthropic → Kiro 协议转换器，包含模型映射、消息转换、工具转换
  - `stream.rs` — Kiro 事件流 → Anthropic SSE 转换（~1989 行）
  - `middleware.rs` — 认证中间件和 `AppState`
- `src/kiro/provider.rs` — 上游 API 调用，支持多凭据故障转移
- `src/kiro/parser/` — AWS Event Stream 二进制解码器
- `src/main.rs:161-165` — 路由组装入口

### 库选型调研

| 库 | Stars | 下载量 | 最近更新 | 特点 |
|---|---|---|---|---|
| `async-openai` | ~1,900 | 4.55M | 2026-04-26 | 支持 `chat-completion-types` feature flag（仅类型，不含 HTTP 客户端） |
| `openai-api-rs` | 484 | 572K | 2026-04-17 | 无 types-only feature，全量依赖 |
| `openai-rust` | - | 25K | 2023-12 | 已废弃 |

**决策：手写类型定义**（见 Key Technical Decisions）

---

## Key Technical Decisions

- **手写 OpenAI 类型而非引入 `async-openai` 依赖**：项目已有手写 Anthropic 类型的先例（`src/anthropic/types.rs`，283 行），OpenAI Chat Completions 的类型面也不大（约 15-20 个结构体）。手写可以：(1) 避免引入 `derive_builder` 等不需要的依赖；(2) 精确控制 `Option<serde_json::Value>` 的使用以实现最大兼容性；(3) 保持项目风格一致。如果未来需要支持更多 OpenAI 端点，可以重新评估引入 `async-openai`
- **新建 `src/openai/` 模块，与 `src/anthropic/` 平行**：保持架构对称，两个兼容层独立演进，共享 `src/kiro/` 后端
- **OpenAI → Kiro 直接转换，不经过 Anthropic 中间层**：避免双重转换的性能损耗和信息丢失。复用 `src/anthropic/converter.rs` 中的核心逻辑（模型映射、工具名缩短等），但 OpenAI converter 直接产出 Kiro `ConversationState`
- **OpenAI `/v1/models` 和 Anthropic `/v1/models` 共存**：OpenAI 格式的 models 端点挂载在 `/openai/v1/models`（或通过 `Accept` header 区分），避免路径冲突。或者更简单地，在同一个 `/v1/models` 端点返回兼容两种格式的响应（OpenAI 格式是 Anthropic 格式的超集）
- **路由挂载方案**：OpenAI 端点挂载在 `/openai/v1/chat/completions` 和 `/openai/v1/models`，与现有 Anthropic 端点 `/v1/messages` 互不干扰

---

## Open Questions

### Resolved During Planning

- **Q: 是否使用 `async-openai` 库？** → 不使用，手写类型。理由见 Key Technical Decisions
- **Q: OpenAI 端点路径如何避免与 Anthropic 端点冲突？** → 使用 `/openai/v1/` 前缀

### Deferred to Implementation

- **Q: thinking 内容在 OpenAI 格式中如何表达？** → OpenAI 原生不支持 thinking block。实现时需要决定是放在 `content` 中用特殊标记包裹，还是通过自定义字段返回。可参考其他 OpenAI 兼容代理的做法
- **Q: `tool_choice` 的精确映射** → OpenAI 支持 `"auto"` / `"none"` / `"required"` / `{"type":"function","function":{"name":"..."}}` 等，需要在实现时确定哪些能映射到 Kiro

---

## Implementation Units

- [ ] U1. **OpenAI 类型定义**

**Goal:** 定义 OpenAI Chat Completions API 的请求和响应结构体

**Requirements:** R1, R2, R3, R4, R7

**Dependencies:** None

**Files:**
- Create: `src/openai/types.rs`
- Create: `src/openai/mod.rs`

**Approach:**
- 定义 `ChatCompletionRequest`：包含 `model`, `messages`, `temperature`, `top_p`, `n`, `stream`, `stop`, `max_tokens`, `tools`, `tool_choice` 等字段，大量使用 `Option` 以兼容不同客户端
- 定义 `ChatCompletionMessage`：支持 `system`/`developer`/`user`/`assistant`/`tool` 角色，`content` 支持 string 和 array 两种格式
- 定义 `ChatCompletionResponse`（非流式）：包含 `id`, `object`, `created`, `model`, `choices`, `usage`
- 定义 `ChatCompletionChunk`（流式）：`delta` 替代 `message`，增量内容
- 定义 `ToolCall`、`Function`、`ToolChoiceOption` 等工具相关类型
- 定义 OpenAI 格式的 `ErrorResponse`
- 定义 OpenAI 格式的 `Model` 和 `ModelsResponse`

**Patterns to follow:**
- `src/anthropic/types.rs` — 手写 serde 结构体风格，使用 `#[serde(skip_serializing_if)]` 控制可选字段

**Test scenarios:**
- Happy path: `ChatCompletionRequest` 能正确反序列化包含 messages + tools 的 JSON
- Happy path: `ChatCompletionResponse` 序列化后符合 OpenAI 格式规范
- Happy path: `ChatCompletionChunk` 序列化后包含 `delta` 而非 `message`
- Edge case: `content` 为 null 时（tool_calls 场景）能正确处理
- Edge case: `messages` 中 `tool` 角色消息包含 `tool_call_id`

**Verification:**
- 所有类型能通过 `serde_json::from_str` / `serde_json::to_string` 往返测试

---

- [ ] U2. **OpenAI → Kiro 协议转换器**

**Goal:** 将 OpenAI Chat Completions 请求转换为 Kiro `ConversationState`

**Requirements:** R1, R3, R5, R7, R8

**Dependencies:** U1

**Files:**
- Create: `src/openai/converter.rs`
- Modify: `src/anthropic/converter.rs` — 提取可复用的公共函数（`map_model`, `shorten_tool_name`, `map_tool_name`, `normalize_json_schema`）到共享位置

**Approach:**
- 核心函数 `convert_openai_request(req: &ChatCompletionRequest) -> Result<ConversionResult, ConversionError>`
- 消息角色映射：`system`/`developer` → Kiro 系统消息（user+assistant 配对）；`user` → Kiro user 消息；`assistant` → Kiro assistant 消息；`tool` → Kiro tool_result
- 工具定义转换：OpenAI `tools[].function` → Kiro `ToolSpecification`
- `tool_calls` 在 assistant 消息中 → Kiro `ToolUseEntry`
- 复用 `map_model()` 进行模型映射（OpenAI 客户端可能发送 `gpt-4o` 等模型名，需要额外映射或直接传递 Claude 模型名）
- 复用 `shorten_tool_name()` / `map_tool_name()` 处理超长工具名
- 复用 thinking 前缀注入逻辑
- 处理 `max_tokens` / `max_completion_tokens` 字段

**Patterns to follow:**
- `src/anthropic/converter.rs` — `convert_request()` 函数结构、`ConversionResult` 返回类型、工具配对验证逻辑

**Test scenarios:**
- Happy path: 基本的 user/assistant 消息对话转换为正确的 Kiro ConversationState
- Happy path: 包含 system 消息的请求正确转换为 Kiro 的 user+assistant 配对
- Happy path: 包含 tool_calls 和 tool 结果的多轮对话正确转换
- Happy path: 模型名 `claude-sonnet-4-6` 正确映射
- Edge case: `developer` 角色等同于 `system` 处理
- Edge case: 连续多条 assistant 消息合并
- Edge case: 空 messages 列表返回错误
- Error path: 不支持的模型名返回 `ConversionError`

**Verification:**
- 转换后的 `ConversationState` 结构与 Anthropic converter 产出的格式一致
- 工具名映射正确往返

---

- [ ] U3. **OpenAI SSE 流式响应转换**

**Goal:** 将 Kiro 事件流转换为 OpenAI Chat Completions 的 SSE 格式

**Requirements:** R2, R3

**Dependencies:** U1, U2

**Files:**
- Create: `src/openai/stream.rs`

**Approach:**
- 创建 `OpenAIStreamContext`，类似 `src/anthropic/stream.rs` 中的 `StreamContext`
- Kiro `AssistantResponse` 事件 → `ChatCompletionChunk` with `delta.content`
- Kiro `ToolUse` 事件 → `ChatCompletionChunk` with `delta.tool_calls`（增量 JSON）
- 流结束 → `finish_reason: "stop"` 或 `"tool_calls"`，最后发送 `data: [DONE]`
- 处理 `contextUsageEvent` 计算 usage（可选，通过 `stream_options.include_usage` 控制）
- SSE 格式：`data: {json}\n\n`，最后 `data: [DONE]\n\n`
- thinking 内容处理：如果启用 thinking，将 thinking 内容作为特殊的 content 块或自定义字段返回

**Patterns to follow:**
- `src/anthropic/stream.rs` — `StreamContext` 的状态机设计、事件处理流程
- `src/anthropic/handlers.rs` — `create_sse_stream()` 的 `stream::unfold` 模式

**Test scenarios:**
- Happy path: 纯文本响应生成正确的 SSE chunk 序列（role delta → content deltas → finish_reason stop → [DONE]）
- Happy path: tool_calls 响应生成正确的增量 tool_calls delta
- Happy path: 流结束时 `finish_reason` 正确（`stop` / `tool_calls`）
- Edge case: 空内容响应
- Edge case: 多个 tool_calls 在同一响应中

**Verification:**
- SSE 输出格式符合 `data: {json}\n\n` 规范
- 最后一个 chunk 包含 `finish_reason`，之后是 `data: [DONE]`

---

- [ ] U4. **OpenAI 请求处理器和路由**

**Goal:** 实现 OpenAI 端点的 Axum handler 和路由配置

**Requirements:** R1, R2, R4, R5, R6

**Dependencies:** U1, U2, U3

**Files:**
- Create: `src/openai/handlers.rs`
- Create: `src/openai/router.rs`
- Modify: `src/main.rs` — 挂载 OpenAI 路由

**Approach:**
- `get_models()` handler：返回 OpenAI 格式的模型列表（`object: "list"`, `data: [...]`，每个模型 `object: "model"`）
- `post_chat_completions()` handler：
  1. 解析 `ChatCompletionRequest`
  2. 检测 `-thinking` 模型名后缀，覆写 thinking 配置
  3. 调用 `convert_openai_request()` 转换为 Kiro 请求
  4. 根据 `stream` 字段分流到流式/非流式处理
  5. 流式：使用 `OpenAIStreamContext` + `create_sse_stream` 模式
  6. 非流式：收集所有事件，组装 `ChatCompletionResponse`
- 路由配置：`Router::new().nest("/openai/v1", openai_routes)`
- 复用现有的 `auth_middleware` 和 `AppState`

**Patterns to follow:**
- `src/anthropic/handlers.rs` — `post_messages()` 的完整处理流程
- `src/anthropic/router.rs` — 路由定义和中间件挂载

**Test scenarios:**
- Happy path: `POST /openai/v1/chat/completions` 非流式请求返回正确的 `ChatCompletionResponse`
- Happy path: `POST /openai/v1/chat/completions` 流式请求返回 SSE 格式
- Happy path: `GET /openai/v1/models` 返回 OpenAI 格式模型列表
- Error path: 无效 API key 返回 401
- Error path: 不支持的模型返回 400
- Error path: KiroProvider 未配置返回 503

**Verification:**
- 使用 `curl` 或 OpenAI SDK 能成功调用 `/openai/v1/chat/completions`
- 流式和非流式响应格式均符合 OpenAI API 规范

---

- [ ] U5. **公共转换逻辑提取和重构**

**Goal:** 将 Anthropic converter 和 OpenAI converter 共用的逻辑提取到共享模块

**Requirements:** R5

**Dependencies:** U2

**Files:**
- Create: `src/common/converter.rs` 或在 `src/common/` 下新建共享模块
- Modify: `src/anthropic/converter.rs` — 将公共函数改为调用共享模块
- Modify: `src/openai/converter.rs` — 调用共享模块

**Approach:**
- 提取到共享位置的函数：
  - `map_model()` — 模型名映射
  - `get_context_window_size()` — 上下文窗口大小
  - `shorten_tool_name()` / `map_tool_name()` — 工具名缩短
  - `normalize_json_schema()` — JSON Schema 规范化
  - `validate_tool_pairing()` / `remove_orphaned_tool_uses()` — 工具配对验证
  - `collect_history_tool_names()` / `create_placeholder_tool()` — 历史工具处理
- 保持 Anthropic converter 的现有测试通过
- 注意：此单元可以与 U2 合并实施，在实现 OpenAI converter 时同步提取

**Patterns to follow:**
- `src/common/auth.rs` — 现有的共享模块组织方式

**Test scenarios:**
- Happy path: 提取后 Anthropic converter 的所有现有测试仍然通过
- Happy path: OpenAI converter 能正确调用共享的 `map_model()`

**Verification:**
- `cargo test` 全部通过
- 无重复代码

---

- [ ] U6. **集成测试和文档更新**

**Goal:** 端到端验证 OpenAI 兼容端点，更新 README

**Requirements:** R1, R2, R3, R4

**Dependencies:** U4

**Files:**
- Modify: `src/test.rs` — 添加 OpenAI 端点的集成测试
- Modify: `README.md` — 添加 OpenAI 兼容端点的使用说明

**Approach:**
- 添加 OpenAI 格式的请求/响应集成测试
- 更新 README 说明新增的 `/openai/v1/chat/completions` 端点
- 更新 main.rs 中的启动日志，显示 OpenAI 端点信息

**Patterns to follow:**
- `src/test.rs` — 现有测试风格

**Test scenarios:**
- Integration: 完整的非流式对话请求 → 响应往返
- Integration: 完整的流式对话请求 → SSE 响应
- Integration: 包含 tool_calls 的多轮对话

**Verification:**
- `cargo test` 全部通过
- `cargo build --release` 编译成功

---

## System-Wide Impact

- **路由层**：新增 `/openai/v1/` 路由前缀，与现有 `/v1/` 和 `/cc/v1/` 互不干扰
- **认证**：复用现有 `auth_middleware`，无需修改
- **KiroProvider**：完全复用，无修改
- **Token 计数**：复用现有 `token::count_all_tokens()`
- **Admin API**：无影响
- **Unchanged invariants**：现有 Anthropic 端点的行为完全不变

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| OpenAI 格式的 tool_calls 增量 JSON 拼接复杂度高 | 参考 Anthropic stream.rs 中已有的 tool JSON 缓冲逻辑 |
| 不同 OpenAI 客户端发送的请求格式差异大 | 大量使用 `Option` 和 `serde_json::Value` 提高兼容性 |
| thinking 内容在 OpenAI 格式中无标准表达方式 | 先用自定义字段或 content 标记，后续可根据社区惯例调整 |
| 公共逻辑提取可能引入回归 | 提取后立即运行全量测试 |

---

## Sources & References

- OpenAI Chat Completions API 文档: https://platform.openai.com/docs/api-reference/chat
- `async-openai` crate: https://crates.io/crates/async-openai (调研参考，未采用)
- 现有代码: `src/anthropic/` 全部文件
