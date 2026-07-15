# OpenAI Compatible API Design

**Date:** 2026-07-10  
**Status:** Approved (brainstorm)  
**Branch context:** `feat/openai-compatible-api` already contains a partial Chat Completions implementation; this spec defines the target architecture and the gaps to close.

## Goal

Add OpenAI-compatible HTTP endpoints to kiro.rs that forward to Kiro via an **independent** OpenAI pipeline (not coupled to the Anthropic module), supporting:

- `POST /v1/chat/completions` — streaming and non-streaming
- `POST /v1/responses` — streaming and non-streaming (stateless)
- Core parity: text, tools/function calling, thinking, usage + credits

## Decisions

| Topic | Choice |
|-------|--------|
| Pipeline | Independent OpenAI → Kiro (no OpenAI ↔ Anthropic type conversion) |
| Feature scope | Text + tools + thinking + usage/credits; images deferred |
| Route prefix | Same `/v1` as Anthropic (no path collision with `/v1/messages`) |
| Thinking exposure | Chat: `reasoning_content`; Responses: `reasoning` output item; `-thinking` model suffix |
| Responses state | Stateless only — no `previous_response_id` / `store` |
| Module structure | Single `src/openai/` with shared converter + Kiro event accumulator; Chat/Responses are view adapters |

## Architecture

```
Client (OpenAI SDK)
  → POST /v1/chat/completions | POST /v1/responses
  → openai auth (OpenAI error shape; may reuse common::auth)
  → openai/handlers
       → openai/converter → KiroRequest
       → KiroProvider.call_api / call_api_stream
       → openai/stream (Chat chunks | Responses events)
```

**Isolation rules**

- `openai/**` must not `use crate::anthropic::...`
- Allowed: `crate::kiro::*`, `crate::common::auth`, `crate::common::converter` (shared Kiro helpers only: model map, tool-name shortening, pairing validation — not Anthropic types), `crate::token` if needed for estimates
- Thinking prefix injection, stream thinking-tag parse, Write/Edit description suffixes: implemented inside `openai/` (intentional duplication vs Anthropic is acceptable)

**Routing**

- Mount OpenAI routes on the same `/v1` router as Anthropic
- Do not re-register `GET /v1/models` if Anthropic already owns it; keep one models list readable by both client families
- Auth: same `config.apiKey` via `Authorization: Bearer` or `x-api-key`

## Endpoints

### `POST /v1/chat/completions`

**Request (consumed):** `model`, `messages`, `stream`, `tools` (`type: function`), `tool_choice` (best-effort: `auto` / `none` / named), `max_tokens` / `max_completion_tokens`, optional `user` (conversation id hint). Unsupported upstream knobs (`temperature`, `top_p`, `stop`, …) may be accepted and ignored.

**Messages:** `system` / `user` / `assistant` / `tool`; text content required in v1; images out of scope. Assistant history may include `tool_calls`; tool messages use `tool_call_id`.

**Thinking:** Enabled via `-thinking` model suffix (and any equivalent request flag if added later). Response exposes `reasoning_content` on message / delta.

**Non-stream response:** `chat.completion` with `choices[0].message` (`content`, `reasoning_content`, `tool_calls`) and `finish_reason` ∈ `stop` | `length` | `tool_calls`.

**Stream response:** SSE `chat.completion.chunk` deltas; end with `data: [DONE]`. Optional `stream_options.include_usage`.

### `POST /v1/responses`

**Request (consumed):** `model`, `input` (string or input items), `instructions` → system, `tools`, `stream`.

**Not implemented:** `previous_response_id`, `store`, retrieve/delete, non-function advanced tools.

**Non-stream response:** `response` object; `output[]` may include `reasoning`, `message` (`output_text`), `function_call`.

**Stream response:** Minimal stable Responses SSE subset, e.g. `response.created` → output item lifecycle → `response.reasoning_*` / `response.output_text.delta` / `response.function_call_arguments.delta` → `response.completed`. Align names to official Responses streaming docs; do not require full event parity.

## Mapping to Kiro

| OpenAI | Kiro |
|--------|------|
| messages / input history | `ConversationState.history` + `currentMessage` |
| function tools | `ToolSpecification` (+ name shorten/restore in openai path) |
| thinking on | openai-owned thinking system prefix |
| usage | `prompt_tokens` / `completion_tokens` / `total_tokens`; extend with `credits` + `metering_unit` when `meteringEvent` present |
| context full / max tokens | Chat `finish_reason`; Responses `status` / error |

Model ID list and `-thinking` suffix follow the same catalog as Anthropic-facing models; `map_model` lives in shared common helpers or openai-local copy — not via `anthropic::converter`.

## Data flow

### Shared inside `openai/`

1. Normalize request → internal OpenAI-shaped conversation (messages + tools + thinking flag)
2. `converter` → `KiroRequest`
3. Provider call (retry/failover stays in `KiroProvider`)
4. Decode Kiro events once
5. View adapters:
   - Chat Completions JSON / SSE chunks
   - Responses JSON / SSE events

### Stream state machine (`openai/stream`)

- Parse `<thinking>` tags from assistant text into reasoning vs content (openai-owned)
- Map `ToolUse` → Chat `tool_calls` deltas or Responses `function_call` + argument deltas
- `ContextUsage` → prompt token correction; 100% → terminal reason
- `Metering` → final usage credits
- Keepalive ping (~25s) as needed for long streams
- Empty completion (no text, no tool calls) → 502-style API error, not silent empty success

### Errors (OpenAI shape)

| Case | HTTP | `error.type` |
|------|------|----------------|
| Bad/missing API key | 401 | `invalid_request_error` (auth) |
| Bad body / unsupported model / context full / input too long | 400 | `invalid_request_error` |
| Upstream failure / empty completion | 502 | `api_error` / `server_error` |
| Rate limit if detectable | 429 | `rate_limit_error` |

## Module layout

| File | Responsibility |
|------|----------------|
| `src/openai/mod.rs` | Module exports |
| `src/openai/types.rs` | Chat + Responses request/response/error types |
| `src/openai/converter.rs` | OpenAI → `KiroRequest` |
| `src/openai/stream.rs` | Kiro events → Chat / Responses views |
| `src/openai/handlers.rs` | Endpoint orchestration |
| `src/openai/router.rs` | `/v1/chat/completions`, `/v1/responses` |
| `src/openai/middleware.rs` | Own `AppState` + OpenAI-format auth (no anthropic import) |
| `src/openai/usage.rs` | Optional: build usage JSON including credits |

`main.rs`: merge openai router with anthropic; log new endpoints.

## Gap vs current branch WIP

Existing `feat/openai-compatible-api` work is a starting point, not the final design. Known gaps:

1. **Prefix:** currently `/openai/v1/...` → move to `/v1/chat/completions` and `/v1/responses`
2. **Coupling:** `handlers`/`router` import `crate::anthropic::middleware` → replace with openai-owned middleware / shared app state that does not live under `anthropic`
3. **Responses API:** missing entirely
4. **Thinking / `reasoning_content`:** not implemented in stream or non-stream
5. **Credits / metering:** not wired into OpenAI usage
6. **Models list:** may be stale vs Anthropic `get_models`; keep single `/v1/models` source of truth
7. **Isolation:** verify no remaining `use crate::anthropic`

## Non-goals (v1)

- Images / vision
- Stateful Responses (`previous_response_id`, store, CRUD)
- Sharing Anthropic `StreamContext` / `convert_request`
- Full OpenAI parameter parity
- Embeddings, audio, Assistants, and other OpenAI APIs
- Claude Code–style `/cc` buffered variant for OpenAI

## Success criteria

1. Official OpenAI Python/Node SDK can run Chat Completions stream + non-stream with `tool_calls`
2. Same SDK can run Responses stream + non-stream for text / reasoning / `function_call` with client-supplied full `input` each turn
3. `-thinking` models expose reasoning fields on both APIs
4. Usage includes credits when upstream sends metering
5. `cargo test` passes; `openai` does not depend on `anthropic`

## Testing

- Converter: message/tool history ↔ Kiro conversation; `-thinking`; schema normalization
- Stream: thinking across chunks; tool call streaming; usage/credits
- Handlers: Chat + Responses contract snapshots (stream and non-stream)
- Boundary: CI or test asserting no `crate::anthropic` imports under `src/openai/`
