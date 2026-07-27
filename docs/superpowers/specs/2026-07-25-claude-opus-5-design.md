# Claude Opus 5 适配设计

日期：2026-07-25

## 背景

Kiro 已上线 Claude Opus 5（[changelog](https://kiro.dev/changelog/models/claude-opus-5/)）。Bedrock 模型卡确认：

- Model ID：`anthropic.claude-opus-5`（Kiro 侧对应 `claude-opus-5`）
- 上下文窗口：1M tokens
- Max output：128K tokens
- Reasoning：adaptive thinking 默认开启（可关；关闭时 effort 上限为 high）

本仓库此前已按相同模式适配过 Sonnet 5（#184）与 Opus 4.8（#171）。本次做最小增量，使客户端能请求 `claude-opus-5`。

## 目标

客户端传入 `claude-opus-5` 或 `claude-opus-5-thinking` 时：

1. 能映射到 Kiro 模型 ID `claude-opus-5`
2. 出现在 `GET /v1/models`
3. 按 1M 上下文窗口计算 usage / stop reason
4. `-thinking` 后缀走 adaptive thinking（与 Sonnet 5 一致）

## 非目标

- 不做宽别名：裸 `opus`、`claude-opus`、无版本号别名不映射到 Opus 5
- 不更新 README 映射表文案（后续再补）
- 不在客户端未传 `thinking` 时自动注入 adaptive（模型卡默认开启，留待后续）
- 不修复 Opus 4.7 / 4.8 的 `-thinking` 仍走 `enabled` 的已知缺口
- 不改动 `.worktrees/openai-compatible-api`

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 映射范围 | 仅名称含 `opus-5` | 精确、可跑通；避免误升旧别名 |
| Thinking | 对齐 Sonnet 5：仅 `-thinking` → adaptive + effort high | 行为一致、改动最小 |
| 实现路径 | 增量改 `converter` / `handlers`，不抽注册表 | 与 #184 / #171 一致 |

## 改动点

### 1. `src/anthropic/converter.rs` — `map_model`

在 opus 分支最前增加：

```text
contains("opus-5") → Some("claude-opus-5")
```

顺序必须优先于 `4-5` / `4.5` 等分支，避免将来误判；同时 `opus-4-5` 不含子串 `opus-5`，不会被误匹配。

### 2. `src/anthropic/converter.rs` — `get_context_window_size`

将 `claude-opus-5` 纳入 1M 上下文集合（与 Sonnet 5 / Opus 4.6+ 一致）。

### 3. `src/anthropic/handlers.rs` — `get_models`

新增两条模型记录，插在 `/v1/models` 列表最前（新于 Sonnet 5）：

| id | display_name | max_tokens | created |
|----|--------------|------------|---------|
| `claude-opus-5` | Claude Opus 5 | 128000 | 2026-07-24 |
| `claude-opus-5-thinking` | Claude Opus 5 (Thinking) | 128000 | 2026-07-24 |

### 4. `src/anthropic/handlers.rs` — `override_thinking_from_model_name`

将 `opus-5` 纳入 `is_adaptive_thinking` 判定（与现有 `sonnet-5` 并列），使 `claude-opus-5-thinking` 覆写为：

- `thinking.type = "adaptive"`
- `budget_tokens = 20000`
- `output_config.effort = "high"`

### 5. 测试

在 `converter.rs` 测试中补充：

- `map_model("claude-opus-5") == Some("claude-opus-5")`
- `map_model("claude-opus-5-thinking") == Some("claude-opus-5")`
- `get_context_window_size("claude-opus-5") == 1_000_000`
- `map_model("claude-opus-4-5-20251101")` 仍为 `claude-opus-4.5`（防误匹配）

## 验证

- `cargo test map_model`（及相关 converter 单测）通过
- 手工或现有流程确认 `/v1/models` 含新条目（实现阶段）

## 后续（本次不做）

1. README 映射表增加 `*opus-5*` → `claude-opus-5`
2. Opus 5 默认开启 adaptive（客户端未传 `thinking` 时注入）
3. Opus 4.7 / 4.8 `-thinking` 统一改为 adaptive
4. openai-compatible worktree 同步
