# Claude Sonnet 5 支持说明

Kiro 于 2026-06-30 上线 Claude Sonnet 5（Experimental）。本代理已支持基础模型映射与 `/v1/models` 列表，以下为上游行为与**暂未实现**的改进项记录。

## 上游行为（Anthropic / Kiro）

参考：[What's new in Claude Sonnet 5](https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5)、[Kiro Models](https://kiro.dev/docs/models/)

| 属性 | 值 |
|------|-----|
| API / Kiro 模型 ID | `claude-sonnet-5` |
| 上下文窗口 | 1M |
| 区域 | 仅 `us-east-1`（实验性） |
| 积分倍率 | 1.3x |

### Thinking 行为

| 请求配置 | Sonnet 5 行为 |
|----------|---------------|
| 不传 `thinking` | **默认开启 adaptive thinking** |
| `thinking: { "type": "adaptive" }` | 推荐方式，可配合 `output_config.effort` |
| `thinking: { "type": "enabled", "budget_tokens": N }` | **400 错误**（manual extended thinking 已移除） |
| `thinking: { "type": "disabled" }` | 关闭 thinking |

与 Sonnet 4.6 对比：4.6 无 `thinking` 字段时不思考；Sonnet 5 默认即 adaptive。

## 当前实现

| 场景 | 行为 | 状态 |
|------|------|------|
| `claude-sonnet-5` | 映射为 `claude-sonnet-5`，不向上游注入 thinking 标签 | ✅ 符合上游默认 |
| `claude-sonnet-5-thinking` | 覆写为 `adaptive` + `effort: high` | ✅ 正确 |
| 1M 上下文窗口 | `get_context_window_size` 返回 1_000_000 | ✅ |

实现位置：`src/anthropic/converter.rs`（`map_model`）、`src/anthropic/handlers.rs`（`override_thinking_from_model_name`）。

## 已知缺口（暂不实现）

以下问题已识别，**当前版本刻意不做**，留待后续按需补齐。

### 1. 客户端显式传 `thinking.enabled`（优先级：高）

**现象：** Claude Code 等客户端可能对 Sonnet 5 发送：

```json
{
  "model": "claude-sonnet-5",
  "thinking": { "type": "enabled", "budget_tokens": 20000 }
}
```

**现状：** 代理原样转换为 Kiro 的 `<thinking_mode>enabled</thinking_mode>` 标签。

**风险：** Sonnet 5 不支持 manual extended thinking，上游可能返回 400。

**建议改动：** 检测到 Sonnet 5 时，将 `thinking.type == "enabled"` 自动改写为 `adaptive`（保留或映射 `output_config.effort`）。

### 2. 流式响应未拆分 thinking 块（优先级：中）

**现象：** 普通 `claude-sonnet-5` 请求无 `thinking` 字段时，`thinking_enabled` 为 `false`。

**现状：** 上游仍会输出 `<thinking>...</thinking>`，但流式解析器不会拆成独立的 `thinking` content block，思考内容可能混入 `text` 块。

**建议改动：** 对 Sonnet 5 在**响应解析侧**默认视 `thinking_enabled = true`（不必向上游额外注入 thinking 标签）。

相关代码：`src/anthropic/handlers.rs`（`thinking_enabled` 判断）、`src/anthropic/stream.rs`（`StreamContext::thinking_enabled`）。

### 3. Opus 4.7 / 4.8 的 `-thinking` 后缀（优先级：低）

**现象：** `claude-opus-4-7-thinking`、`claude-opus-4-8-thinking` 仍映射为 `enabled` 类型。

**背景：** Anthropic 在 Opus 4.7+ 同样废弃 manual extended thinking，推荐 adaptive。

**建议改动：** 与 Opus 4.6 / Sonnet 5 对齐，`-thinking` 后缀统一走 `adaptive`。

### 4. effort 档位（优先级：低）

Kiro 文档尚未列出 Sonnet 5 的 effort 可选档位（模型为 Experimental）。当前 `-thinking` 变体默认 `effort: high`，与 Opus 4.6 一致，暂维持不变。

## 使用注意

- 配置 `region` / `apiRegion` 为 `us-east-1`（Sonnet 5 暂不支持 `eu-central-1`）。
- Sonnet 5 使用新 tokenizer，相同文本 token 数约比 Sonnet 4.6 多 30%，需重新评估 `max_tokens` 预算。
- 不支持 `temperature` / `top_p` / `top_k` 非默认值（上游 400）。

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-07-01 | 初版：基础映射与 `-thinking` → adaptive；记录已知缺口，暂不实现后续项 |
