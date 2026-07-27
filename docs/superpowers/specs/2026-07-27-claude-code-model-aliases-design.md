# Claude Code 风格模型别名设计

日期：2026-07-27

## 背景

[Claude Code 模型配置](https://code.claude.com/docs/en/model-config) 提供家族别名（`opus` / `sonnet` / `haiku` 等），用户无需记忆完整版本号。在自定义 `ANTHROPIC_BASE_URL`（本代理）场景下，客户端会把模型字符串原样传给网关，因此别名需要在 kiro.rs 侧解析。

现状：`src/common/converter.rs` 的 `map_model` 用 `contains("opus"|"sonnet"|"haiku")` 做版本匹配；无精确版本时落入旧兜底：

| 输入特征 | 当前兜底 |
|----------|----------|
| 含 `sonnet` 且无精确版本 | `claude-sonnet-4.5` |
| 含 `opus` 且无精确版本 | `claude-opus-4.6` |
| 含 `haiku` | `claude-haiku-4.5` |

因此裸别名 `opus` / `sonnet` 已能“通”，但落到的是旧版，与 Claude Code 在 Anthropic API 上的推荐（Opus 5 / Sonnet 5）不一致。

此前 Opus 5 适配设计（`2026-07-25-claude-opus-5-design.md`）曾将「宽别名」列为非目标；本期是有意、受控地增加 **精确裸别名**，不扩大无版本兜底。

## 目标

客户端传入下列裸别名时，映射到当前 Anthropic API 推荐的最新可用 Kiro 模型 ID：

| 别名（大小写不敏感） | 映射到 |
|----------------------|--------|
| `opus` | `claude-opus-5` |
| `sonnet` | `claude-sonnet-5` |
| `haiku` | `claude-haiku-4.5` |

Anthropic 与 OpenAI 兼容路径均生效（二者共用 `map_model`）。上下文窗口大小通过现有 `get_context_window_size` 自动与映射结果一致。

## 非目标

- 不在 `/v1/models` 中挂出别名条目
- 不做 `sonnet[1m]` / `opus[1m]`（Sonnet 5 / Opus 5 原生已是 1M；后缀留待后续）
- 不做 `best` / `fable`（Kiro 当前无 Fable）
- 不做 `default` / `opusplan`（Claude Code 客户端语义，代理无法实现）
- 不做 `opus-thinking` 等别名变体；thinking 仍靠全名 `-thinking` 后缀或请求体字段
- 不改变既有「无精确版本」宽兜底（如 `claude-opus-4` → `claude-opus-4.6`）
- 不修复 `opusplan` 因含 `opus` 而误入宽兜底的已知限制
- 不引入配置文件 / 环境变量覆盖别名目标

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 范围 | 仅精确裸别名三元组 | 对齐用户选择的方案 A；改动最小 |
| 目标模型 | 固定为 Sonnet 5 / Opus 5 / Haiku 4.5 | 对齐 Claude Code 在 Anthropic API 上的当前推荐 |
| 匹配规则 | trim + lowercase 后整串相等 | 避免 `my-opus`、`opus-5`、`claude-opus` 误升 |
| 实现位置 | `map_model` 入口先查别名表 | 方案 1：Anthropic/OpenAI 自动生效，无需抽注册表 |
| 列表暴露 | 不改 `/v1/models` | 别名是解析便利，不是独立模型产品 |
| 宽兜底 | 保持现状 | 与「只做精确别名」一致，避免静默升到 Opus 5 |

## 改动点

### 1. `src/common/converter.rs` — `map_model`

在 GPT 映射之后、`contains("sonnet"|"opus"|"haiku")` 分支之前，增加精确别名表：

```text
trim + to_lowercase:
  "opus"   → Some("claude-opus-5")
  "sonnet" → Some("claude-sonnet-5")
  "haiku"  → Some("claude-haiku-4.5")
```

未命中则保持现有逻辑不变。

### 2. 上下文窗口

不改 `get_context_window_size`。别名经 `map_model` 落到 Opus 5 / Sonnet 5 后，已有分支返回 `1_000_000`；Haiku 走默认 `200_000`。

### 3. 测试

在 `map_model` / `get_context_window_size` 相关测试中补充：

- `map_model("opus"|"OPUS"|"sonnet"|"haiku")` → 上表目标 ID
- `map_model(" opus ")` → `claude-opus-5`（trim 后精确匹配）
- 全名行为不变：`claude-opus-5`、`claude-sonnet-5`、`claude-haiku-4-5-20251001` 等
- 宽兜底不变：`claude-opus-4` 仍为 `claude-opus-4.6`，`claude-sonnet-4` 仍为 `claude-sonnet-4.5`
- 非精确别名不命中表：`claude-opus`、`opus-5`、`my-opus` 不走别名分支（分别落入现有 contains / `None` 逻辑）
- `get_context_window_size("opus")` / `"sonnet"` 与对应全名一致（1M）

## 已知限制

- `opusplan`、`default`、`best`、`fable`、`*[1m]` 本期不支持；若客户端原样传入，行为与改前相同（可能失败或落入宽兜底）。
- 别名目标写死在代码中；Anthropic / Kiro 推荐模型升级时需手动改常量并发版。

## 验收标准

1. `map_model("opus") == Some("claude-opus-5")`（`sonnet` / `haiku` 同理）
2. `map_model("claude-opus-4")` 仍为 `Some("claude-opus-4.6")`（防误升）
3. `cargo test` 中 `map_model` 相关用例全部通过
4. 不改动 `/v1/models` 返回内容
