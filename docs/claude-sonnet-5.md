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

### 2. 流式响应未拆分 thinking 块（优先级：中）— ✅ 已修复（2026-07-07）

**现象：** 普通 `claude-sonnet-5` 请求无 `thinking` 字段时，`thinking_enabled` 为 `false`。

**现状：** 上游仍会输出 `<thinking>...</thinking>`，但流式解析器不会拆成独立的 `thinking` content block，思考内容可能混入 `text` 块。

**已落地改动：** 新增 `should_extract_thinking`，`post_messages` 与 `post_messages_cc` 均改用。对 Sonnet 5 在**响应解析侧**默认视 `thinking_enabled = true`（除非客户端显式 `type = "disabled"`），不再向上游额外注入 thinking 标签。

相关代码：`src/anthropic/handlers.rs`（`should_extract_thinking`）、`src/anthropic/stream.rs`（`StreamContext::thinking_enabled`）。

### 3. Opus 4.7 / 4.8 的 `-thinking` 后缀（优先级：低）

**现象：** `claude-opus-4-7-thinking`、`claude-opus-4-8-thinking` 仍映射为 `enabled` 类型。

**背景：** Anthropic 在 Opus 4.7+ 同样废弃 manual extended thinking，推荐 adaptive。

**建议改动：** 与 Opus 4.6 / Sonnet 5 对齐，`-thinking` 后缀统一走 `adaptive`。

### 4. effort 档位（优先级：低）

Kiro 文档尚未列出 Sonnet 5 的 effort 可选档位（模型为 Experimental）。当前 `-thinking` 变体默认 `effort: high`，与 Opus 4.6 一致，暂维持不变。

## 使用注意

- 配置 `region` / `apiRegion` 为 `us-east-1`（Sonnet 5 暂不支持 `eu-central-1`）。
- Sonnet 5 使用新 tokenizer，相同文本 token 数约比 Sonnet 4.6 多 **~74%**（实测 ktok=130 同等填充文本：sonnet-5 = 152080 tokens，sonnet-4.6 = 87315 tokens），需重新评估 `max_tokens` 预算与上下文占用。
- 不支持 `temperature` / `top_p` / `top_k` 非默认值（上游 400）。

## 大 input 空 completion 调查记录（2026-07-07）

### 现象

`claude-sonnet-5` 在大 input 下网关返回空 completion：`200` + 正常 SSE + `stop_reason=end_turn` + `output_tokens=0` + **无任何 `content_block_delta`**。`claude-sonnet-4.6` / `claude-haiku-4.5` 同等 input 正常出正文。客户端表现为静默拿到空结果。

### 复现

复现脚本（纯标准库，指向本地或部署的 kiro.rs）：

```python
import json, urllib.request, time
BASE = "http://127.0.0.1:8990"   # 或部署地址
TOKEN = "<kiro.rs apiKey>"
URL = BASE.rstrip("/") + "/v1/messages"
unit = "The momentum factor captures persistence in asset returns across horizons. "
def call(model, ktok):
    filler = unit * ((ktok * 4000) // len(unit)) if ktok > 0 else ""
    hdr = {"content-type": "application/json",
           "authorization": "Bearer " + TOKEN,
           "anthropic-version": "2023-06-01"}
    body = {"model": model, "max_tokens": 32, "stream": True,
            "messages": [{"role": "user",
                          "content": filler + "\n\nIn 5 words, summarize the topic above."}]}
    r = urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers=hdr), timeout=200)
    text = sum(1 for raw in r if b"text_delta" in raw)
    print(f"{model:24} ktok={ktok:4} {'EMPTY' if text==0 else 'OK'} txt_d={text}")
for k in [50, 60, 90, 130]:
    call("claude-sonnet-5", k)
call("claude-sonnet-4.6", 130)   # 对照：OK
```

### 调查过程与结论

带新诊断日志的 kiro.rs 本地跑 22 个请求，三类诊断 warn 命中：

| 诊断 warn | 命中 | 含义 |
|------|------|------|
| `收到未识别的事件类型`（`EventType::Unknown`） | 0 | 上游没用新事件类型下发 thinking |
| `assistantResponseEvent 的 content 为空但存在其他字段` | 0 | 上游没把 thinking 塞进 `content` 之外字段 |
| `流式空 completion`（`output_tokens=0 && stop_reason=end_turn`） | 16 | 与所有 EMPTY 结果一一对应 |

**结论：上游流里就是没有 content 事件（只有 `contextUsageEvent` + 流结束），kiro.rs 没有丢弃任何 thinking。** 排除了"网关丢弃 thinking"假设。

### 排除的变量

- `max_tokens`：ktok=130 下 32 / 128 / 512 / 2048 / 8192 全部 EMPTY → **不是 thinking 吃光预算**。
- 显式 `thinking`：`{"type":"enabled"}` / `{"type":"disabled"}` 都 EMPTY → **与 thinking 配置无关**。
- `[1m]` 后缀 + `context-1m-2025-08-07` beta：107K / 152K 仍 EMPTY → **1M 上下文变体未抬升有效上限**。

### 真正根因：上游对 sonnet-5 有 ~70K 硬输入上限

阈值二分（`max_tokens=64`）：

| model | input_tokens | 结果 |
|------|------|------|
| sonnet-5 | 62479 / 64731 / 66978 / **69199** | OK |
| sonnet-5 | **71439** / 73692 / 84885 / 96075 / 107286 / 152080 | EMPTY |
| sonnet-5[1m] + 1m beta | 107286 / 152080 | EMPTY |
| sonnet-4.6 | 87315 / 106518 / **125718** | OK |

- sonnet-5 断崖在 **~70K tokens**（69199 OK → 71439 EMPTY），与 `max_tokens`、`thinking`、`[1m]`、1m beta 全部无关。
- sonnet-4.6 在 126K 仍 OK，上限远高于 sonnet-5。
- 同样填充文本，sonnet-5 算 152080 tokens、sonnet-4.6 算 87315 tokens（新 tokenizer ~1.74×），sonnet-5 更易撞顶。

**判定：上游 Kiro/CodeWhisperer 端点对 sonnet-5 的有效上下文上限仅 ~70K tokens，远低于标称 200K / 1M。超出即返回合法空流。这是上游侧缺陷，不是 kiro.rs 的 bug。**

### kiro.rs 侧已落地的防护（PR #2）

1. **诊断日志**（定位用）：
   - `src/kiro/model/events/base.rs`：`EventType::Unknown` 不再静默丢弃，记录原始事件类型 + payload 片段。
   - `src/kiro/model/events/assistant.rs`：`content` 为空但 `extra` 含字段时告警。
2. **防御性空 completion 处理**：
   - `src/anthropic/handlers.rs` `handle_non_stream_request`：空 content + `output_tokens=0` + `stop_reason=end_turn` → 返回 `502`（排除 `model_context_window_exceeded` / `max_tokens` 等合法空输出）。
   - `src/anthropic/stream.rs` `generate_final_events`：同条件打 warn 日志（流式已发 `message_start`，无法改状态码）。
3. **sonnet-5 响应侧默认 `thinking_enabled`**：见已知缺口 #2。

### 未做的 / 待定

- **真正修 sonnet-5 大 input** 需在 kiro-nexus / 上游路由层：把 sonnet-5 路由到支持满上下文的上游端点；或在网关侧对 `sonnet-5 + input > ~70K` 主动返回 `context_too_large` 错误，而不是转发上游的空流。kiro.rs 当前只做防御性转换，不改路由。
- **回归用例**：建议把上面的复现脚本纳入回归矩阵，覆盖 `sonnet-5 / sonnet-4.6 / haiku × [小/中/大 input]`，断言大 input 也 `output_tokens > 0`（或至少不再静默空）。

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-07-01 | 初版：基础映射与 `-thinking` → adaptive；记录已知缺口，暂不实现后续项 |
| 2026-07-07 | 修复已知缺口 #2（sonnet-5 响应侧默认 thinking_enabled）；新增大 input 空 completion 调查记录与防御性 502/warn 处理（PR #2） |
