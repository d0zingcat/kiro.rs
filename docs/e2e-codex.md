# Codex + kiro.rs 本地 E2E 记录

本文记录用 **Codex CLI（headless）** 对接本地 **kiro.rs OpenAI 兼容端点**（`POST /v1/responses`）做端到端验证的流程、配置与实测结果。目标：确认 Codex 能经代理完成工具调用、多轮对话、大上下文与真实写代码场景，且**不走本机 ChatGPT 订阅**。

> 日期：2026-07-10 ~ 2026-07-20（含 `feat/gpt-5-6-models`）  
> Codex CLI：`0.144.1`  
> 工作区产物默认落在：`/tmp/kiro-e2e/`、`/tmp/kiro-gpt-e2e/`

---

## 1. 结论摘要

| 场景 | 模型 | 结果 |
|------|------|------|
| 多轮 + 工具 + 大上下文（3 turn） | `claude-haiku-4-5-20251001` | ✅ 通过；Turn3 ~236k input tokens |
| 从零生成可玩贪食蛇 | `claude-haiku-4-5-20251001` | ✅ `snake.html` + README；exit 0 |
| Opus 迭代 Neon Snake | `claude-opus-4-8` | ✅ 单文件增强版；exit 0 |
| Codex × GPT-5.6 Sol 多轮写游戏 | `gpt-5.6-sol` | ✅ `snake.html` + localStorage 最高分 |
| Codex × GPT-5.6 Terra 多轮写游戏 | `gpt-5.6-terra` | ✅ `tictactoe.html` + 比分 |
| 流量是否打到 ChatGPT | — | ❌ 否（隔离 `CODEX_HOME` + `base_url=127.0.0.1:8990`） |

Codex 侧曾出现 `GET /v1/models` 解析告警（期望字段 `models`，本代理返回 OpenAI 式 `data`），**不影响** `responses` 主路径。详见文末「已知问题」。

GPT-5.6（Sol/Terra/Luna）走 Codex **Responses Lite** 工具形态（`additional_tools` + freeform `exec`）。本代理**不是** ChatGPT 原生后端，采用「提升 tools + custom↔function 往返」适配；说明见 [§8.1](#81-codex-responses-lite--gpt-56-适配策略)。

---

## 2. 架构

```
Codex CLI (CODEX_HOME=/tmp/kiro-e2e/codex-home)
    │  wire_api = responses
    │  Authorization: Bearer <本地 api-key>
    ▼
kiro.rs  :8990
    POST /v1/responses
    GET  /v1/models   （Codex 启动时会刷一次，格式不完全兼容）
    ▼
上游 Kiro（credentials.json，endpoint=ide）
```

关键点：**必须用独立 `CODEX_HOME`**，否则会读 `~/.codex`，可能误用已登录的 ChatGPT / 默认 OpenAI provider。

---

## 3. 前置条件

1. 本机已安装 `codex`（`which codex`）
2. 已构建带 OpenAI 端点的二进制，例如：

```bash
cd /path/to/kiro.rs   # 或 .worktrees/openai-compatible-api
cargo build --release --no-default-features
# 若启用默认 admin-ui feature，需先有 admin-ui/dist
```

3. 有效的上游 Kiro 凭据（本文用 `authMethod: api_key` + `endpoint: ide`）
4. 端口 `8990` 空闲

---

## 4. 目录与配置（可复现）

建议统一工作根目录：

```bash
mkdir -p /tmp/kiro-e2e/{codex-home,snake-game,snake-results,codex-multi,codex-multi-results}
```

### 4.1 代理配置 `/tmp/kiro-e2e/config.json`

```json
{
  "host": "127.0.0.1",
  "port": 8990,
  "apiKey": "sk-kiro-rs-qazWSXedcRFV123456",
  "region": "us-east-1"
}
```

### 4.2 上游凭据 `/tmp/kiro-e2e/credentials.json`

```json
{
  "authMethod": "api_key",
  "disabled": false,
  "kiroApiKey": "<你的 ksk_...>",
  "endpoint": "ide"
}
```

注意：`endpoint` 必须是已注册端点（当前常见为 `ide`）。写成 `cli` 会在启动时报「未知端点」并退出。

### 4.3 给 Codex 读的 Bearer Key

```bash
printf '%s' 'sk-kiro-rs-qazWSXedcRFV123456' > /tmp/kiro-e2e/api-key
```

须与 `config.json` 的 `apiKey` 一致。

### 4.4 环境变量（可选）`.e2e.env`

```bash
KIRO_API_KEY=sk-kiro-rs-qazWSXedcRFV123456
KIRO_BASE=http://127.0.0.1:8990
KIRO_HOST=127.0.0.1
KIRO_PORT=8990
```

### 4.5 Codex 隔离配置 `/tmp/kiro-e2e/codex-home/config.toml`

```toml
model = "claude-haiku-4-5-20251001"   # 或 claude-opus-4-8
model_provider = "kiro"
model_reasoning_effort = "low"        # Opus 迭代时可改 medium
sandbox_mode = "workspace-write"
web_search = "disabled"

[model_providers.kiro]
name = "kiro-rs local"
base_url = "http://127.0.0.1:8990/v1"
wire_api = "responses"

[model_providers.kiro.auth]
command = "/bin/cat"
args = ["/tmp/kiro-e2e/api-key"]
timeout_ms = 5000
refresh_interval_ms = 0

# 按实际工作区路径信任项目（macOS 上 /tmp 常解析为 /private/tmp）
[projects."/tmp/kiro-e2e/snake-game"]
trust_level = "trusted"

[projects."/private/tmp/kiro-e2e/snake-game"]
trust_level = "trusted"
```

`codex exec` 时也可加 `-m claude-opus-4-8` 临时覆盖模型。

---

## 5. 启动与冒烟

```bash
export CODEX_HOME=/tmp/kiro-e2e/codex-home

./target/release/kiro-rs \
  -c /tmp/kiro-e2e/config.json \
  --credentials /tmp/kiro-e2e/credentials.json
```

启动日志应包含 OpenAI 路由，例如：

- `POST /v1/chat/completions`
- `POST /v1/responses`

冒烟（确认打到本地、模型可用）：

```bash
set -a; source .e2e.env; set +a   # 或手动 export KIRO_*

curl -sS -H "Authorization: Bearer $KIRO_API_KEY" "$KIRO_BASE/v1/models" | head -c 200

curl -sS -H "Authorization: Bearer $KIRO_API_KEY" \
  -H "Content-Type: application/json" \
  "$KIRO_BASE/v1/chat/completions" \
  -d '{"model":"claude-opus-4-8","messages":[{"role":"user","content":"reply exactly: OPUS48_OK"}],"max_tokens":20}'
```

---

## 6. Codex headless 用法

### 6.1 单次任务（新建 thread）

```bash
export CODEX_HOME=/tmp/kiro-e2e/codex-home
cd /tmp/kiro-e2e/snake-game   # 或其他信任过的工作区

codex exec --skip-git-repo-check -s workspace-write --json \
  -m claude-haiku-4-5-20251001 \
  -o /tmp/kiro-e2e/snake-results/last.txt \
  "你的任务提示..." \
  > /tmp/kiro-e2e/snake-results/codex.jsonl \
  2> /tmp/kiro-e2e/snake-results/codex.err
echo $? > /tmp/kiro-e2e/snake-results/exit.txt
```

### 6.2 多轮 resume

```bash
# thread_id 来自 jsonl 里 thread.started 事件
THREAD=<uuid>

codex exec resume --skip-git-repo-check --json \
  -o /tmp/kiro-e2e/codex-multi-results/turn2-last.txt \
  "$THREAD" \
  "下一轮提示..." \
  > /tmp/kiro-e2e/codex-multi-results/turn2.jsonl \
  2> /tmp/kiro-e2e/codex-multi-results/turn2.err
```

注意：`resume` 子命令与 `-s` 等 flag 的顺序以当前 `codex exec resume -h` 为准；实测部分版本对 flag 顺序敏感。

### 6.3 如何确认没用 ChatGPT 订阅

同时满足即可：

1. 进程环境 `CODEX_HOME=/tmp/kiro-e2e/codex-home`（不是默认 `~/.codex`）
2. `config.toml` 里 `model_provider = "kiro"` 且 `base_url = "http://127.0.0.1:8990/v1"`
3. kiro 日志出现 `Received POST /v1/responses request model=<你指定的模型>`，且模型为 Claude id（如 `claude-opus-4-8`），而非 `gpt-*`

---

## 7. 已跑场景与结果

### 7.1 多轮工具 + 大上下文（Codex）

- 工作区：`/tmp/kiro-e2e/codex-multi`（预置 docs / data / src，总体积加压）
- 模型：`claude-haiku-4-5-20251001`
- 协议：Responses + tools（含 Codex 扁平 function tools；代理需容忍并跳过 `local_shell` 等非 function 项）

| Turn | 工具执行约 | input tokens | exit |
|------|------------|--------------|------|
| 1 | 5 | ~35.6k | 0 |
| 2 | 5 | ~106k | 0 |
| 3 | 7 | ~236k | 0 |

验证点：跨轮记忆正确、大文件读取 / 日志统计无 422、无代理崩溃。

产物目录：`/tmp/kiro-e2e/codex-multi-results/`。

### 7.2 生成贪食蛇（Haiku）

- 工作区：`/tmp/kiro-e2e/snake-game`
- 模型：`claude-haiku-4-5-20251001`
- 提示：单文件 `snake.html` + README，方向键/WASD、暂停、重开、得分等
- 结果：exit 0；约 **146k** input / **5.4k** output；`node --check` 抽脚本通过
- 产物：`snake.html`、`README.md`

### 7.3 Neon Snake 迭代（Opus 4.8）

- 同一工作区，覆盖模型为 `claude-opus-4-8`
- 提示：在现有文件上自由增强（仍保持单 HTML、无外链）
- 结果：exit 0；约 **418k** input / **11k** output；kiro 日志全部 `model=claude-opus-4-8`
- 增强示例：combo、金色食物、障碍、Walls/Wrap、粒子/震动、Web Audio、开始/结算 UI

运行：

```bash
open /tmp/kiro-e2e/snake-game/snake.html
```

---

## 8. 代理侧相关实现要点（E2E 踩坑）

| 问题 | 现象 | 处理 |
|------|------|------|
| Codex tools 扁平格式 | `422`：`tools[0]: missing field function` | `src/openai/types.rs` 宽松反序列化；跳过 `local_shell` 等 |
| Codex Responses Lite `additional_tools` | GPT-5.6 报「无 shell / 无法写文件」 | 吸收并提升到顶层 tools；见 [§8.1](#81-codex-responses-lite--gpt-56-适配策略) |
| freeform `exec`（`type: custom`） | 模型调了工具但 Codex 不执行 | 上游按 JSON function 收；回传 `custom_tool_call` |
| GPT hidden CoT reasoning SSE | `ReasoningRawContentDelta without active item` | GPT-5.6 **不转发**原生 reasoning 事件 |
| 凭据 `endpoint: cli` | 启动失败：未知端点 | 改为 `ide`（或注册对应端点） |
| 误用用户 `~/.codex` | 可能打到 ChatGPT | 强制 `CODEX_HOME` + 自定义 `model_providers.kiro` |
| `/v1/models` 形状 | Codex stderr：`missing field models` | 启动刷模型列表失败，主会话仍可用；见下节 |

### 8.1 Codex Responses Lite / GPT-5.6 适配策略

Codex CLI ≥ 0.144 对 `gpt-5.6-*`（及部分新模型）使用 **Responses Lite** 线格式：

- 顶层常为 `"tools": null`
- 工具定义放在 `input[]` 里一条 `{"type":"additional_tools","role":"developer","tools":[...]}`
- 主工具是 freeform **`exec`**（`type: custom`，grammar），模型写 JS 调 `tools.exec_command` / `tools.apply_patch` 等
- 客户端期望模型输出 **`custom_tool_call`**（`input` 为原文），而不是 `function_call` JSON arguments

kiro.rs 上游是 Kiro 的 JSON `tool_use`，**不能**像对接 ChatGPT 那样整段透传 Responses Lite。因此采用与社区代理（如 LiteLLM hoist、metapi absorb）同类的适配：

```text
Codex 请求                          kiro.rs                              Codex 客户端
─────────                          ───────                              ──────────
input: additional_tools            吸收 tools → 合并顶层 tools
  ├─ custom exec  ───────────────► 转为 function{input:string} ──► Kiro tool_use
  ├─ function wait / ...
  └─ namespace …（展开嵌套 function）
                                   Kiro tool_use(exec, {"input":"…"})
                                     ───────────────────────────────► custom_tool_call
                                                                      (unwrap 成 freeform input)
```

实现位置：

| 步骤 | 代码 |
|------|------|
| 吸收 `additional_tools`；识别 custom | `src/openai/converter.rs`（`absorb_additional_tools_item` / `normalize_responses_to_chat`） |
| custom / namespace 解析 | `src/openai/types.rs`（`ToolDefinition::from_custom_tool` / `collect_from_codex_entry`） |
| 按名回传 `custom_tool_call` | `src/openai/responses_stream.rs`（`custom_tool_names`） |
| GPT 不转发 reasoning SSE | `src/openai/thinking.rs`（`is_gpt_hidden_cot_model`）+ `responses_stream.rs` |

**不是** Responses Lite 字节级透传；若将来对接真正的 OpenAI Codex 后端，应优先透传。对 Kiro 这类非 Lite 上游，当前 hoist + 往返是必要折中。

多轮历史中的 `custom_tool_call` / `custom_tool_call_output` 会归一化为 Chat 侧 `tool_calls` / `tool` 消息再转 Kiro。

---

## 9. 已知问题（观察项）

### 9.1 Codex 刷新 models 列表失败

Codex 期望类似 `{ "models": [...] }`，kiro.rs 的 `GET /v1/models` 返回 Anthropic/OpenAI 常见的 `{ "object":"list", "data":[...] }`。会在 stderr 打 ERROR，但指定了 `model` + `model_provider` 后 **不影响** `POST /v1/responses`。

若要消除噪音，可后续为 Codex 增加兼容字段或独立 models 视图（未做）。

### 9.2 `reasoningContentEvent`

上游（例如 Opus 4.8）会推送：

```text
event_type=reasoningContentEvent
payload ≈ {"text":"..."} 或带 signature 的收尾块
```

当前行为：

- ✅ Claude 等：`POST /v1/responses` 流式与非流式转发为 Responses `reasoning` item（`response.reasoning_text.delta` 等）
- ✅ 不依赖模型名 `-thinking` 后缀（原生事件对非 GPT 模型始终转发）
- ✅ **GPT-5.6**：hidden CoT，**刻意不转发** reasoning SSE（避免 Codex `ReasoningRawContentDelta without active item`）
- ⚪ Chat Completions：暂不专门转发（本场景以 Responses 为准）
- ⚪ Anthropic `/v1/messages`：暂未映射为 `thinking` block（可后续补）

### 9.3 `collaboration` namespace

Codex `additional_tools` 常带 `type: namespace, name: collaboration`（子代理）。对接 Azure / 部分官方 Responses Lite 时该 namespace 会被拒绝。本代理会将其**展开为普通 function** 发给 Kiro；本地 Sol/Terra 单 agent 写文件场景已验证可用。若上游拒绝 namespace，可再考虑直接丢弃该条目。

---

## 10. 建议复测清单

- [ ] `cargo build --release`（或当前 feature 组合）后启动 `-c` + `--credentials`
- [ ] `curl`：`/v1/models`、`/v1/chat/completions`、`/v1/responses`（流式一次）
- [ ] `CODEX_HOME=... codex exec ...` 小任务（读文件 + 写文件）
- [ ] 确认 kiro 日志出现 `additional_tools` 吸收 / `custom_tools={"exec"}`（GPT-5.6）
- [ ] （可选）`codex exec resume` 第二轮
- [ ] （可选）`-m gpt-5.6-sol` / `gpt-5.6-terra` 写单文件小游戏
- [ ] （可选）换 `claude-opus-4-8` 再跑一轮写代码任务

---

## 11. 相关文件

| 路径 | 说明 |
|------|------|
| `src/openai/` | OpenAI 兼容层（与 Anthropic 解耦） |
| `src/openai/types.rs` | 扁平 tools / custom / namespace 兼容 |
| `src/openai/converter.rs` | Responses 归一化；吸收 `additional_tools` |
| `src/openai/responses_stream.rs` | Responses SSE；`custom_tool_call` 回传 |
| `src/openai/thinking.rs` | GPT hidden CoT 判定 |
| `src/kiro/model/events/base.rs` | 事件类型；未知类型 WARN 丢弃 |
| `README.md` | 对外端点与 curl 示例 |
| `docs/claude-sonnet-5.md` | Sonnet 5 / thinking 说明 |
