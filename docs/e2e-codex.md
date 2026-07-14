# Codex + kiro.rs 本地 E2E 记录

本文记录用 **Codex CLI（headless）** 对接本地 **kiro.rs OpenAI 兼容端点**（`POST /v1/responses`）做端到端验证的流程、配置与实测结果。目标：确认 Codex 能经代理完成工具调用、多轮对话、大上下文与真实写代码场景，且**不走本机 ChatGPT 订阅**。

> 日期：2026-07-10 ~ 2026-07-14（`feat/openai-compatible-api` 分支）  
> Codex CLI：`0.144.1`  
> 工作区产物默认落在：`/tmp/kiro-e2e/`

---

## 1. 结论摘要

| 场景 | 模型 | 结果 |
|------|------|------|
| 多轮 + 工具 + 大上下文（3 turn） | `claude-haiku-4-5-20251001` | ✅ 通过；Turn3 ~236k input tokens |
| 从零生成可玩贪食蛇 | `claude-haiku-4-5-20251001` | ✅ `snake.html` + README；exit 0 |
| Opus 迭代 Neon Snake | `claude-opus-4-8` | ✅ 单文件增强版；exit 0 |
| 流量是否打到 ChatGPT | — | ❌ 否（隔离 `CODEX_HOME` + `base_url=127.0.0.1:8990`） |

Codex 侧曾出现 `GET /v1/models` 解析告警（期望字段 `models`，本代理返回 OpenAI 式 `data`），**不影响** `responses` 主路径。详见文末「已知问题」。

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
| Codex tools 扁平格式 | `422`：`tools[0]: missing field function` | `src/openai/types.rs` 宽松反序列化；跳过非 `function` 工具 |
| 凭据 `endpoint: cli` | 启动失败：未知端点 | 改为 `ide`（或注册对应端点） |
| 误用用户 `~/.codex` | 可能打到 ChatGPT | 强制 `CODEX_HOME` + 自定义 `model_providers.kiro` |
| `/v1/models` 形状 | Codex stderr：`missing field models` | 启动刷模型列表失败，主会话仍可用；见下节 |
| `reasoningContentEvent` | WARN 静默丢弃 | Opus 等会发原生 reasoning 事件；当前不转发，不影响工具/终稿；见下节 |

---

## 9. 已知问题（观察项）

### 9.1 Codex 刷新 models 列表失败

Codex 期望类似 `{ "models": [...] }`，kiro.rs 的 `GET /v1/models` 返回 Anthropic/OpenAI 常见的 `{ "object":"list", "data":[...] }`。会在 stderr 打 ERROR，但指定了 `model` + `model_provider` 后 **不影响** `POST /v1/responses`。

若要消除噪音，可后续为 Codex 增加兼容字段或独立 models 视图（未做）。

### 9.2 `reasoningContentEvent`（已接入 Responses）

上游（例如 Opus 4.8）会推送：

```text
event_type=reasoningContentEvent
payload ≈ {"text":"..."} 或带 signature 的收尾块
```

当前行为（coding / Codex 场景）：

- ✅ `POST /v1/responses` 流式与非流式：转发为 Responses `reasoning` item（`response.reasoning_text.delta` 等）
- ✅ 不依赖模型名 `-thinking` 后缀（原生事件始终转发）
- ⚪ Chat Completions：暂不专门转发（本场景以 Responses 为准）
- ⚪ Anthropic `/v1/messages`：暂未映射为 `thinking` block（可后续补）
- `signature`：用于关闭当前 reasoning item；尚未作为 OpenAI `encrypted_content` 原样回传

---

## 10. 建议复测清单

- [ ] `cargo build --release`（或当前 feature 组合）后启动 `-c` + `--credentials`
- [ ] `curl`：`/v1/models`、`/v1/chat/completions`、`/v1/responses`（流式一次）
- [ ] `CODEX_HOME=... codex exec ...` 小任务（读文件 + 写文件）
- [ ] 确认 kiro 日志模型名与配置一致，且无 `gpt-*`
- [ ] （可选）`codex exec resume` 第二轮
- [ ] （可选）换 `claude-opus-4-8` 再跑一轮写代码任务

---

## 11. 相关文件

| 路径 | 说明 |
|------|------|
| `src/openai/` | OpenAI 兼容层（与 Anthropic 解耦） |
| `src/openai/types.rs` | Codex 扁平 tools 兼容 |
| `src/openai/responses_stream.rs` | Responses SSE |
| `src/kiro/model/events/base.rs` | 事件类型；未知类型 WARN 丢弃 |
| `README.md` | 对外端点与 curl 示例 |
| `docs/claude-sonnet-5.md` | Sonnet 5 / thinking 说明 |
