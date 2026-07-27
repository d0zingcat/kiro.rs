# Claude Opus 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让客户端能通过 `claude-opus-5` / `claude-opus-5-thinking` 调用 Kiro 的 Claude Opus 5（1M 上下文、`-thinking` → adaptive）。

**Architecture:** 沿用 Sonnet 5（#184）增量模式：在 `map_model` 增加精确匹配、扩展 1M 窗口集合、在 `/v1/models` 挂出条目、把 `opus-5` 纳入 adaptive thinking 覆写。不抽注册表、不改宽别名、不改 README、不改 worktree。

**Tech Stack:** Rust、现有 Anthropic 兼容层（`src/anthropic/`）、`cargo test`

**Spec:** `docs/superpowers/specs/2026-07-25-claude-opus-5-design.md`

## Global Constraints

- 仅映射名称含 `opus-5` 的模型 → `claude-opus-5`（覆盖 `-thinking` 后缀）
- 不做裸 `opus` / 无版本别名映射
- 不更新 README 映射表
- 不在未传 `thinking` 时自动注入 adaptive
- 不修复 Opus 4.7 / 4.8 `-thinking` 仍走 `enabled` 的缺口
- 不改动 `.worktrees/openai-compatible-api`
- 不在 `master`/`main` 上直接提交（当前分支：`feat/claude-opus-5`）
- Commit 信息使用中文

## File Structure

| 文件 | 职责 |
|------|------|
| `src/anthropic/converter.rs` | `map_model`、`get_context_window_size`、相关单测 |
| `src/anthropic/handlers.rs` | `GET /v1/models` 列表、`override_thinking_from_model_name` |
| `docs/superpowers/specs/2026-07-25-claude-opus-5-design.md` | 已有设计（只读） |

不新增源码文件。

---

### Task 1: 模型映射与 1M 上下文窗口

**Files:**
- Modify: `src/anthropic/converter.rs`（`map_model` ~L93–104、`get_context_window_size` ~L117–121、tests 模块内 `test_map_model_sonnet_5` 附近）
- Test: 同文件 `mod tests`

**Interfaces:**
- Consumes: 现有 `pub fn map_model(model: &str) -> Option<String>`、`pub fn get_context_window_size(model: &str) -> i32`
- Produces: `map_model` 对含 `opus-5` 的输入返回 `Some("claude-opus-5")`；`get_context_window_size` 对映射结果为 `claude-opus-5` 返回 `1_000_000`

- [ ] **Step 1: 写失败单测**

在 `src/anthropic/converter.rs` 的 `mod tests` 中、`test_map_model_sonnet_5` 之后插入：

```rust
    #[test]
    fn test_map_model_opus_5() {
        assert_eq!(
            map_model("claude-opus-5"),
            Some("claude-opus-5".to_string())
        );
        assert_eq!(
            map_model("claude-opus-5-thinking"),
            Some("claude-opus-5".to_string())
        );
        assert_eq!(get_context_window_size("claude-opus-5"), 1_000_000);
        // opus-4-5 不应误匹配为 opus-5
        assert_eq!(
            map_model("claude-opus-4-5-20251101"),
            Some("claude-opus-4.5".to_string())
        );
    }
```

- [ ] **Step 2: 跑测试确认失败**

Run:

```bash
cargo test --lib test_map_model_opus_5 -- --nocapture
```

Expected: FAIL（`map_model("claude-opus-5")` 得到 `None`，或 assert 失败）

- [ ] **Step 3: 实现 `map_model` 与窗口大小**

将 opus 分支改为（`opus-5` 必须排在最前）：

```rust
    } else if model_lower.contains("opus") {
        if model_lower.contains("opus-5") {
            Some("claude-opus-5".to_string())
        } else if model_lower.contains("4-5") || model_lower.contains("4.5") {
            Some("claude-opus-4.5".to_string())
        } else if model_lower.contains("4-6") || model_lower.contains("4.6") {
            Some("claude-opus-4.6".to_string())
        } else if model_lower.contains("4-7") || model_lower.contains("4.7") {
            Some("claude-opus-4.7".to_string())
        } else if model_lower.contains("4-8") || model_lower.contains("4.8") {
            Some("claude-opus-4.8".to_string())
        } else {
            None
        }
```

更新注释与 `get_context_window_size`：

```rust
/// Sonnet 5 / Opus 5 / Opus 4.7 / 4.8 同 1M
pub fn get_context_window_size(model: &str) -> i32 {
    match map_model(model) {
        Some(mapped)
            if mapped == "claude-sonnet-5"
                || mapped == "claude-sonnet-4.6"
                || mapped == "claude-opus-5"
                || mapped == "claude-opus-4.6"
                || mapped == "claude-opus-4.7"
                || mapped == "claude-opus-4.8" =>
        {
            1_000_000
        }
        _ => 200_000,
    }
}
```

- [ ] **Step 4: 跑测试确认通过**

Run:

```bash
cargo test --lib test_map_model -- --nocapture
```

Expected: 全部 PASS（含 `test_map_model_opus_5` 与既有 sonnet/opus/haiku 测试）

- [ ] **Step 5: Commit**

```bash
git add src/anthropic/converter.rs
git commit -m "$(cat <<'EOF'
feat(anthropic): 映射 Claude Opus 5 并设置 1M 上下文

支持 claude-opus-5 / -thinking 精确映射，避免误伤 opus-4-5。

EOF
)"
```

---

### Task 2: `/v1/models` 列表与 adaptive thinking 覆写

**Files:**
- Modify: `src/anthropic/handlers.rs`（`get_models` 列表开头 ~L77、`override_thinking_from_model_name` ~L688–690）

**Interfaces:**
- Consumes: Task 1 的 `map_model("claude-opus-5*") → claude-opus-5`；现有 `Thinking` / `OutputConfig` 类型
- Produces: `/v1/models` 返回含 `claude-opus-5` 与 `claude-opus-5-thinking`；`claude-opus-5-thinking` 触发 `thinking.type = "adaptive"` 且 `output_config.effort = "high"`

- [ ] **Step 1: 在 `get_models` 列表最前插入两条记录**

在 `let models = vec![` 之后、现有 `claude-sonnet-5` 条目之前插入（`created` = 2026-07-24 00:00:00 UTC = `1784851200`）：

```rust
        Model {
            id: "claude-opus-5".to_string(),
            object: "model".to_string(),
            created: 1784851200, // Jul 24, 2026
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 5".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128_000,
        },
        Model {
            id: "claude-opus-5-thinking".to_string(),
            object: "model".to_string(),
            created: 1784851200, // Jul 24, 2026
            owned_by: "anthropic".to_string(),
            display_name: "Claude Opus 5 (Thinking)".to_string(),
            model_type: "chat".to_string(),
            max_tokens: 128_000,
        },
```

- [ ] **Step 2: 更新 `override_thinking_from_model_name`**

将判定改为：

```rust
    let is_adaptive_thinking = (model_lower.contains("opus")
        && (model_lower.contains("4-6") || model_lower.contains("4.6")))
        || model_lower.contains("sonnet-5")
        || model_lower.contains("opus-5");
```

同步更新函数注释，说明 Opus 5 / Sonnet 5 / Opus 4.6 走 adaptive。

- [ ] **Step 3: 编译与相关测试**

Run:

```bash
cargo test --lib -- --nocapture
```

Expected: PASS（至少 anthropic converter 相关测试全部通过；handlers 无现成单测时以编译通过为准）

可选冒烟（服务已启动时）：

```bash
curl -s http://127.0.0.1:8080/v1/models | rg 'claude-opus-5'
```

Expected: 输出中含 `claude-opus-5` 与 `claude-opus-5-thinking`

- [ ] **Step 4: Commit**

```bash
git add src/anthropic/handlers.rs
git commit -m "$(cat <<'EOF'
feat(anthropic): 在模型列表中增加 Claude Opus 5

挂出 claude-opus-5 与 -thinking 变体，thinking 后缀走 adaptive。

EOF
)"
```

---

## Spec Coverage Checklist

| Spec 要求 | 对应 Task |
|-----------|-----------|
| `map_model` 含 `opus-5` → `claude-opus-5` | Task 1 |
| 1M 上下文 | Task 1 |
| `/v1/models` 两条记录、128K max_tokens | Task 2 |
| `-thinking` → adaptive + effort high | Task 2 |
| 防 `opus-4-5` 误匹配单测 | Task 1 |
| 非目标（README / 默认 thinking / 4.7·4.8 / worktree） | 全计划不包含 |

## Out of Scope Reminders

实现时若想「顺手」改 README 或把 4.7/4.8 改成 adaptive——停住，那些在 spec 的非目标里。
