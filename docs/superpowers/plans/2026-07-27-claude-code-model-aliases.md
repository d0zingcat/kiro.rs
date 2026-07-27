# Claude Code 模型别名 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让客户端传入裸别名 `opus` / `sonnet` / `haiku` 时，映射到当前最新可用 Kiro 模型（Opus 5 / Sonnet 5 / Haiku 4.5）。

**Architecture:** 在共享的 `map_model` 入口增加 trim + lowercase 精确别名表；未命中则保持现有 contains / 宽兜底逻辑。不改 `/v1/models`，不抽注册表，不做 `[1m]` / `fable` / `opusplan`。

**Tech Stack:** Rust、`src/common/converter.rs`（实现）、`src/anthropic/converter.rs`（既有 `map_model` 单测）、`cargo test`

**Spec:** `docs/superpowers/specs/2026-07-27-claude-code-model-aliases-design.md`

## Global Constraints

- 仅精确匹配裸别名：`opus` → `claude-opus-5`，`sonnet` → `claude-sonnet-5`，`haiku` → `claude-haiku-4.5`
- 匹配前必须 `trim` + `to_lowercase`
- 不改变宽兜底：`claude-opus-4` 仍 → `claude-opus-4.6`，`claude-sonnet-4` 仍 → `claude-sonnet-4.5`
- 不改 `/v1/models`
- 不做 `sonnet[1m]` / `opus[1m]` / `best` / `fable` / `default` / `opusplan`
- 不改动 `.worktrees/`
- 不在 `master`/`main` 上直接提交（当前分支：`feat/claude-code-model-aliases`）
- Commit 信息使用中文

## File Structure

| 文件 | 职责 |
|------|------|
| `src/common/converter.rs` | `map_model` 实现（别名表） |
| `src/anthropic/converter.rs` | 既有 `map_model` 单测模块（re-export 自 common） |
| `docs/superpowers/specs/2026-07-27-claude-code-model-aliases-design.md` | 已有设计（只读） |

不新增源码文件。

---

### Task 1: 精确别名映射

**Files:**
- Modify: `src/common/converter.rs`（`map_model` ~L84–122）
- Modify: `src/anthropic/converter.rs`（`mod tests` 内 `test_map_model_opus_5` 附近）
- Test: `src/anthropic/converter.rs` `mod tests`

**Interfaces:**
- Consumes: 现有 `pub fn map_model(model: &str) -> Option<String>`、`pub fn get_context_window_size(model: &str) -> i32`
- Produces: 对 trim 后精确等于 `opus`/`sonnet`/`haiku`（大小写不敏感）的输入返回最新模型 ID；其它输入行为不变

- [ ] **Step 1: 写失败单测**

在 `src/anthropic/converter.rs` 的 `mod tests` 中、`test_map_model_opus_5` 之后插入：

```rust
    #[test]
    fn test_map_model_family_aliases() {
        assert_eq!(map_model("opus"), Some("claude-opus-5".to_string()));
        assert_eq!(map_model("OPUS"), Some("claude-opus-5".to_string()));
        assert_eq!(map_model(" opus "), Some("claude-opus-5".to_string()));
        assert_eq!(map_model("sonnet"), Some("claude-sonnet-5".to_string()));
        assert_eq!(map_model("SONNET"), Some("claude-sonnet-5".to_string()));
        assert_eq!(map_model("haiku"), Some("claude-haiku-4.5".to_string()));
        assert_eq!(map_model("HAIKU"), Some("claude-haiku-4.5".to_string()));

        assert_eq!(get_context_window_size("opus"), 1_000_000);
        assert_eq!(get_context_window_size("sonnet"), 1_000_000);
        assert_eq!(get_context_window_size("haiku"), 200_000);

        // 宽兜底不变：无精确版本的旧名不升到最新
        assert_eq!(
            map_model("claude-opus-4"),
            Some("claude-opus-4.6".to_string())
        );
        assert_eq!(
            map_model("claude-sonnet-4"),
            Some("claude-sonnet-4.5".to_string())
        );

        // 非精确别名不走别名表
        assert_eq!(
            map_model("claude-opus"),
            Some("claude-opus-4.6".to_string())
        );
        assert_eq!(
            map_model("opus-5"),
            Some("claude-opus-5".to_string())
        );
        assert_eq!(
            map_model("my-opus"),
            Some("claude-opus-4.6".to_string())
        );
    }
```

- [ ] **Step 2: 跑测试确认失败**

Run:

```bash
cargo test --lib test_map_model_family_aliases -- --nocapture
```

Expected: FAIL。当前裸 `opus` / `sonnet` 落入宽兜底，分别得到 `claude-opus-4.6` / `claude-sonnet-4.5`，与断言的 Opus 5 / Sonnet 5 不符。（`haiku` 目标与现有 contains 结果相同，可能单独通过；以 `opus`/`sonnet` 断言失败为准。）

- [ ] **Step 3: 实现精确别名表**

将 `src/common/converter.rs` 中的 `map_model` 改为（在 GPT 映射之后、contains 分支之前插入别名匹配；并对整段输入使用 trim）：

```rust
/// 模型映射：将 Anthropic/OpenAI 模型名映射到 Kiro 模型 ID
/// 严格对照版本号；裸别名 opus/sonnet/haiku 映射到当前最新可用模型
pub fn map_model(model: &str) -> Option<String> {
    let model_lower = model.trim().to_lowercase();

    if let Some(mapped) = map_gpt_model(&model_lower) {
        return Some(mapped);
    }

    // Claude Code 风格家族别名（精确匹配）
    match model_lower.as_str() {
        "opus" => return Some("claude-opus-5".to_string()),
        "sonnet" => return Some("claude-sonnet-5".to_string()),
        "haiku" => return Some("claude-haiku-4.5".to_string()),
        _ => {}
    }

    if model_lower.contains("sonnet") {
        if model_lower.contains("sonnet-5") {
            Some("claude-sonnet-5".to_string())
        } else if model_lower.contains("4-6") || model_lower.contains("4.6") {
            Some("claude-sonnet-4.6".to_string())
        } else if model_lower.contains("4-5") || model_lower.contains("4.5") {
            Some("claude-sonnet-4.5".to_string())
        } else {
            // 兼容旧别名（如 claude-sonnet-4 / claude-3-5-sonnet）
            Some("claude-sonnet-4.5".to_string())
        }
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
            // 兼容旧别名（如 claude-opus-4）
            Some("claude-opus-4.6".to_string())
        }
    } else if model_lower.contains("haiku") {
        Some("claude-haiku-4.5".to_string())
    } else {
        None
    }
}
```

注意：`map_gpt_model` 入参改为已 trim 的 `model_lower`；其余 contains 分支逻辑保持原样，仅依赖同一 `model_lower`。

- [ ] **Step 4: 跑测试确认通过**

Run:

```bash
cargo test --lib test_map_model -- --nocapture
```

Expected: 全部 PASS（含 `test_map_model_family_aliases` 与既有 sonnet/opus/haiku/gpt 测试）

- [ ] **Step 5: Commit**

```bash
git add \
  src/common/converter.rs \
  src/anthropic/converter.rs \
  docs/superpowers/specs/2026-07-27-claude-code-model-aliases-design.md \
  docs/superpowers/plans/2026-07-27-claude-code-model-aliases.md
git commit -m "$(cat <<'EOF'
feat: 支持 Claude Code 风格 opus/sonnet/haiku 模型别名

将裸别名精确映射到当前最新可用模型，避免用户手动切换完整版本号。
EOF
)"
```

---

## Spec coverage (self-review)

| Spec 要求 | 对应任务 |
|-----------|----------|
| `opus`/`sonnet`/`haiku` → 最新 ID | Task 1 |
| trim + 大小写不敏感 | Task 1 Step 3 |
| 宽兜底不变 | Task 1 回归断言 |
| 不改 `/v1/models` | 无改动（刻意） |
| 不做 `[1m]`/`fable`/`opusplan` 等 | Global Constraints |
| 上下文窗口随 `map_model` | Task 1 对 `get_context_window_size` 的断言 |
| 单测覆盖 | Task 1 Step 1 |
