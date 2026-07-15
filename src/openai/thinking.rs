//! Thinking 标签解析模块（OpenAI Chat Completions）
//!
//! 从 Kiro 返回的原始文本中识别 `<thinking>...</thinking>` 标签，
//! 拆分为 reasoning（思考内容）与 content（正文内容）两部分。
//!
//! 逻辑移植自 Anthropic 兼容层的流式处理模块（`find_real_thinking_start_tag` /
//! `extract_thinking_from_complete_text` / `process_content_with_thinking`），
//! 但本模块不依赖该模块，供 OpenAI 兼容层独立使用（保持两者完全隔离）。

/// 找到小于等于目标位置的最近有效 UTF-8 字符边界
fn find_char_boundary(s: &str, target: usize) -> usize {
    if target >= s.len() {
        return s.len();
    }
    if target == 0 {
        return 0;
    }
    let mut pos = target;
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// 需要跳过的包裹字符
///
/// 当 thinking 标签被这些字符包裹时，认为是在引用标签而非真正的标签：
/// - 反引号 (`)：行内代码
/// - 双引号 (")：字符串
/// - 单引号 (')：字符串
const QUOTE_CHARS: &[u8] = &[
    b'`', b'"', b'\'', b'\\', b'#', b'!', b'@', b'$', b'%', b'^', b'&', b'*', b'(', b')', b'-',
    b'_', b'=', b'+', b'[', b']', b'{', b'}', b';', b':', b'<', b'>', b',', b'.', b'?', b'/',
];

/// 检查指定位置的字符是否是引用字符
fn is_quote_char(buffer: &str, pos: usize) -> bool {
    buffer
        .as_bytes()
        .get(pos)
        .map(|c| QUOTE_CHARS.contains(c))
        .unwrap_or(false)
}

/// 判断给定模型名是否为 thinking 模型（模型名包含 `-thinking`，大小写不敏感）
pub(crate) fn is_thinking_model(model: &str) -> bool {
    model.to_lowercase().contains("-thinking")
}

/// 查找真正的 thinking 结束标签（不被引用字符包裹，且后面有双换行符）
fn find_real_thinking_end_tag(buffer: &str) -> Option<usize> {
    const TAG: &str = "</thinking>";
    let mut search_start = 0;

    while let Some(pos) = buffer[search_start..].find(TAG) {
        let absolute_pos = search_start + pos;

        let has_quote_before = absolute_pos > 0 && is_quote_char(buffer, absolute_pos - 1);
        let after_pos = absolute_pos + TAG.len();
        let has_quote_after = is_quote_char(buffer, after_pos);

        if has_quote_before || has_quote_after {
            search_start = absolute_pos + 1;
            continue;
        }

        let after_content = &buffer[after_pos..];

        if after_content.len() < 2 {
            return None;
        }

        if after_content.starts_with("\n\n") {
            return Some(absolute_pos);
        }

        search_start = absolute_pos + 1;
    }

    None
}

/// 查找缓冲区末尾的 thinking 结束标签（允许末尾只有空白字符）
///
/// 用于"边界事件"场景：例如流结束时 `</thinking>` 后面没有 `\n\n`，
/// 此时结束标签依然应被识别并过滤。
fn find_real_thinking_end_tag_at_buffer_end(buffer: &str) -> Option<usize> {
    const TAG: &str = "</thinking>";
    let mut search_start = 0;

    while let Some(pos) = buffer[search_start..].find(TAG) {
        let absolute_pos = search_start + pos;

        let has_quote_before = absolute_pos > 0 && is_quote_char(buffer, absolute_pos - 1);
        let after_pos = absolute_pos + TAG.len();
        let has_quote_after = is_quote_char(buffer, after_pos);

        if has_quote_before || has_quote_after {
            search_start = absolute_pos + 1;
            continue;
        }

        if buffer[after_pos..].trim().is_empty() {
            return Some(absolute_pos);
        }

        search_start = absolute_pos + 1;
    }

    None
}

/// 查找真正的 thinking 开始标签（不被引用字符包裹）
pub(crate) fn find_real_thinking_start_tag(buffer: &str) -> Option<usize> {
    const TAG: &str = "<thinking>";
    let mut search_start = 0;

    while let Some(pos) = buffer[search_start..].find(TAG) {
        let absolute_pos = search_start + pos;

        let has_quote_before = absolute_pos > 0 && is_quote_char(buffer, absolute_pos - 1);
        let after_pos = absolute_pos + TAG.len();
        let has_quote_after = is_quote_char(buffer, after_pos);

        if !has_quote_before && !has_quote_after {
            return Some(absolute_pos);
        }

        search_start = absolute_pos + 1;
    }

    None
}

/// 从完整文本中提取 thinking 块（用于非流式响应）
///
/// # 返回值
/// - `(Some(thinking_content), remaining_text)` — 检测到有效 thinking 块
/// - `(None, original_text)` — 未检测到，原样返回
pub(crate) fn extract_thinking_from_complete_text(text: &str) -> (Option<String>, String) {
    let start_pos = match find_real_thinking_start_tag(text) {
        Some(pos) => pos,
        None => return (None, text.to_string()),
    };

    let before = &text[..start_pos];
    let after_open = &text[start_pos + "<thinking>".len()..];

    let (thinking_raw, text_after) = if let Some(end_pos) = find_real_thinking_end_tag(after_open)
    {
        (
            &after_open[..end_pos],
            &after_open[end_pos + "</thinking>\n\n".len()..],
        )
    } else if let Some(end_pos) = find_real_thinking_end_tag_at_buffer_end(after_open) {
        let after_tag = end_pos + "</thinking>".len();
        (&after_open[..end_pos], after_open[after_tag..].trim_start())
    } else {
        return (None, text.to_string());
    };

    let thinking_content = thinking_raw.strip_prefix('\n').unwrap_or(thinking_raw);

    let mut remaining = String::new();
    if !before.trim().is_empty() {
        remaining.push_str(before);
    }
    remaining.push_str(text_after);

    if thinking_content.is_empty() {
        (None, remaining)
    } else {
        (Some(thinking_content.to_string()), remaining)
    }
}

/// 流式解析产生的事件：思考内容（reasoning）或正文内容（content）
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ThinkingEvent {
    /// 思考内容片段（对应 OpenAI `reasoning_content`）
    Reasoning(String),
    /// 正文内容片段（对应 OpenAI `content`）
    Content(String),
}

/// 流式 thinking 标签解析器
///
/// 逐段接收原始文本（`push`），识别 `<thinking>...</thinking>` 标签，
/// 将内容拆分为 reasoning 和 content 事件序列。
/// 与 Anthropic `StreamContext::process_content_with_thinking` 逻辑一致，
/// 但不产生 SSE 事件，只产生中间事件供调用方自行封装。
#[derive(Debug, Default)]
pub(crate) struct ThinkingStreamParser {
    /// 待处理的缓冲区
    buffer: String,
    /// 是否在 thinking 块内
    in_thinking_block: bool,
    /// thinking 块是否已提取完成（一个响应只处理一次）
    thinking_extracted: bool,
    /// 是否需要剥离 thinking 内容开头的换行符
    strip_leading_newline: bool,
}

impl ThinkingStreamParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// 处理新到达的文本片段，返回本次产生的事件序列
    pub(crate) fn push(&mut self, content: &str) -> Vec<ThinkingEvent> {
        let mut events = Vec::new();
        self.buffer.push_str(content);

        loop {
            if !self.in_thinking_block && !self.thinking_extracted {
                if let Some(start_pos) = find_real_thinking_start_tag(&self.buffer) {
                    let before = self.buffer[..start_pos].to_string();
                    if !before.is_empty() && !before.trim().is_empty() {
                        events.push(ThinkingEvent::Content(before));
                    }

                    self.in_thinking_block = true;
                    self.strip_leading_newline = true;
                    self.buffer = self.buffer[start_pos + "<thinking>".len()..].to_string();
                } else {
                    // 没找到开始标签，保留可能是部分标签的内容
                    let target_len = self.buffer.len().saturating_sub("<thinking>".len());
                    let safe_len = find_char_boundary(&self.buffer, target_len);
                    if safe_len > 0 {
                        let safe_content = self.buffer[..safe_len].to_string();
                        if !safe_content.is_empty() && !safe_content.trim().is_empty() {
                            events.push(ThinkingEvent::Content(safe_content));
                            self.buffer = self.buffer[safe_len..].to_string();
                        }
                    }
                    break;
                }
            } else if self.in_thinking_block {
                if self.strip_leading_newline {
                    if self.buffer.starts_with('\n') {
                        self.buffer = self.buffer[1..].to_string();
                        self.strip_leading_newline = false;
                    } else if !self.buffer.is_empty() {
                        self.strip_leading_newline = false;
                    }
                }

                if let Some(end_pos) = find_real_thinking_end_tag(&self.buffer) {
                    let thinking_content = self.buffer[..end_pos].to_string();
                    if !thinking_content.is_empty() {
                        events.push(ThinkingEvent::Reasoning(thinking_content));
                    }

                    self.in_thinking_block = false;
                    self.thinking_extracted = true;
                    self.buffer = self.buffer[end_pos + "</thinking>\n\n".len()..].to_string();
                } else {
                    let target_len = self.buffer.len().saturating_sub("</thinking>\n\n".len());
                    let safe_len = find_char_boundary(&self.buffer, target_len);
                    if safe_len > 0 {
                        let safe_content = self.buffer[..safe_len].to_string();
                        if !safe_content.is_empty() {
                            events.push(ThinkingEvent::Reasoning(safe_content));
                        }
                        self.buffer = self.buffer[safe_len..].to_string();
                    }
                    break;
                }
            } else {
                if !self.buffer.is_empty() {
                    let remaining = std::mem::take(&mut self.buffer);
                    events.push(ThinkingEvent::Content(remaining));
                }
                break;
            }
        }

        events
    }

    /// 流结束时调用，冲刷缓冲区中剩余的内容
    pub(crate) fn finish(&mut self) -> Vec<ThinkingEvent> {
        let mut events = Vec::new();
        if self.buffer.is_empty() {
            return events;
        }

        if self.in_thinking_block {
            if let Some(end_pos) = find_real_thinking_end_tag_at_buffer_end(&self.buffer) {
                let thinking_content = self.buffer[..end_pos].to_string();
                if !thinking_content.is_empty() {
                    events.push(ThinkingEvent::Reasoning(thinking_content));
                }

                let after_pos = end_pos + "</thinking>".len();
                let remaining = self.buffer[after_pos..].trim_start().to_string();
                self.buffer.clear();
                self.in_thinking_block = false;
                self.thinking_extracted = true;
                if !remaining.is_empty() {
                    events.push(ThinkingEvent::Content(remaining));
                }
            } else {
                let remaining = std::mem::take(&mut self.buffer);
                events.push(ThinkingEvent::Reasoning(remaining));
            }
        } else {
            let remaining = std::mem::take(&mut self.buffer);
            events.push(ThinkingEvent::Content(remaining));
        }

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_real_thinking_start_tag_basic() {
        assert_eq!(find_real_thinking_start_tag("<thinking>"), Some(0));
        assert_eq!(find_real_thinking_start_tag("prefix<thinking>"), Some(6));
    }

    #[test]
    fn test_find_real_thinking_start_tag_ignores_quoted_fake_tags() {
        // 被反引号包裹的应该被跳过
        assert_eq!(find_real_thinking_start_tag("`<thinking>`"), None);
        assert_eq!(find_real_thinking_start_tag("use `<thinking>` tag"), None);

        // 被双引号 / 单引号包裹的应该被跳过
        assert_eq!(find_real_thinking_start_tag("\"<thinking>\""), None);
        assert_eq!(find_real_thinking_start_tag("'<thinking>'"), None);

        // 先有被包裹的假标签，后有真正的开始标签
        assert_eq!(
            find_real_thinking_start_tag("about `<thinking>` tag<thinking>content"),
            Some(22)
        );
    }

    #[test]
    fn test_extract_thinking_from_complete_text_basic() {
        let (thinking, remaining) =
            extract_thinking_from_complete_text("<thinking>\nabc</thinking>\n\nhello");
        assert_eq!(thinking, Some("abc".to_string()));
        assert_eq!(remaining, "hello");
    }

    #[test]
    fn test_extract_thinking_from_complete_text_no_tag() {
        let (thinking, remaining) = extract_thinking_from_complete_text("just plain text");
        assert_eq!(thinking, None);
        assert_eq!(remaining, "just plain text");
    }

    #[test]
    fn test_extract_thinking_from_complete_text_ignores_quoted_fake_tags() {
        // 反引号包裹的假标签不应被误判为真正的 thinking 标签
        let (thinking, remaining) =
            extract_thinking_from_complete_text("mention `<thinking>` in text, no real tag");
        assert_eq!(thinking, None);
        assert_eq!(remaining, "mention `<thinking>` in text, no real tag");
    }

    #[test]
    fn test_extract_thinking_from_complete_text_end_tag_at_buffer_end() {
        // 结束标签在末尾，后面没有 \n\n（边界场景）
        let (thinking, remaining) =
            extract_thinking_from_complete_text("<thinking>\nabc</thinking>");
        assert_eq!(thinking, Some("abc".to_string()));
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_is_thinking_model() {
        assert!(is_thinking_model("claude-sonnet-4-6-thinking"));
        assert!(is_thinking_model("claude-opus-4-6-THINKING"));
        assert!(!is_thinking_model("claude-sonnet-4-6"));
    }

    #[test]
    fn test_parser_basic_split() {
        let mut parser = ThinkingStreamParser::new();
        let events = parser.push("<thinking>\nabc</thinking>\n\nhello");
        assert_eq!(
            events,
            vec![
                ThinkingEvent::Reasoning("abc".to_string()),
                ThinkingEvent::Content("hello".to_string()),
            ]
        );
    }

    #[test]
    fn test_parser_cross_chunk_split() {
        let mut parser = ThinkingStreamParser::new();
        let mut all = Vec::new();
        all.extend(parser.push("<thin"));
        all.extend(parser.push("king>\nhel"));
        all.extend(parser.push("lo</thinking>\n\nwor"));
        all.extend(parser.push("ld"));
        all.extend(parser.finish());

        let reasoning: String = all
            .iter()
            .filter_map(|e| match e {
                ThinkingEvent::Reasoning(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        let content: String = all
            .iter()
            .filter_map(|e| match e {
                ThinkingEvent::Content(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(reasoning, "hello");
        assert_eq!(content, "world");
    }

    #[test]
    fn test_parser_finish_flushes_open_thinking_block_without_double_newline() {
        let mut parser = ThinkingStreamParser::new();
        let mut all = Vec::new();
        all.extend(parser.push("<thinking>abc</thinking>"));
        all.extend(parser.finish());

        let reasoning: String = all
            .iter()
            .filter_map(|e| match e {
                ThinkingEvent::Reasoning(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();

        assert!(!reasoning.contains("</thinking>"));
        assert_eq!(reasoning, "abc");
        assert!(all.iter().all(|e| !matches!(e, ThinkingEvent::Content(_))));
    }

    #[test]
    fn test_parser_no_thinking_tag_passes_through_as_content() {
        let mut parser = ThinkingStreamParser::new();
        let mut all = Vec::new();
        all.extend(parser.push("just plain text"));
        all.extend(parser.finish());

        assert!(all.iter().all(|e| matches!(e, ThinkingEvent::Content(_))));
        let content: String = all
            .iter()
            .map(|e| match e {
                ThinkingEvent::Content(s) => s.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(content, "just plain text");
    }
}
