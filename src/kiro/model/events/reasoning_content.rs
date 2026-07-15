//! Reasoning / thinking 内容事件
//!
//! 处理 `reasoningContentEvent`（上游原生思考流，常见于 Opus 4.8 / Sonnet 5 等）

use serde::Deserialize;

use crate::kiro::parser::error::ParseResult;
use crate::kiro::parser::frame::Frame;

use super::base::EventPayload;

/// 原生 reasoning 内容事件
///
/// 上游以独立事件推送思考增量，典型 payload：
/// - 增量文本：`{"text":"I've"}`
/// - 收尾签名：`{"signature":"..."}`（可能无 `text`）
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningContentEvent {
    /// 思考文本增量
    #[serde(default)]
    pub text: Option<String>,

    /// 思考块签名（Anthropic 风格收尾字段，可选）
    #[serde(default)]
    pub signature: Option<String>,

    /// 捕获其他未使用字段，保证反序列化兼容
    #[serde(flatten)]
    #[allow(dead_code)]
    extra: serde_json::Value,
}

impl EventPayload for ReasoningContentEvent {
    fn from_frame(frame: &Frame) -> ParseResult<Self> {
        frame.payload_as_json()
    }
}

impl ReasoningContentEvent {
    /// 文本增量事件
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            signature: None,
            extra: serde_json::Value::Null,
        }
    }

    /// signature 收尾事件
    pub fn signature(signature: impl Into<String>) -> Self {
        Self {
            text: None,
            signature: Some(signature.into()),
            extra: serde_json::Value::Null,
        }
    }

    /// 文本增量（空串视为无）
    pub fn text_delta(&self) -> Option<&str> {
        self.text.as_deref().filter(|s| !s.is_empty())
    }

    /// 是否为带 signature 的收尾事件
    pub fn has_signature(&self) -> bool {
        self.signature
            .as_deref()
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    }
}

impl Default for ReasoningContentEvent {
    fn default() -> Self {
        Self {
            text: None,
            signature: None,
            extra: serde_json::Value::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_text_delta() {
        let event: ReasoningContentEvent =
            serde_json::from_str(r#"{"text":"I've"}"#).unwrap();
        assert_eq!(event.text_delta(), Some("I've"));
        assert!(!event.has_signature());
    }

    #[test]
    fn test_deserialize_signature_only() {
        let event: ReasoningContentEvent =
            serde_json::from_str(r#"{"signature":"EugCCnEIDxAB"}"#).unwrap();
        assert_eq!(event.text_delta(), None);
        assert!(event.has_signature());
        assert_eq!(event.signature.as_deref(), Some("EugCCnEIDxAB"));
    }

    #[test]
    fn test_deserialize_text_and_extra() {
        let event: ReasoningContentEvent =
            serde_json::from_str(r#"{"text":"ok","foo":1}"#).unwrap();
        assert_eq!(event.text_delta(), Some("ok"));
    }
}
