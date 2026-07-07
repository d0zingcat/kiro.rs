//! 计费事件
//!
//! 处理 meteringEvent 类型的事件（上游按 credit 计费）

use serde::Deserialize;

use crate::kiro::parser::error::ParseResult;
use crate::kiro::parser::frame::Frame;

use super::base::EventPayload;

/// 计费事件
///
/// 上游在流式响应末尾返回本次请求的 credits 消耗量。
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeteringEvent {
    /// 计费单位（通常为 "credit"）
    #[serde(default)]
    pub unit: Option<String>,

    /// 计费单位复数形式（通常为 "credits"）
    #[serde(default)]
    pub unit_plural: Option<String>,

    /// 本次请求消耗的 credits（浮点数）
    #[serde(default)]
    pub usage: f64,
}

impl EventPayload for MeteringEvent {
    fn from_frame(frame: &Frame) -> ParseResult<Self> {
        frame.payload_as_json()
    }
}

impl MeteringEvent {
    /// 是否为 credit 计费
    pub fn is_credit(&self) -> bool {
        self.unit
            .as_deref()
            .map(|u| u.eq_ignore_ascii_case("credit"))
            .unwrap_or(true)
    }
}

impl std::fmt::Display for MeteringEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let unit = self
            .unit_plural
            .as_deref()
            .or(self.unit.as_deref())
            .unwrap_or("credits");
        write!(f, "{:.6} {}", self.usage, unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_metering_event() {
        let event: MeteringEvent = serde_json::from_str(
            r#"{"unit":"credit","unitPlural":"credits","usage":0.005835419900497513}"#,
        )
        .unwrap();
        assert_eq!(event.unit.as_deref(), Some("credit"));
        assert_eq!(event.unit_plural.as_deref(), Some("credits"));
        assert!((event.usage - 0.005835419900497513).abs() < 1e-12);
        assert!(event.is_credit());
    }
}
