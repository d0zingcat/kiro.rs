//! Anthropic 兼容 usage 字段构建
//!
//! tokens 与 credits 是不同计量单位，不可互相替代：
//! - `input_tokens` / `output_tokens`：上下文 token（来自估算或 contextUsageEvent）
//! - `credits`：上游 meteringEvent 返回的计费 credits（可选）

use serde_json::{Value, json};

use crate::kiro::model::events::MeteringEvent;

/// 构建 Anthropic 兼容的 usage 对象
pub fn build_usage_value(
    input_tokens: i32,
    output_tokens: i32,
    metering: Option<&MeteringEvent>,
) -> Value {
    let mut usage = json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    });

    if let Some(metering) = metering {
        if let Some(obj) = usage.as_object_mut() {
            if metering.is_credit() {
                obj.insert("credits".to_string(), json!(metering.usage));
            }
            if let Some(unit) = &metering.unit {
                obj.insert("metering_unit".to_string(), json!(unit));
            }
        }
    }

    usage
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_usage_without_metering() {
        let usage = build_usage_value(100, 20, None);
        assert_eq!(usage["input_tokens"], 100);
        assert_eq!(usage["output_tokens"], 20);
        assert!(usage.get("credits").is_none());
    }

    #[test]
    fn test_build_usage_with_metering() {
        let metering = MeteringEvent {
            unit: Some("credit".to_string()),
            unit_plural: Some("credits".to_string()),
            usage: 0.005835,
        };
        let usage = build_usage_value(100, 20, Some(&metering));
        assert_eq!(usage["input_tokens"], 100);
        assert_eq!(usage["output_tokens"], 20);
        assert!((usage["credits"].as_f64().unwrap() - 0.005835).abs() < 1e-9);
        assert_eq!(usage["metering_unit"], "credit");
    }
}
