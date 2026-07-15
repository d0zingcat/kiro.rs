//! OpenAI Usage 构建辅助

use crate::kiro::model::events::MeteringEvent;

use super::types::Usage;

/// 根据 token 计数与可选 metering 事件构建 Usage
pub fn build_usage(
    prompt_tokens: i32,
    completion_tokens: i32,
    metering: Option<&MeteringEvent>,
) -> Usage {
    let mut usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        credits: None,
        metering_unit: None,
    };
    if let Some(m) = metering {
        if m.is_credit() {
            usage.credits = Some(m.usage);
        }
        usage.metering_unit = m.unit.clone();
    }
    usage
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::model::events::MeteringEvent;

    #[test]
    fn usage_without_metering() {
        let u = build_usage(10, 5, None);
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 15);
        assert!(u.credits.is_none());
    }

    #[test]
    fn usage_with_credits() {
        let m = MeteringEvent {
            unit: Some("credit".into()),
            unit_plural: Some("credits".into()),
            usage: 0.01,
        };
        let u = build_usage(10, 5, Some(&m));
        assert_eq!(u.credits, Some(0.01));
        assert_eq!(u.metering_unit.as_deref(), Some("credit"));
    }
}
