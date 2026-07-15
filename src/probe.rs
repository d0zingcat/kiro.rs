//! 探测 Amazon Q 上游流式事件（meteringEvent / contextUsageEvent 等）

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use uuid::Uuid;

use crate::http_client::ProxyConfig;
use crate::kiro::endpoint::{IdeEndpoint, KiroEndpoint};
use crate::kiro::model::credentials::CredentialsConfig;
use crate::kiro::model::events::Event;
use crate::kiro::model::requests::conversation::{
    ConversationState, CurrentMessage, UserInputMessage,
};
use crate::kiro::model::requests::kiro::KiroRequest;
use crate::kiro::parser::decoder::EventStreamDecoder;
use crate::kiro::provider::KiroProvider;
use crate::kiro::token_manager::MultiTokenManager;
use crate::model::config::Config;

pub struct ProbeOptions {
    pub config_path: String,
    pub credentials_path: String,
    pub model: String,
    pub message: String,
    pub usage_limits: bool,
}

pub async fn run_probe(opts: ProbeOptions) -> anyhow::Result<()> {
    let config = Config::load(&opts.config_path)?;
    let credentials_config = CredentialsConfig::load(&opts.credentials_path)?;
    let is_multiple = credentials_config.is_multiple();
    let credentials_list = credentials_config.into_sorted_credentials();

    let proxy_config = config.proxy_url.as_ref().map(|url| {
        let mut proxy = ProxyConfig::new(url);
        if let (Some(username), Some(password)) = (&config.proxy_username, &config.proxy_password) {
            proxy = proxy.with_auth(username, password);
        }
        proxy
    });

    let mut endpoints: HashMap<String, Arc<dyn KiroEndpoint>> = HashMap::new();
    let ide = IdeEndpoint::new();
    endpoints.insert(ide.name().to_string(), Arc::new(ide));

    let token_manager = Arc::new(MultiTokenManager::new(
        config.clone(),
        credentials_list,
        proxy_config.clone(),
        Some(opts.credentials_path.into()),
        is_multiple,
    )?);
    let provider = KiroProvider::with_proxy(
        token_manager.clone(),
        proxy_config,
        endpoints,
        config.default_endpoint.clone(),
    );

    if opts.usage_limits {
        print_usage_limits("请求前", &token_manager).await;
    }

    let state = ConversationState::new(Uuid::new_v4().to_string())
        .with_agent_task_type("vibe")
        .with_chat_trigger_type("MANUAL")
        .with_current_message(CurrentMessage::new(UserInputMessage::new(
            &opts.message,
            &opts.model,
        )));

    let request = KiroRequest {
        conversation_state: state,
        profile_arn: None,
    };
    let request_body = serde_json::to_string(&request)?;

    println!("\n=== 上游流式探测 ===");
    println!("model: {}", opts.model);
    println!("message: {}", opts.message);
    println!("request bytes: {}\n", request_body.len());

    let api_result = provider.call_api_stream(&request_body).await?;
    println!("HTTP status: {}\n", api_result.response.status());
    let response = api_result.response;

    let mut stream = response.bytes_stream();
    let mut decoder = EventStreamDecoder::new();

    let mut event_counts: HashMap<String, usize> = HashMap::new();
    let mut assistant_chars = 0usize;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        if let Err(e) = decoder.feed(&chunk) {
            eprintln!("[buffer error] {e}");
            continue;
        }

        for result in decoder.decode_iter() {
            let frame = match result {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[frame decode error] {e}");
                    continue;
                }
            };

            let event_type = frame.event_type().unwrap_or("(no event type)").to_string();
            let payload = frame.payload_as_str();
            *event_counts.entry(event_type.clone()).or_default() += 1;

            match Event::from_frame(frame) {
                Ok(event) => match &event {
                    Event::AssistantResponse(resp) => {
                        assistant_chars += resp.content.len();
                        print!("{}", resp.content);
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }
                    Event::ContextUsage(ctx) => {
                        println!(
                            "\n\n[contextUsageEvent] percentage={:.4}% payload={}",
                            ctx.context_usage_percentage, payload
                        );
                    }
                    Event::Metering(metering) => {
                        println!("\n\n[meteringEvent] {}", metering);
                        println!("  payload={payload}");
                    }
                    Event::ToolUse(tool) => {
                        println!(
                            "\n\n[toolUseEvent] name={} id={} stop={} payload={}",
                            tool.name, tool.tool_use_id, tool.stop, payload
                        );
                    }
                    Event::ReasoningContent(reasoning) => {
                        if let Some(text) = reasoning.text_delta() {
                            print!("{text}");
                            std::io::Write::flush(&mut std::io::stdout()).ok();
                        }
                        if reasoning.has_signature() {
                            println!(
                                "\n\n[reasoningContentEvent] signature_len={}",
                                reasoning.signature.as_deref().unwrap_or("").len()
                            );
                        }
                    }
                    Event::Error {
                        error_code,
                        error_message,
                    } => {
                        println!(
                            "\n\n[error] code={error_code} message={error_message} payload={payload}"
                        );
                    }
                    Event::Exception {
                        exception_type,
                        message,
                    } => {
                        println!(
                            "\n\n[exception] type={exception_type} message={message} payload={payload}"
                        );
                    }
                    Event::Unknown {} => {
                        println!("\n\n[unknown event] type={event_type} payload={payload}");
                    }
                },
                Err(e) => {
                    println!("\n\n[parse error] type={event_type} err={e} payload={payload}");
                }
            }
        }
    }

    println!("\n\n=== 事件统计 ===");
    let mut types: Vec<_> = event_counts.iter().collect();
    types.sort_by_key(|(k, _)| (*k).clone());
    for (ty, count) in types {
        println!("  {ty}: {count}");
    }
    println!("assistant chars: {assistant_chars}");
    let total_events: usize = event_counts.values().sum();
    println!("total events: {total_events}");

    if opts.usage_limits {
        print_usage_limits("请求后", &token_manager).await;
    }

    Ok(())
}

async fn print_usage_limits(label: &str, token_manager: &Arc<MultiTokenManager>) {
    let ctx = match token_manager.acquire_context(None).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[{label} getUsageLimits] 获取凭据失败: {e}");
            return;
        }
    };

    match token_manager.get_usage_limits_for(ctx.id).await {
        Ok(usage) => {
            println!(
                "\n[{label} getUsageLimits] subscription={:?} current={:.4} limit={:.4} remaining={:.4}",
                usage.subscription_title(),
                usage.current_usage(),
                usage.usage_limit(),
                (usage.usage_limit() - usage.current_usage()).max(0.0),
            );
            if let Some(breakdown) = usage.usage_breakdown_list.first() {
                println!(
                    "  breakdown: current={:.4} limit={:.4} bonuses={} free_trial={:?}",
                    breakdown.current_usage_with_precision,
                    breakdown.usage_limit_with_precision,
                    breakdown.bonuses.len(),
                    breakdown
                        .free_trial_info
                        .as_ref()
                        .map(|t| format!(
                            "status={:?} current={:.4} limit={:.4}",
                            t.free_trial_status,
                            t.current_usage_with_precision,
                            t.usage_limit_with_precision
                        )),
                );
            }
        }
        Err(e) => eprintln!("[{label} getUsageLimits] 查询失败: {e}"),
    }
}
