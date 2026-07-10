use clap::Parser;

/// Anthropic <-> Kiro API 客户端
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// 配置文件路径
    #[arg(short, long)]
    pub config: Option<String>,

    /// 凭证文件路径
    #[arg(long)]
    pub credentials: Option<String>,

    /// 探测上游流式事件（不启动 HTTP 服务）
    #[arg(long)]
    pub probe_events: bool,

    /// 探测使用的模型
    #[arg(long, default_value = "claude-haiku-4.5")]
    pub probe_model: String,

    /// 探测发送的消息
    #[arg(long, default_value = "Say hi in one short sentence.")]
    pub probe_message: String,

    /// 探测时跳过 getUsageLimits 前后对比
    #[arg(long)]
    pub probe_no_usage_limits: bool,
}
