use std::{sync::Arc, time::Duration};

use api::{AppState, router};
use db::Versions;
use queue::Queue;
use sqlx::postgres::PgPoolOptions;
use tokio::net::TcpListener;
use types::{MetadataVersion, ModelVersion, QuestionVersion};

struct Config {
    addr: String,
    database_url: String,
    scraper_queue_url: String,
    versions: Versions,
    bot_threshold: f64,
    query_reuse: Duration,
    quota_enabled: bool,
    db_max_connections: u32,
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let var = |name: &str| std::env::var(name).map_err(|_| format!("{name} is not set"));
        Ok(Self {
            addr: std::env::var("API_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string()),
            database_url: var("DATABASE_URL")?,
            scraper_queue_url: var("SCRAPER_QUEUE_URL")?,
            query_reuse: Duration::from_secs(match std::env::var("QUERY_REUSE_SECS") {
                Ok(secs) => secs
                    .parse()
                    .map_err(|_| format!("QUERY_REUSE_SECS is not a number: {secs}"))?,
                Err(_) => 3600,
            }),
            db_max_connections: match std::env::var("DB_MAX_CONNECTIONS") {
                Ok(count) => count
                    .parse()
                    .map_err(|_| format!("DB_MAX_CONNECTIONS is not a number: {count}"))?,
                Err(_) => 10,
            },
            quota_enabled: parse_quota_enabled(
                std::env::var("QUERY_QUOTA_ENABLED").ok().as_deref(),
            )?,
            bot_threshold: {
                let threshold = var("BOT_THRESHOLD")?;
                threshold
                    .parse()
                    .map_err(|_| format!("BOT_THRESHOLD is not a number: {threshold}"))?
            },
            versions: Versions {
                model: ModelVersion(var("DEFAULT_MODEL_VERSION")?),
                metadata: MetadataVersion(var("DEFAULT_METADATA_VERSION")?),
                question: QuestionVersion(var("DEFAULT_QUESTION_VERSION")?),
            },
        })
    }
}

fn parse_quota_enabled(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(other) => Err(format!(
            "QUERY_QUOTA_ENABLED must be true or false: {other}"
        )),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _telemetry = telemetry::init("api")?;
    let config = Config::from_env()?;

    let db = PgPoolOptions::new()
        .max_connections(config.db_max_connections)
        .acquire_timeout(Duration::from_secs(5))
        .connect(&config.database_url)
        .await?;
    db::MIGRATOR.run(&db).await?;
    tracing::info!("migrations applied");

    let state = AppState {
        db,
        queue: Queue::connect(config.scraper_queue_url).await,
        versions: Arc::new(config.versions),
        bot_threshold: config.bot_threshold,
        query_reuse: config.query_reuse,
        quota_enabled: config.quota_enabled,
    };
    if !config.quota_enabled {
        tracing::warn!("the query quota is off");
    }

    let listener = TcpListener::bind(&config.addr).await?;
    tracing::info!(addr = %config.addr, "api listening");

    axum::serve(listener, router(state))
        .with_graceful_shutdown(telemetry::shutdown_signal())
        .await?;
    tracing::info!("api stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_quota_enabled;

    #[test]
    fn quota_is_on_when_unset_or_true() {
        assert_eq!(parse_quota_enabled(None), Ok(true));
        assert_eq!(parse_quota_enabled(Some("true")), Ok(true));
        assert_eq!(parse_quota_enabled(Some("false")), Ok(false));
    }

    #[test]
    fn any_other_value_fails() {
        for value in ["", "TRUE", "0", "off", "yes"] {
            let error = parse_quota_enabled(Some(value)).unwrap_err();
            assert!(error.contains("QUERY_QUOTA_ENABLED"), "{error}");
        }
    }
}
