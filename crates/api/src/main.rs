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
}

impl Config {
    fn from_env() -> Result<Self, String> {
        let var = |name: &str| std::env::var(name).map_err(|_| format!("{name} is not set"));
        Ok(Self {
            addr: std::env::var("API_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string()),
            database_url: var("DATABASE_URL")?,
            scraper_queue_url: var("SCRAPER_QUEUE_URL")?,
            versions: Versions {
                model: ModelVersion(var("DEFAULT_MODEL_VERSION")?),
                metadata: MetadataVersion(var("DEFAULT_METADATA_VERSION")?),
                question: QuestionVersion(var("DEFAULT_QUESTION_VERSION")?),
            },
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _telemetry = telemetry::init()?;
    let config = Config::from_env()?;

    let state = AppState {
        db: PgPoolOptions::new()
            .acquire_timeout(Duration::from_secs(5))
            .connect(&config.database_url)
            .await?,
        queue: Queue::connect(config.scraper_queue_url).await,
        versions: Arc::new(config.versions),
    };

    let listener = TcpListener::bind(&config.addr).await?;
    tracing::info!(addr = %config.addr, "api listening");

    axum::serve(listener, router(state)).await?;
    Ok(())
}
