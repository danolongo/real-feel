use std::{env, time::Duration};

use classifier::Classifier;
use embedder::Embedder;
use queue::Queue;
use scraper::Scraper;
use sqlx::postgres::PgPoolOptions;
use types::{MetadataVersion, QuestionVersion};
use worker::{ClassifierRole, Context, ScraperRole};

fn var(name: &str) -> Result<String, String> {
    env::var(name).map_err(|_| format!("{name} is not set"))
}

fn var_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn number<T: std::str::FromStr>(name: &str) -> Result<T, String> {
    var(name)?
        .parse()
        .map_err(|_| format!("{name} is not a number"))
}

async fn context(jobs: &str, dlq: &str) -> Result<Context, Box<dyn std::error::Error>> {
    Ok(Context {
        db: PgPoolOptions::new()
            .acquire_timeout(Duration::from_secs(5))
            .connect(&var("DATABASE_URL")?)
            .await?,
        jobs: Queue::connect(var(jobs)?).await,
        dlq: Queue::connect(var(dlq)?).await,
        deadline: Duration::from_secs(number("JOB_DEADLINE_SECS")?),
    })
}

async fn run_scraper(scraper: impl Scraper + Sync) -> Result<(), Box<dyn std::error::Error>> {
    ScraperRole {
        context: context("SCRAPER_QUEUE_URL", "SCRAPER_DLQ_URL").await?,
        scraper,
        classifier_jobs: Queue::connect(var("CLASSIFIER_QUEUE_URL")?).await,
        batch_size: number("BATCH_SIZE")?,
    }
    .run()
    .await;
    Ok(())
}

async fn run_classifier(
    classifier: impl Classifier + Sync,
    embedder: impl Embedder + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    ClassifierRole {
        context: context("CLASSIFIER_QUEUE_URL", "CLASSIFIER_DLQ_URL").await?,
        classifier,
        embedder,
        metadata_version: MetadataVersion(var("DEFAULT_METADATA_VERSION")?),
        question_version: QuestionVersion(var("DEFAULT_QUESTION_VERSION")?),
    }
    .run()
    .await;
    Ok(())
}

async fn run_classifier_with(
    classifier: impl Classifier + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    match var_or("EMBEDDER", "hosted").as_str() {
        "hosted" => run_classifier(classifier, embedder::Hosted::from_env()?).await,
        "fake" => run_classifier(classifier, embedder::Fake).await,
        other => Err(format!("EMBEDDER must be hosted or fake, not {other}").into()),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _telemetry = telemetry::init()?;
    let args: Vec<String> = env::args().collect();
    let role = match args.as_slice() {
        [_, flag, role] if flag == "--role" => role.as_str(),
        _ => return Err("usage: worker --role scraper|classifier".into()),
    };
    match role {
        "scraper" => match var_or("SCRAPER", "firecrawl").as_str() {
            "firecrawl" => {
                run_scraper(scraper::Firecrawl::new(var_or(
                    "FIRECRAWL_URL",
                    "http://localhost:3002",
                )))
                .await
            }
            "fixture" => {
                let path = var_or("FIXTURE_PATH", "crates/scraper/fixtures/default.json");
                run_scraper(scraper::Fixture::load(path)?).await
            }
            other => Err(format!("SCRAPER must be firecrawl or fixture, not {other}").into()),
        },
        "classifier" => match var_or("CLASSIFIER", "hosted").as_str() {
            "hosted" => run_classifier_with(classifier::Hosted::from_env()).await,
            "fake" => run_classifier_with(classifier::Fake::new(0.5, 0)).await,
            other => Err(format!("CLASSIFIER must be hosted or fake, not {other}").into()),
        },
        other => Err(format!("--role must be scraper or classifier, not {other}").into()),
    }
}
