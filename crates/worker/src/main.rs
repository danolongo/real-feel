use std::{env, time::Duration};

use classifier::Classifier;
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
) -> Result<(), Box<dyn std::error::Error>> {
    ClassifierRole {
        context: context("CLASSIFIER_QUEUE_URL", "CLASSIFIER_DLQ_URL").await?,
        classifier,
        metadata_version: MetadataVersion(var("DEFAULT_METADATA_VERSION")?),
        question_version: QuestionVersion(var("DEFAULT_QUESTION_VERSION")?),
    }
    .run()
    .await;
    Ok(())
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
        "scraper" => match var_or("SCRAPER", "twitterapi").as_str() {
            "twitterapi" => {
                run_scraper(scraper::TwitterApi::new(
                    "https://api.twitterapi.io",
                    var("TWITTERAPI_KEY")?,
                    var_or("TWITTERAPI_PAGES", "5")
                        .parse()
                        .map_err(|_| "TWITTERAPI_PAGES is not a number")?,
                    Duration::from_secs(
                        var_or("TWITTERAPI_INTERVAL_SECS", "5")
                            .parse()
                            .map_err(|_| "TWITTERAPI_INTERVAL_SECS is not a number")?,
                    ),
                ))
                .await
            }
            "fixture" => {
                let path = var_or("FIXTURE_PATH", "crates/scraper/fixtures/default.json");
                run_scraper(scraper::Fixture::load(path)?).await
            }
            other => Err(format!("SCRAPER must be twitterapi or fixture, not {other}").into()),
        },
        "classifier" => match var_or("CLASSIFIER", "hosted").as_str() {
            "hosted" => run_classifier(classifier::Hosted::from_env()).await,
            "fake" => run_classifier(classifier::Fake::new(0.5, 0)).await,
            other => Err(format!("CLASSIFIER must be hosted or fake, not {other}").into()),
        },
        other => Err(format!("--role must be scraper or classifier, not {other}").into()),
    }
}
