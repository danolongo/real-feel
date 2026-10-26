use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot read or write {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("line {line} is not valid: {source}")]
    Line {
        line: usize,
        source: serde_json::Error,
    },
    #[error("cannot write JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("account {account_id} has more than one label record")]
    DuplicateLabelRecord { account_id: String },
    #[error("twitterapi.io call failed: {0}")]
    Scraper(#[from] scraper::Error),
    #[error("{0} is not set")]
    MissingSetting(&'static str),
    #[error("{0} is not a number")]
    BadSetting(&'static str),
    #[error("the queries file holds no query")]
    NoQueries,
    #[error("cannot build a model input: {0}")]
    Metadata(#[from] metadata::Error),
    #[error("the backend failed: {0}")]
    Classifier(#[from] classifier::Error),
    #[error("Postgres failed: {0}")]
    Db(#[from] sqlx::Error),
    #[error("the backend answered {got} predictions for {sent} model inputs")]
    PredictionCount { sent: usize, got: usize },
    #[error("experiment {0} does not exist")]
    NoSuchExperiment(uuid::Uuid),
    #[error("the stored label {0:?} is not human, bot, or agent")]
    BadLabel(String),
    #[error("the eval set holds no item")]
    EmptyEvalSet,
}
