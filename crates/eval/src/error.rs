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
}
