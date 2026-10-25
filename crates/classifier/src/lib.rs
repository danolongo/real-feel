mod fake;
mod hosted;

use std::future::Future;

use types::{ModelInput, ModelVersion, Prediction};

pub use fake::Fake;
pub use hosted::Hosted;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("backend returned status {0}")]
    Status(u16),
    #[error("backend response has no bot probability: {0}")]
    Response(String),
    #[error("the fake failed on purpose")]
    Fake,
}

pub trait Classifier {
    fn model_version(&self) -> ModelVersion;

    /// Returns one prediction for each model input, in the same order.
    fn classify(
        &self,
        inputs: &[ModelInput],
    ) -> impl Future<Output = Result<Vec<Prediction>, Error>> + Send;
}
