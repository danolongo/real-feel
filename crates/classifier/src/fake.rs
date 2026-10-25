use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use types::{ModelInput, ModelVersion, Prediction};

use crate::{Classifier, Error};

/// Returns the same bot probability for every model input. Fails its first `failures` calls.
pub struct Fake {
    p_bot: f64,
    failures: usize,
    calls: AtomicUsize,
}

impl Fake {
    pub fn new(p_bot: f64, failures: usize) -> Self {
        Self {
            p_bot,
            failures,
            calls: AtomicUsize::new(0),
        }
    }
}

impl Classifier for Fake {
    fn model_version(&self) -> ModelVersion {
        ModelVersion("fake".into())
    }

    async fn classify(&self, inputs: &[ModelInput]) -> Result<Vec<Prediction>, Error> {
        if self.calls.fetch_add(1, Ordering::SeqCst) < self.failures {
            return Err(Error::Fake);
        }
        Ok(inputs
            .iter()
            .map(|input| Prediction {
                tweet_id: input.tweet_id.clone(),
                model_version: self.model_version(),
                metadata_version: input.metadata_version.clone(),
                question_version: input.question_version.clone(),
                p_bot: self.p_bot,
                raw: json!({ "p_bot": self.p_bot }),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use types::{MetadataVersion, QuestionVersion};

    use super::*;

    fn input(tweet_id: &str) -> ModelInput {
        ModelInput {
            tweet_id: tweet_id.into(),
            metadata_version: MetadataVersion("v1".into()),
            question_version: QuestionVersion("v1".into()),
            text: "hello".into(),
            question: "Is this account a bot?".into(),
        }
    }

    #[tokio::test]
    async fn returns_one_prediction_for_each_input_in_order() {
        let fake = Fake::new(0.8, 0);
        let predictions = fake.classify(&[input("2"), input("1")]).await.unwrap();
        let ids: Vec<_> = predictions.iter().map(|p| p.tweet_id.as_str()).collect();
        assert_eq!(ids, ["2", "1"]);
        assert!(predictions.iter().all(|p| p.p_bot == 0.8));
        assert_eq!(predictions[0].model_version, ModelVersion("fake".into()));
        assert_eq!(
            predictions[0].metadata_version,
            MetadataVersion("v1".into())
        );
    }

    #[tokio::test]
    async fn fails_its_first_calls_then_succeeds() {
        let fake = Fake::new(0.1, 2);
        assert!(matches!(
            fake.classify(&[input("1")]).await,
            Err(Error::Fake)
        ));
        assert!(matches!(
            fake.classify(&[input("1")]).await,
            Err(Error::Fake)
        ));
        assert!(fake.classify(&[input("1")]).await.is_ok());
    }
}
