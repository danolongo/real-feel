use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use types::{ModelInput, ModelVersion, Prediction};

use crate::{Classifier, Error};

/// Returns a bot probability for every model input. Fails its first `failures` calls.
pub struct Fake {
    p_bot: Option<f64>,
    failures: usize,
    calls: AtomicUsize,
}

impl Fake {
    /// Returns `p_bot` for every model input.
    pub fn new(p_bot: f64, failures: usize) -> Self {
        Self {
            p_bot: Some(p_bot),
            failures,
            calls: AtomicUsize::new(0),
        }
    }

    /// Returns a bot probability in [0, 1) from a stable hash of the tweet ID. One tweet ID always gives the same bot probability.
    pub fn by_tweet_id(failures: usize) -> Self {
        Self {
            p_bot: None,
            failures,
            calls: AtomicUsize::new(0),
        }
    }
}

/// FNV-1a, 64 bits. Unlike `DefaultHasher`, its output never changes between platforms or Rust versions.
fn p_bot_from_tweet_id(tweet_id: &str) -> f64 {
    let hash = tweet_id
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    // The top 53 bits fill the mantissa of an f64 exactly.
    (hash >> 11) as f64 / (1u64 << 53) as f64
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
            .map(|input| {
                let p_bot = self
                    .p_bot
                    .unwrap_or_else(|| p_bot_from_tweet_id(&input.tweet_id));
                Prediction {
                    tweet_id: input.tweet_id.clone(),
                    model_version: self.model_version(),
                    metadata_version: input.metadata_version.clone(),
                    question_version: input.question_version.clone(),
                    p_bot,
                    p_llm_text: None,
                    raw: json!({ "p_bot": p_bot }),
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use types::{MetadataVersion, Question, QuestionVersion};

    use super::*;

    fn input(tweet_id: &str) -> ModelInput {
        ModelInput {
            tweet_id: tweet_id.into(),
            metadata_version: MetadataVersion("v1".into()),
            question_version: QuestionVersion("v1".into()),
            state: json!("hello"),
            questions: vec![Question {
                id: "bot".into(),
                instructions: "Is this account a bot?".into(),
            }],
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

    #[tokio::test]
    async fn by_tweet_id_gives_each_tweet_its_own_stable_bot_probability() {
        let fake = Fake::by_tweet_id(0);
        let first = fake.classify(&[input("1"), input("2")]).await.unwrap();
        let again = fake.classify(&[input("1")]).await.unwrap();
        assert_ne!(first[0].p_bot, first[1].p_bot);
        assert_eq!(first[0].p_bot, again[0].p_bot);
        assert!(first.iter().all(|p| (0.0..1.0).contains(&p.p_bot)));
        assert!(first.iter().all(|p| p.p_llm_text.is_none()));
    }
}
