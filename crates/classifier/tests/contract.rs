#![cfg(feature = "contract")]

use classifier::{Classifier, Hosted};
use serde_json::json;
use types::{MetadataVersion, ModelInput, Question, QuestionVersion};

#[tokio::test]
async fn hosted_backend_returns_a_bot_probability() {
    let input = ModelInput {
        tweet_id: "1".into(),
        metadata_version: MetadataVersion("v1".into()),
        question_version: QuestionVersion("v1".into()),
        state: json!("Follow me for free crypto giveaways! Link in bio. #airdrop"),
        questions: vec![Question {
            id: "bot".into(),
            instructions: "Is the account behind this tweet a bot?".into(),
        }],
    };

    let predictions = Hosted::from_env().classify(&[input]).await.unwrap();

    assert_eq!(predictions.len(), 1);
    assert!((0.0..=1.0).contains(&predictions[0].p_bot));
}
