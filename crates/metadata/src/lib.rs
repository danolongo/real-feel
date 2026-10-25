use types::{Account, MetadataVersion, ModelInput, QuestionVersion, Tweet};

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("unknown metadata version: {0}")]
    UnknownMetadataVersion(String),
    #[error("unknown question version: {0}")]
    UnknownQuestionVersion(String),
}

const QUESTION_V1: &str = "Is the account behind this tweet a bot?";

/// Builds the model input for one tweet. The same inputs always give the same model input.
pub fn build(
    tweet: &Tweet,
    _account: &Account,
    metadata_version: &MetadataVersion,
    question_version: &QuestionVersion,
) -> Result<ModelInput, Error> {
    let text = match metadata_version.0.as_str() {
        "v1" => tweet.text.clone(),
        other => return Err(Error::UnknownMetadataVersion(other.into())),
    };
    let question = match question_version.0.as_str() {
        "v1" => QUESTION_V1.to_string(),
        other => return Err(Error::UnknownQuestionVersion(other.into())),
    };
    Ok(ModelInput {
        tweet_id: tweet.id.clone(),
        metadata_version: metadata_version.clone(),
        question_version: question_version.clone(),
        text,
        question,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tweet() -> Tweet {
        Tweet {
            id: "1".into(),
            account_id: "a1".into(),
            text: "hello world".into(),
            posted_at: None,
            raw: json!({"lang": "en"}),
        }
    }

    fn account(handle: &str) -> Account {
        Account {
            id: "a1".into(),
            handle: handle.into(),
            joined_at: None,
            followers: None,
            following: None,
            tweet_count: None,
            is_verified: None,
            is_automated: None,
            raw: json!({"followers": 12}),
        }
    }

    fn v1() -> (MetadataVersion, QuestionVersion) {
        (MetadataVersion("v1".into()), QuestionVersion("v1".into()))
    }

    #[test]
    fn equal_inputs_give_equal_model_inputs() {
        let (metadata_version, question_version) = v1();
        let first = build(
            &tweet(),
            &account("x"),
            &metadata_version,
            &question_version,
        );
        let second = build(
            &tweet(),
            &account("x"),
            &metadata_version,
            &question_version,
        );
        assert_eq!(first, second);
    }

    #[test]
    fn v1_holds_only_the_tweet_text() {
        let (metadata_version, question_version) = v1();
        let a = build(
            &tweet(),
            &account("x"),
            &metadata_version,
            &question_version,
        )
        .unwrap();
        let b = build(
            &tweet(),
            &account("y"),
            &metadata_version,
            &question_version,
        )
        .unwrap();
        assert_eq!(a.text, "hello world");
        assert_eq!(a, b);
        assert_eq!(a.question, QUESTION_V1);
    }

    #[test]
    fn unknown_metadata_version_returns_an_error() {
        let (_, question_version) = v1();
        let result = build(
            &tweet(),
            &account("x"),
            &MetadataVersion("v9".into()),
            &question_version,
        );
        assert_eq!(result, Err(Error::UnknownMetadataVersion("v9".into())));
    }

    #[test]
    fn unknown_question_version_returns_an_error() {
        let (metadata_version, _) = v1();
        let result = build(
            &tweet(),
            &account("x"),
            &metadata_version,
            &QuestionVersion("v9".into()),
        );
        assert_eq!(result, Err(Error::UnknownQuestionVersion("v9".into())));
    }
}
