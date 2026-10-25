use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "query_status", rename_all = "snake_case")]
pub enum QueryStatus {
    Queued,
    Scraping,
    Classifying,
    Done,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub id: Uuid,
    pub text: String,
    pub status: QueryStatus,
    pub error: Option<String>,
    pub batch_count: Option<i32>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Account {
    pub id: String,
    pub handle: String,
    pub raw: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tweet {
    pub id: String,
    pub account_id: String,
    pub text: String,
    pub posted_at: Option<DateTime<Utc>>,
    pub raw: Value,
}

/// `trace_context` holds a W3C `traceparent` value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScraperJob {
    pub query_id: Uuid,
    pub trace_context: Option<String>,
}

/// Carries tweet IDs only, so the job stays far below the SQS message size limit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassifierJob {
    pub query_id: Uuid,
    pub batch_number: i32,
    pub tweet_ids: Vec<String>,
    pub trace_context: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type)]
#[serde(transparent)]
#[sqlx(transparent)]
pub struct ModelVersion(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type)]
#[serde(transparent)]
#[sqlx(transparent)]
pub struct MetadataVersion(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type)]
#[serde(transparent)]
#[sqlx(transparent)]
pub struct QuestionVersion(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub tweet_id: String,
    pub model_version: ModelVersion,
    pub metadata_version: MetadataVersion,
    pub question_version: QuestionVersion,
    pub p_bot: f64,
    pub raw: Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_status_serializes_in_snake_case() {
        assert_eq!(
            serde_json::to_string(&QueryStatus::Classifying).unwrap(),
            "\"classifying\""
        );
    }

    #[test]
    fn versions_serialize_as_plain_strings() {
        assert_eq!(
            serde_json::to_string(&MetadataVersion("v1".into())).unwrap(),
            "\"v1\""
        );
    }

    #[test]
    fn classifier_job_round_trips() {
        let job = ClassifierJob {
            query_id: Uuid::new_v4(),
            batch_number: 3,
            tweet_ids: vec!["1".into(), "2".into()],
            trace_context: None,
        };
        let json = serde_json::to_string(&job).unwrap();
        assert_eq!(serde_json::from_str::<ClassifierJob>(&json).unwrap(), job);
    }
}
