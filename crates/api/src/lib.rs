mod error;

use std::{error::Error, sync::Arc, time::Duration};

use axum::{
    Json, Router,
    extract::{
        Path, Query, State,
        rejection::{JsonRejection, PathRejection, QueryRejection},
    },
    http::StatusCode,
    routing::get,
};
use chrono::{DateTime, Utc};
use db::{CreatedQuery, PgPool, ResultTweet, Versions};
use queue::Queue;
use serde::{Deserialize, Serialize};
use types::{QueryStatus, ScraperJob};
use uuid::Uuid;

pub use error::ApiError;

const MAX_TEXT_CHARS: usize = 512;
const DEFAULT_LIMIT: i64 = 50;
const MAX_LIMIT: i64 = 200;

#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
    pub queue: Queue,
    pub versions: Arc<Versions>,
    pub bot_threshold: f64,
    pub query_reuse: Duration,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/queries", axum::routing::post(create_query))
        .route("/queries/{id}", get(get_query))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .with_state(state)
}

#[derive(Deserialize)]
struct NewQuery {
    text: String,
}

#[derive(Serialize)]
struct Created {
    query_id: Uuid,
}

#[tracing::instrument(skip_all, fields(query_id = tracing::field::Empty, trace_id = tracing::field::Empty))]
async fn create_query(
    State(state): State<AppState>,
    body: Result<Json<NewQuery>, JsonRejection>,
) -> Result<(StatusCode, Json<Created>), ApiError> {
    let span = tracing::Span::current();
    if let Some(trace_id) = telemetry::trace_id(&span) {
        span.record("trace_id", trace_id);
    }
    let Json(body) = body.map_err(|_| ApiError::InvalidQuery)?;
    let text = body.text.trim();
    if !(1..=MAX_TEXT_CHARS).contains(&text.chars().count()) {
        return Err(ApiError::InvalidQuery);
    }

    let query = match db::create_or_reuse_query(&state.db, text, state.query_reuse).await? {
        CreatedQuery::Reused(query) => {
            span.record("query_id", query.id.to_string());
            tracing::info!(status = ?query.status, "query reused");
            return Ok((StatusCode::ACCEPTED, Json(Created { query_id: query.id })));
        }
        CreatedQuery::New(query) => query,
    };
    span.record("query_id", query.id.to_string());
    tracing::info!(text = %query.text, "query created");
    let job = ScraperJob {
        query_id: query.id,
        trace_context: None,
    };
    if let Err(error) = state.queue.send(job).await {
        tracing::error!(%error, query_id = %query.id, "scraper job send failed");
        db::set_status(
            &state.db,
            query.id,
            QueryStatus::Failed,
            Some(&error.to_string()),
        )
        .await?;
        return Err(ApiError::QueueUnavailable);
    }
    tracing::info!("scraper job sent");
    Ok((StatusCode::ACCEPTED, Json(Created { query_id: query.id })))
}

#[derive(Deserialize)]
struct PageParams {
    limit: Option<i64>,
    after: Option<String>,
}

#[derive(Serialize)]
struct QueryResponse {
    query_id: Uuid,
    status: QueryStatus,
    error: Option<String>,
    result: Option<ResultPage>,
}

#[derive(Serialize)]
struct ResultPage {
    tweets: Vec<TweetView>,
    next_cursor: Option<String>,
}

#[derive(Serialize)]
struct TweetView {
    tweet_id: String,
    text: String,
    posted_at: Option<DateTime<Utc>>,
    account: AccountView,
    p_bot: Option<f64>,
    is_bot: Option<bool>,
}

#[derive(Serialize)]
struct AccountView {
    id: String,
    handle: String,
}

impl TweetView {
    fn new(row: ResultTweet, bot_threshold: f64) -> Self {
        let p_bot = row.prediction.map(|prediction| prediction.p_bot);
        Self {
            tweet_id: row.tweet.id,
            text: row.tweet.text,
            posted_at: row.tweet.posted_at,
            account: AccountView {
                id: row.account.id,
                handle: row.account.handle,
            },
            p_bot,
            is_bot: p_bot.map(|p_bot| p_bot >= bot_threshold),
        }
    }
}

async fn get_query(
    State(state): State<AppState>,
    id: Result<Path<Uuid>, PathRejection>,
    params: Result<Query<PageParams>, QueryRejection>,
) -> Result<Json<QueryResponse>, ApiError> {
    let Path(id) = id.map_err(|_| ApiError::InvalidQueryId)?;
    let Query(params) = params.map_err(|_| ApiError::InvalidCursor)?;
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT);
    if !(1..=MAX_LIMIT).contains(&limit) || params.after.as_deref() == Some("") {
        return Err(ApiError::InvalidCursor);
    }

    let query = db::fetch_query(&state.db, id)
        .await?
        .ok_or(ApiError::QueryNotFound)?;
    let result = if query.status == QueryStatus::Done {
        let mut rows = db::result_page(
            &state.db,
            id,
            &state.versions,
            params.after.as_deref(),
            limit + 1,
        )
        .await?;
        let has_more = rows.len() > limit as usize;
        rows.truncate(limit as usize);
        let next_cursor = rows
            .last()
            .filter(|_| has_more)
            .map(|row| row.tweet.id.clone());
        Some(ResultPage {
            tweets: rows
                .into_iter()
                .map(|row| TweetView::new(row, state.bot_threshold))
                .collect(),
            next_cursor,
        })
    } else {
        None
    };
    Ok(Json(QueryResponse {
        query_id: query.id,
        status: query.status,
        error: query.error,
        result,
    }))
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn ready(State(state): State<AppState>) -> StatusCode {
    match check_dependencies(&state).await {
        Ok(()) => StatusCode::OK,
        Err(error) => {
            tracing::warn!(%error, "api is not ready");
            StatusCode::SERVICE_UNAVAILABLE
        }
    }
}

async fn check_dependencies(state: &AppState) -> Result<(), Box<dyn Error>> {
    db::ping(&state.db).await?;
    state.queue.ping().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use serde_json::{Value, json};
    use sqlx::postgres::PgPoolOptions;
    use tower::ServiceExt;
    use types::{MetadataVersion, ModelVersion, QuestionVersion};

    async fn unreachable_services() -> Router {
        let state = AppState {
            db: PgPoolOptions::new()
                .acquire_timeout(Duration::from_secs(1))
                .connect_lazy("postgres://realfeel:none@127.0.0.1:1/realfeel")
                .unwrap(),
            queue: Queue::connect("http://127.0.0.1:1/000000000000/none").await,
            versions: Arc::new(Versions {
                model: ModelVersion("m1".into()),
                metadata: MetadataVersion("v1".into()),
                question: QuestionVersion("v1".into()),
            }),
            bot_threshold: 0.5,
            query_reuse: Duration::ZERO,
        };
        router(state)
    }

    fn result_tweet(p_bot: Option<f64>) -> ResultTweet {
        ResultTweet {
            tweet: types::Tweet {
                id: "t1".into(),
                account_id: "a1".into(),
                text: "hi".into(),
                posted_at: None,
                raw: json!({}),
            },
            account: types::Account {
                id: "a1".into(),
                handle: "alice".into(),
                joined_at: None,
                followers: None,
                following: None,
                tweet_count: None,
                is_verified: None,
                is_automated: None,
                raw: json!({}),
            },
            prediction: p_bot.map(|p_bot| types::Prediction {
                tweet_id: "t1".into(),
                model_version: ModelVersion("m1".into()),
                metadata_version: MetadataVersion("v1".into()),
                question_version: QuestionVersion("v1".into()),
                p_bot,
                p_llm_text: None,
                raw: json!({}),
            }),
        }
    }

    #[test]
    fn is_bot_compares_p_bot_with_the_threshold() {
        let is_bot = |p_bot| TweetView::new(result_tweet(p_bot), 0.5).is_bot;
        assert_eq!(is_bot(Some(0.5)), Some(true));
        assert_eq!(is_bot(Some(0.49)), Some(false));
        assert_eq!(is_bot(None), None);
    }

    async fn call(request: Request<Body>) -> (StatusCode, Value) {
        let response = unreachable_services().await.oneshot(request).await.unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
    }

    fn post(body: Value) -> Request<Body> {
        Request::post("/queries")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn get_uri(uri: &str) -> Request<Body> {
        Request::get(uri).body(Body::empty()).unwrap()
    }

    #[tokio::test]
    async fn health_returns_200_with_postgres_unreachable() {
        let (status, _) = call(get_uri("/health")).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn ready_returns_503_with_postgres_unreachable() {
        let (status, _) = call(get_uri("/ready")).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn unknown_path_returns_404() {
        let (status, _) = call(get_uri("/nope")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn invalid_query_text_returns_400() {
        let too_long = "x".repeat(MAX_TEXT_CHARS + 1);
        for body in [
            json!({ "text": "" }),
            json!({ "text": "  \n " }),
            json!({ "text": too_long }),
            json!({ "wrong": "field" }),
        ] {
            let (status, body) = call(post(body)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(body["code"], "invalid_query");
        }
    }

    #[tokio::test]
    async fn invalid_query_id_returns_400() {
        let (status, body) = call(get_uri("/queries/not-a-uuid")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["code"], "invalid_query_id");
    }

    #[tokio::test]
    async fn invalid_limit_or_after_returns_400() {
        let id = Uuid::new_v4();
        for query in ["limit=0", "limit=201", "limit=abc", "after="] {
            let (status, body) = call(get_uri(&format!("/queries/{id}?{query}"))).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{query}");
            assert_eq!(body["code"], "invalid_cursor", "{query}");
        }
    }

    #[tokio::test]
    async fn postgres_failure_returns_500_without_detail() {
        let (status, body) = call(get_uri(&format!("/queries/{}", Uuid::new_v4()))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["code"], "internal");
        assert_eq!(body["message"], "internal error");
    }
}
