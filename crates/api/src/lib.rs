mod auth;
mod error;

use std::{error::Error, sync::Arc, time::Duration};

use axum::{
    Extension, Json, Router,
    extract::{
        Path, Query, State,
        rejection::{JsonRejection, PathRejection, QueryRejection},
    },
    http::StatusCode,
    middleware,
    routing::get,
};
use chrono::{DateTime, Utc};
use db::{ApiKey, CreatedQuery, PgPool, ResultTweet, Versions};
use queue::Queue;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use types::{QueryStatus, ScraperJob};
use uuid::Uuid;

pub use auth::{generate_key, hash_key};
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
    pub quota_enabled: bool,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/queries", axum::routing::post(create_query))
        .route("/queries/{id}", get(get_query))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            auth::require_api_key,
        ))
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
    Extension(api_key): Extension<ApiKey>,
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

    let created = db::create_or_reuse_query(
        &state.db,
        text,
        state.query_reuse,
        &api_key,
        state.quota_enabled,
    )
    .await?;
    let query = match created {
        CreatedQuery::Reused(query) => {
            span.record("query_id", query.id.to_string());
            tracing::info!(status = ?query.status, "query reused");
            return Ok((StatusCode::ACCEPTED, Json(Created { query_id: query.id })));
        }
        CreatedQuery::New(query) => query,
        CreatedQuery::OverQuota { retry_after } => {
            tracing::info!(api_key_id = %api_key.id, "query quota used");
            return Err(ApiError::QuotaExceeded { retry_after });
        }
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
    progress: Progress,
    result: Option<ResultPage>,
}

#[derive(Serialize)]
struct Progress {
    batches_done: i64,
    batch_count: Option<i32>,
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
    like_count: Option<i64>,
    retweet_count: Option<i64>,
    reply_count: Option<i64>,
    view_count: Option<i64>,
    account: AccountView,
    p_bot: Option<f64>,
    p_llm_text: Option<f64>,
    is_bot: Option<bool>,
}

#[derive(Serialize)]
struct AccountView {
    id: String,
    handle: String,
    name: Option<String>,
    joined_at: Option<DateTime<Utc>>,
    followers: Option<i32>,
    following: Option<i32>,
    tweet_count: Option<i32>,
    is_verified: Option<bool>,
    is_automated: Option<bool>,
}

impl TweetView {
    fn new(row: ResultTweet, bot_threshold: f64) -> Self {
        let p_bot = row.prediction.as_ref().map(|prediction| prediction.p_bot);
        let count = |key: &str| row.tweet.raw.get(key).and_then(Value::as_i64);
        Self {
            like_count: count("likeCount"),
            retweet_count: count("retweetCount"),
            reply_count: count("replyCount"),
            view_count: count("viewCount"),
            tweet_id: row.tweet.id,
            text: row.tweet.text,
            posted_at: row.tweet.posted_at,
            account: AccountView {
                name: row
                    .account
                    .raw
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                    .map(str::to_string),
                id: row.account.id,
                handle: row.account.handle,
                joined_at: row.account.joined_at,
                followers: row.account.followers,
                following: row.account.following,
                tweet_count: row.account.tweet_count,
                is_verified: row.account.is_verified,
                is_automated: row.account.is_automated,
            },
            p_bot,
            p_llm_text: row.prediction.and_then(|prediction| prediction.p_llm_text),
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
        progress: Progress {
            batches_done: db::batches_done(&state.db, id).await?,
            batch_count: query.batch_count,
        },
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
    use serde_json::json;
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
            quota_enabled: true,
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
    fn tweet_view_reads_counts_and_name_from_raw() {
        let mut row = result_tweet(Some(0.2));
        row.tweet.raw =
            json!({"likeCount": 3, "retweetCount": 0, "replyCount": 1, "viewCount": 120});
        row.account.raw = json!({"name": "Alice"});
        let view = TweetView::new(row, 0.5);
        assert_eq!(
            (
                view.like_count,
                view.retweet_count,
                view.reply_count,
                view.view_count
            ),
            (Some(3), Some(0), Some(1), Some(120))
        );
        assert_eq!(view.account.name.as_deref(), Some("Alice"));
    }

    #[test]
    fn tweet_view_leaves_missing_raw_fields_null() {
        let mut row = result_tweet(None);
        row.account.raw = json!({"name": ""});
        let view = TweetView::new(row, 0.5);
        assert_eq!(view.like_count, None);
        assert_eq!(view.account.name, None);
        assert_eq!(view.p_llm_text, None);
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
    async fn a_query_route_without_a_key_returns_401() {
        for request in [
            post(json!({ "text": "hi" })),
            get_uri(&format!("/queries/{}", Uuid::new_v4())),
            Request::get("/queries/x")
                .header("authorization", "Basic abc")
                .body(Body::empty())
                .unwrap(),
        ] {
            let (status, body) = call(request).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED);
            assert_eq!(body["code"], "unauthorized");
        }
    }

    #[tokio::test]
    async fn postgres_failure_returns_500_without_detail() {
        let request = Request::get(format!("/queries/{}", Uuid::new_v4()))
            .header("authorization", "Bearer any-key")
            .body(Body::empty())
            .unwrap();
        let (status, body) = call(request).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["code"], "internal");
        assert_eq!(body["message"], "internal error");
    }
}
