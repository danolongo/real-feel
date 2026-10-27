#![cfg(feature = "integration")]

use std::{sync::Arc, time::Duration};

use api::{AppState, router};
use aws_sdk_sqs::Client;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{HeaderMap, Request, StatusCode},
};
use db::{PgPool, Versions};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::SdkTracerProvider;
use queue::Queue;
use serde_json::{Value, json};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;
use types::{
    Account, MetadataVersion, ModelVersion, Prediction, QueryStatus, QuestionVersion, ScraperJob,
    Tweet,
};
use uuid::Uuid;

struct TestQueue {
    client: Client,
    url: String,
}

impl TestQueue {
    async fn create() -> Self {
        let client =
            Client::new(&aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await);
        let url = client
            .create_queue()
            .queue_name(format!("test-{}", Uuid::new_v4()))
            .send()
            .await
            .unwrap()
            .queue_url
            .unwrap();
        Self { client, url }
    }

    async fn delete(&self) {
        self.client
            .delete_queue()
            .queue_url(&self.url)
            .send()
            .await
            .unwrap();
    }
}

fn versions(model: &str) -> Versions {
    Versions {
        model: ModelVersion(model.into()),
        metadata: MetadataVersion("v1".into()),
        question: QuestionVersion("v1".into()),
    }
}

const TEST_KEY: &str = "test-key";

async fn insert_key(pool: &PgPool, key: &str, quota: i32) -> Uuid {
    db::insert_api_key(pool, "test", &api::hash_key(key), quota)
        .await
        .unwrap()
        .id
}

async fn app_with(pool: PgPool, queue_url: &str, quota_enabled: bool) -> Router {
    install_tracing();
    router(AppState {
        db: pool,
        queue: Queue::connect(queue_url).await,
        versions: Arc::new(versions("m1")),
        bot_threshold: 0.5,
        query_reuse: Duration::from_secs(3600),
        quota_enabled,
    })
}

/// The router, and `TEST_KEY` with a quota that no test reaches.
async fn app(pool: PgPool, queue_url: &str) -> Router {
    insert_key(&pool, TEST_KEY, 1000).await;
    app_with(pool, queue_url, true).await
}

async fn call(app: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let (status, _, body) = call_with_headers(app, request).await;
    (status, body)
}

async fn call_with_headers(app: &Router, request: Request<Body>) -> (StatusCode, HeaderMap, Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (
        status,
        headers,
        serde_json::from_slice(&body).unwrap_or(Value::Null),
    )
}

fn post_as(key: &str, text: &str) -> Request<Body> {
    post_json_as(key, json!({ "text": text }))
}

fn post_json_as(key: &str, body: Value) -> Request<Body> {
    Request::post("/queries")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {key}"))
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn post(text: &str) -> Request<Body> {
    post_as(TEST_KEY, text)
}

fn get_as(key: &str, uri: &str) -> Request<Body> {
    Request::get(uri)
        .header("authorization", format!("Bearer {key}"))
        .body(Body::empty())
        .unwrap()
}

fn get(uri: &str) -> Request<Body> {
    get_as(TEST_KEY, uri)
}

async fn query_count(pool: &PgPool, api_key_id: Uuid) -> i64 {
    sqlx::query_scalar!(
        r#"SELECT COUNT(*) AS "count!" FROM queries WHERE api_key_id = $1"#,
        api_key_id
    )
    .fetch_one(pool)
    .await
    .unwrap()
}

/// Installs the tracing subscriber for the whole test binary. A thread-local subscriber races with
/// parallel tests: a test without one can cache a span as uninteresting for every thread.
fn install_tracing() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let provider = SdkTracerProvider::builder().build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
        tracing::subscriber::set_global_default(subscriber).unwrap();
    });
}

async fn seed_done_query(pool: &PgPool, tweets: usize) -> Uuid {
    let query = db::insert_query(pool, "q").await.unwrap();
    db::upsert_account(
        pool,
        &Account {
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
    )
    .await
    .unwrap();
    for n in 0..tweets {
        let id = format!("{n:04}");
        db::upsert_tweet(
            pool,
            &Tweet {
                id: id.clone(),
                account_id: "a1".into(),
                text: format!("tweet {id}"),
                posted_at: None,
                raw: json!({}),
            },
        )
        .await
        .unwrap();
        db::upsert_query_tweet(pool, query.id, &id).await.unwrap();
    }
    for status in [QueryStatus::Scraping, QueryStatus::Done] {
        assert!(db::set_status(pool, query.id, status, None).await.unwrap());
    }
    query.id
}

async fn seed_prediction(pool: &PgPool, tweet_id: &str, model: &str, p_bot: f64) {
    let v = versions(model);
    db::upsert_prediction(
        pool,
        &Prediction {
            tweet_id: tweet_id.into(),
            model_version: v.model,
            metadata_version: v.metadata,
            question_version: v.question,
            p_bot,
            p_llm_text: None,
            raw: json!({}),
        },
    )
    .await
    .unwrap();
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn post_then_get_shows_queued_and_sends_a_traced_scraper_job(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    let app = app(pool.clone(), &test_queue.url).await;

    let (status, body) = call(&app, post("  what do people say?  ")).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let query_id: Uuid = body["query_id"].as_str().unwrap().parse().unwrap();

    let (status, body) = call(&app, get(&format!("/queries/{query_id}"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "queued");
    assert_eq!(
        body["progress"],
        json!({"batches_done": 0, "batch_count": null})
    );
    assert_eq!(body["result"], Value::Null);
    let query = db::fetch_query(&pool, query_id).await.unwrap().unwrap();
    assert_eq!(query.text, "what do people say?");

    let received = Queue::connect(&test_queue.url)
        .await
        .receive::<ScraperJob>(5)
        .await
        .unwrap();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].job.query_id, query_id);
    assert!(received[0].job.trace_context.is_some());

    test_queue.delete().await;
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn missing_queue_returns_503_and_fails_the_query(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    test_queue.delete().await;
    let app = app(pool.clone(), &test_queue.url).await;

    let (status, body) = call(&app, post("hello")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["code"], "queue_unavailable");

    let id = sqlx::query_scalar::<_, Uuid>("SELECT id FROM queries")
        .fetch_one(&pool)
        .await
        .unwrap();
    let query = db::fetch_query(&pool, id).await.unwrap().unwrap();
    assert_eq!(query.status, QueryStatus::Failed);
    assert!(query.error.is_some());

    let (_, body) = call(&app, get(&format!("/queries/{id}"))).await;
    assert_eq!(body["status"], "failed");
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn unknown_query_returns_404(pool: PgPool) {
    let app = app(pool, "http://localhost:4566/000000000000/unused").await;
    let (status, body) = call(&app, get(&format!("/queries/{}", Uuid::new_v4()))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["code"], "query_not_found");
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn done_query_pages_by_cursor(pool: PgPool) {
    let id = seed_done_query(&pool, 120).await;
    for n in 0..120 {
        seed_prediction(&pool, &format!("{n:04}"), "m1", 0.25).await;
    }
    let app = app(pool, "http://localhost:4566/000000000000/unused").await;

    let mut sizes = Vec::new();
    let mut after = String::new();
    loop {
        let (status, body) = call(&app, get(&format!("/queries/{id}?limit=50{after}"))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "done");
        let tweets = body["result"]["tweets"].as_array().unwrap();
        sizes.push(tweets.len());
        assert_eq!(tweets[0]["p_bot"], 0.25);
        assert_eq!(tweets[0]["account"]["handle"], "alice");
        assert_eq!(tweets[0]["is_bot"], false);
        match body["result"]["next_cursor"].as_str() {
            Some(cursor) => after = format!("&after={cursor}"),
            None => break,
        }
    }
    assert_eq!(sizes, [50, 50, 20]);
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn result_shows_the_prediction_of_the_default_versions(pool: PgPool) {
    let id = seed_done_query(&pool, 1).await;
    seed_prediction(&pool, "0000", "m1", 0.1).await;
    seed_prediction(&pool, "0000", "m2", 0.9).await;
    let app = app(pool, "http://localhost:4566/000000000000/unused").await;

    let (_, body) = call(&app, get(&format!("/queries/{id}"))).await;
    let tweets = body["result"]["tweets"].as_array().unwrap();
    assert_eq!(tweets.len(), 1);
    assert_eq!(tweets[0]["p_bot"], 0.1);
    assert_eq!(tweets[0]["is_bot"], false);
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn is_bot_is_true_at_the_threshold_and_null_without_a_prediction(pool: PgPool) {
    let id = seed_done_query(&pool, 3).await;
    seed_prediction(&pool, "0000", "m1", 0.5).await;
    seed_prediction(&pool, "0001", "m1", 0.49).await;
    let app = app(pool, "http://localhost:4566/000000000000/unused").await;

    let (_, body) = call(&app, get(&format!("/queries/{id}"))).await;
    let tweets = body["result"]["tweets"].as_array().unwrap();
    assert_eq!(tweets[0]["is_bot"], true);
    assert_eq!(tweets[1]["is_bot"], false);
    assert_eq!(tweets[2]["p_bot"], Value::Null);
    assert_eq!(tweets[2]["is_bot"], Value::Null);
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn progress_counts_the_finished_batches(pool: PgPool) {
    let query = db::insert_query(&pool, "q").await.unwrap();
    assert!(
        db::set_status(&pool, query.id, QueryStatus::Scraping, None)
            .await
            .unwrap()
    );
    assert!(db::set_batch_count(&pool, query.id, 3).await.unwrap());
    for batch in [0, 1] {
        db::upsert_query_batch(&pool, query.id, batch)
            .await
            .unwrap();
    }
    let app = app(pool, "http://localhost:4566/000000000000/unused").await;

    let (_, body) = call(&app, get(&format!("/queries/{}", query.id))).await;
    assert_eq!(body["status"], "classifying");
    assert_eq!(
        body["progress"],
        json!({"batches_done": 2, "batch_count": 3})
    );
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn done_query_shows_the_counts_the_account_and_p_llm_text(pool: PgPool) {
    let id = seed_done_query(&pool, 1).await;
    sqlx::query(
        r#"UPDATE tweets SET raw = '{"likeCount": 3, "retweetCount": 0, "replyCount": 1, "viewCount": 120}'"#,
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        r#"UPDATE accounts SET raw = '{"name": "Alice"}', followers = 210, tweet_count = 5400"#,
    )
    .execute(&pool)
    .await
    .unwrap();
    seed_prediction(&pool, "0000", "m1", 0.2).await;
    sqlx::query("UPDATE predictions SET p_llm_text = 0.75")
        .execute(&pool)
        .await
        .unwrap();
    let app = app(pool, "http://localhost:4566/000000000000/unused").await;

    let (_, body) = call(&app, get(&format!("/queries/{id}"))).await;
    let tweet = &body["result"]["tweets"][0];
    assert_eq!(tweet["p_llm_text"], 0.75);
    assert_eq!(
        (
            &tweet["like_count"],
            &tweet["retweet_count"],
            &tweet["reply_count"],
            &tweet["view_count"]
        ),
        (&json!(3), &json!(0), &json!(1), &json!(120))
    );
    assert_eq!(tweet["account"]["name"], "Alice");
    assert_eq!(tweet["account"]["followers"], 210);
    assert_eq!(tweet["account"]["tweet_count"], 5400);
    assert_eq!(tweet["account"]["following"], Value::Null);
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn ready_returns_200_when_postgres_and_sqs_respond(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    let app = app(pool, &test_queue.url).await;
    let (status, _) = call(&app, get("/ready")).await;
    assert_eq!(status, StatusCode::OK);
    test_queue.delete().await;
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn ready_returns_503_when_the_queue_is_missing(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    test_queue.delete().await;
    let app = app(pool, &test_queue.url).await;
    let (status, _) = call(&app, get("/ready")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let (status, _) = call(&app, get("/health")).await;
    assert_eq!(status, StatusCode::OK);
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn two_identical_queries_at_once_share_one_query_and_one_scraper_job(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    let app = app(pool.clone(), &test_queue.url).await;

    let ((first_status, first), (second_status, second)) = tokio::join!(
        call(&app, post("Same question")),
        call(&app, post("  same QUESTION "))
    );
    assert_eq!(first_status, StatusCode::ACCEPTED);
    assert_eq!(second_status, StatusCode::ACCEPTED);
    assert_eq!(first["query_id"], second["query_id"]);

    let queue = Queue::connect(&test_queue.url).await;
    let mut jobs = 0;
    for _ in 0..3 {
        jobs += queue.receive::<ScraperJob>(1).await.unwrap().len();
    }
    assert_eq!(jobs, 1);

    test_queue.delete().await;
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn invalid_query_text_returns_400(pool: PgPool) {
    let app = app(pool, "http://localhost:4566/000000000000/unused").await;
    let too_long = "x".repeat(513);
    for body in [
        json!({ "text": "" }),
        json!({ "text": "  \n " }),
        json!({ "text": too_long }),
        json!({ "wrong": "field" }),
    ] {
        let (status, body) = call(&app, post_json_as(TEST_KEY, body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["code"], "invalid_query");
    }
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn invalid_query_id_returns_400(pool: PgPool) {
    let app = app(pool, "http://localhost:4566/000000000000/unused").await;
    let (status, body) = call(&app, get("/queries/not-a-uuid")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["code"], "invalid_query_id");
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn invalid_limit_or_after_returns_400(pool: PgPool) {
    let app = app(pool, "http://localhost:4566/000000000000/unused").await;
    let id = Uuid::new_v4();
    for query in ["limit=0", "limit=201", "limit=abc", "after="] {
        let (status, body) = call(&app, get(&format!("/queries/{id}?{query}"))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{query}");
        assert_eq!(body["code"], "invalid_cursor", "{query}");
    }
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn a_missing_unknown_or_revoked_key_returns_401(pool: PgPool) {
    let revoked = insert_key(&pool, "revoked-key", 10).await;
    assert!(db::revoke_api_key(&pool, revoked).await.unwrap());
    insert_key(&pool, TEST_KEY, 10).await;
    let app = app_with(
        pool.clone(),
        "http://localhost:4566/000000000000/unused",
        true,
    )
    .await;
    let id = Uuid::new_v4();

    let no_key = |request: Request<Body>| {
        let (mut parts, body) = request.into_parts();
        parts.headers.remove("authorization");
        Request::from_parts(parts, body)
    };
    for request in [
        no_key(post("hi")),
        no_key(get(&format!("/queries/{id}"))),
        post_as("unknown-key", "hi"),
        get_as("unknown-key", &format!("/queries/{id}")),
        post_as("revoked-key", "hi"),
        get_as("revoked-key", &format!("/queries/{id}")),
    ] {
        let (status, body) = call(&app, request).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["code"], "unauthorized");
    }
    let rows = sqlx::query_scalar!(r#"SELECT COUNT(*) AS "count!" FROM queries"#)
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(rows, 0);
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn health_and_ready_need_no_key(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    let app = app_with(pool, &test_queue.url, true).await;
    for uri in ["/health", "/ready"] {
        let (status, _) = call(&app, Request::get(uri).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
    }
    test_queue.delete().await;
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn a_new_query_stores_the_id_of_its_key(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    let key_id = insert_key(&pool, TEST_KEY, 10).await;
    let app = app_with(pool.clone(), &test_queue.url, true).await;
    let (status, _) = call(&app, post("hi")).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(query_count(&pool, key_id).await, 1);
    test_queue.delete().await;
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn a_key_over_its_quota_gets_429_with_retry_after(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    let key_id = insert_key(&pool, TEST_KEY, 2).await;
    let app = app_with(pool.clone(), &test_queue.url, true).await;

    for text in ["one", "two"] {
        let (status, _) = call(&app, post(text)).await;
        assert_eq!(status, StatusCode::ACCEPTED);
    }
    let (status, headers, body) = call_with_headers(&app, post("three")).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(body["code"], "quota_exceeded");
    let retry_after: u64 = headers["retry-after"].to_str().unwrap().parse().unwrap();
    assert!((3590..=3600).contains(&retry_after), "{retry_after}");
    assert_eq!(query_count(&pool, key_id).await, 2);

    let other = insert_key(&pool, "other-key", 1).await;
    let (status, _) = call(&app, post_as("other-key", "three")).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    assert_eq!(query_count(&pool, other).await, 1);
    test_queue.delete().await;
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn a_reused_query_does_not_count_against_the_quota(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    let key_id = insert_key(&pool, TEST_KEY, 1).await;
    let app = app_with(pool.clone(), &test_queue.url, true).await;

    let (status, first) = call(&app, post("same text")).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    for _ in 0..3 {
        let (status, again) = call(&app, post("  SAME text ")).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(again["query_id"], first["query_id"]);
    }
    assert_eq!(query_count(&pool, key_id).await, 1);
    test_queue.delete().await;
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn parallel_requests_of_one_key_never_pass_the_quota(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    let key_id = insert_key(&pool, TEST_KEY, 5).await;
    let app = app_with(pool.clone(), &test_queue.url, true).await;

    let tasks: Vec<_> = (0..30)
        .map(|n| {
            let app = app.clone();
            tokio::spawn(async move { call(&app, post(&format!("question {n}"))).await.0 })
        })
        .collect();
    let mut accepted = 0;
    let mut limited = 0;
    for task in tasks {
        match task.await.unwrap() {
            StatusCode::ACCEPTED => accepted += 1,
            StatusCode::TOO_MANY_REQUESTS => limited += 1,
            other => panic!("unexpected status {other}"),
        }
    }
    assert_eq!((accepted, limited), (5, 25));
    assert_eq!(query_count(&pool, key_id).await, 5);
    test_queue.delete().await;
}

#[sqlx::test(migrator = "db::MIGRATOR")]
async fn a_key_over_its_quota_still_creates_queries_with_the_quota_off(pool: PgPool) {
    let test_queue = TestQueue::create().await;
    let key_id = insert_key(&pool, TEST_KEY, 1).await;
    let app = app_with(pool.clone(), &test_queue.url, false).await;

    for n in 0..4 {
        let (status, _) = call(&app, post(&format!("question {n}"))).await;
        assert_eq!(status, StatusCode::ACCEPTED);
    }
    assert_eq!(query_count(&pool, key_id).await, 4);
    let (status, _) = call(&app, post_as("unknown-key", "hi")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    test_queue.delete().await;
}
