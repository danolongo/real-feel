#![cfg(feature = "integration")]

use std::{sync::Arc, time::Duration};

use api::{AppState, router};
use aws_sdk_sqs::Client;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
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

async fn app(pool: PgPool, queue_url: &str) -> Router {
    router(AppState {
        db: pool,
        queue: Queue::connect(queue_url).await,
        versions: Arc::new(versions("m1")),
        bot_threshold: 0.5,
        query_reuse: Duration::from_secs(3600),
    })
}

async fn call(app: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

fn post(text: &str) -> Request<Body> {
    Request::post("/queries")
        .header("content-type", "application/json")
        .body(Body::from(json!({ "text": text }).to_string()))
        .unwrap()
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

fn traced() -> impl tracing::Subscriber {
    let provider = SdkTracerProvider::builder().build();
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")))
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

    let (status, body) = {
        let _guard = tracing::subscriber::set_default(traced());
        call(&app, post("  what do people say?  ")).await
    };
    assert_eq!(status, StatusCode::ACCEPTED);
    let query_id: Uuid = body["query_id"].as_str().unwrap().parse().unwrap();

    let (status, body) = call(&app, get(&format!("/queries/{query_id}"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "queued");
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
