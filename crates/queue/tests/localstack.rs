#![cfg(feature = "integration")]

use std::time::Duration;

use aws_sdk_sqs::Client;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::trace::SdkTracerProvider;
use queue::Queue;
use tracing_subscriber::layer::SubscriberExt;
use types::{ClassifierJob, ScraperJob};
use uuid::Uuid;

const VISIBILITY_TIMEOUT_SECONDS: &str = "2";

struct TestQueue {
    client: Client,
    url: String,
    queue: Queue,
}

impl TestQueue {
    async fn create() -> Self {
        let client =
            Client::new(&aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await);
        let url = client
            .create_queue()
            .queue_name(format!("test-{}", Uuid::new_v4()))
            .attributes(
                aws_sdk_sqs::types::QueueAttributeName::VisibilityTimeout,
                VISIBILITY_TIMEOUT_SECONDS,
            )
            .send()
            .await
            .unwrap()
            .queue_url
            .unwrap();
        let queue = Queue::connect(&url).await;
        Self { client, url, queue }
    }

    async fn delete(self) {
        self.client
            .delete_queue()
            .queue_url(self.url)
            .send()
            .await
            .unwrap();
    }
}

fn traced() -> impl tracing::Subscriber {
    let provider = SdkTracerProvider::builder().build();
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")))
}

fn scraper_job() -> ScraperJob {
    ScraperJob {
        query_id: Uuid::new_v4(),
        trace_context: None,
    }
}

fn classifier_job() -> ClassifierJob {
    ClassifierJob {
        query_id: Uuid::new_v4(),
        batch_number: 2,
        tweet_ids: vec!["1".into(), "2".into()],
        trace_context: None,
    }
}

#[tokio::test]
async fn scraper_job_round_trips_with_trace_context() {
    let test = TestQueue::create().await;
    let job = scraper_job();

    let sent_traceparent = {
        let _guard = tracing::subscriber::set_default(traced());
        let span = tracing::info_span!("send");
        let _entered = span.enter();
        let sent = telemetry::current_traceparent().unwrap();
        test.queue.send(job.clone()).await.unwrap();
        sent
    };

    let received = test.queue.receive::<ScraperJob>(5).await.unwrap();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].receive_count, 1);
    assert_eq!(
        received[0].job,
        ScraperJob {
            trace_context: Some(sent_traceparent.clone()),
            ..job
        }
    );

    let _guard = tracing::subscriber::set_default(traced());
    let span = tracing::info_span!("receive");
    received[0].continue_trace(&span);
    let continued = span.in_scope(|| telemetry::current_traceparent().unwrap());
    assert_eq!(
        continued.split('-').nth(1),
        sent_traceparent.split('-').nth(1)
    );

    test.delete().await;
}

#[tokio::test]
async fn classifier_job_round_trips() {
    let test = TestQueue::create().await;
    let job = classifier_job();

    test.queue.send(job.clone()).await.unwrap();

    let received = test.queue.receive::<ClassifierJob>(5).await.unwrap();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].job, job);

    test.delete().await;
}

#[tokio::test]
async fn receive_count_rises_when_the_job_is_not_deleted() {
    let test = TestQueue::create().await;
    test.queue.send(scraper_job()).await.unwrap();

    let first = test.queue.receive::<ScraperJob>(5).await.unwrap();
    assert_eq!(first[0].receive_count, 1);

    tokio::time::sleep(Duration::from_secs(3)).await;
    let second = test.queue.receive::<ScraperJob>(5).await.unwrap();
    assert_eq!(second[0].job, first[0].job);
    assert_eq!(second[0].receive_count, 2);

    test.delete().await;
}

#[tokio::test]
async fn deleted_job_never_comes_back() {
    let test = TestQueue::create().await;
    test.queue.send(scraper_job()).await.unwrap();

    let received = test.queue.receive::<ScraperJob>(5).await.unwrap();
    test.queue
        .delete(&received[0].receipt_handle)
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(
        test.queue
            .receive::<ScraperJob>(1)
            .await
            .unwrap()
            .is_empty()
    );

    test.delete().await;
}
