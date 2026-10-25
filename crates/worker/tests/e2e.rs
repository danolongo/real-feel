#![cfg(feature = "e2e")]

use std::{sync::Arc, time::Duration};

use api::{AppState, router};
use aws_sdk_sqs::{Client, types::QueueAttributeName};
use classifier::Fake as FakeClassifier;
use db::{PgPool, Versions};
use queue::Queue;
use scraper::Fixture;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use types::{MetadataVersion, ModelVersion, QuestionVersion};
use uuid::Uuid;
use worker::{ClassifierRole, Context, ScraperRole};

const WAIT: Duration = Duration::from_secs(60);
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../scraper/fixtures/default.json"
);

/// A queue and its dead-letter queue, with `maxReceiveCount` 5.
struct TestQueues {
    jobs_url: String,
    dlq_url: String,
}

impl TestQueues {
    async fn create(client: &Client) -> Self {
        let name = format!("test-{}", Uuid::new_v4());
        let dlq_url = client
            .create_queue()
            .queue_name(format!("{name}-dlq"))
            .send()
            .await
            .unwrap()
            .queue_url
            .unwrap();
        let arn = client
            .get_queue_attributes()
            .queue_url(&dlq_url)
            .attribute_names(QueueAttributeName::QueueArn)
            .send()
            .await
            .unwrap()
            .attributes
            .unwrap()[&QueueAttributeName::QueueArn]
            .clone();
        let policy = format!(r#"{{"deadLetterTargetArn":"{arn}","maxReceiveCount":"5"}}"#);
        let jobs_url = client
            .create_queue()
            .queue_name(name)
            .attributes(QueueAttributeName::RedrivePolicy, policy)
            .send()
            .await
            .unwrap()
            .queue_url
            .unwrap();
        Self { jobs_url, dlq_url }
    }

    async fn delete(self, client: &Client) {
        for url in [self.jobs_url, self.dlq_url] {
            client.delete_queue().queue_url(url).send().await.unwrap();
        }
    }

    async fn context(&self, db: &PgPool) -> Context {
        Context {
            db: db.clone(),
            jobs: Queue::connect(&self.jobs_url).await,
            dlq: Queue::connect(&self.dlq_url).await,
            deadline: Duration::from_secs(30),
        }
    }
}

#[tokio::test]
async fn a_query_over_http_ends_done_with_a_prediction_for_every_tweet() {
    let client =
        Client::new(&aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await);
    let db = PgPool::connect(&std::env::var("DATABASE_URL").unwrap())
        .await
        .unwrap();
    let scraper_queues = TestQueues::create(&client).await;
    let classifier_queues = TestQueues::create(&client).await;

    let scraper_role = ScraperRole {
        context: scraper_queues.context(&db).await,
        scraper: Fixture::load(FIXTURE).unwrap(),
        classifier_jobs: Queue::connect(&classifier_queues.jobs_url).await,
        batch_size: 2,
    };
    let classifier_role = ClassifierRole {
        context: classifier_queues.context(&db).await,
        classifier: FakeClassifier::new(0.5, 0),
        metadata_version: MetadataVersion("v1".into()),
        question_version: QuestionVersion("v1".into()),
    };
    let app = router(AppState {
        db: db.clone(),
        queue: Queue::connect(&scraper_queues.jobs_url).await,
        versions: Arc::new(Versions {
            model: ModelVersion("fake".into()),
            metadata: MetadataVersion("v1".into()),
            question: QuestionVersion("v1".into()),
        }),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());

    let http = reqwest::Client::new();
    let exchange = async {
        let created: Value = http
            .post(format!("{base}/queries"))
            .json(&json!({ "text": "ai" }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let url = format!("{base}/queries/{}", created["query_id"].as_str().unwrap());
        tokio::time::timeout(WAIT, async {
            loop {
                let query: Value = http.get(&url).send().await.unwrap().json().await.unwrap();
                assert_ne!(query["status"], "failed", "{query}");
                if query["status"] == "done" {
                    break query;
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        })
        .await
        .expect("the query never reached done")
    };

    let query = tokio::select! {
        served = axum::serve(listener, app) => panic!("the API stopped: {served:?}"),
        () = scraper_role.run() => unreachable!("the scraper role ended"),
        () = classifier_role.run() => unreachable!("the classifier role ended"),
        query = exchange => query,
    };

    let tweets = query["result"]["tweets"].as_array().unwrap();
    assert_eq!(tweets.len(), 5);
    assert!(tweets.iter().all(|tweet| tweet["p_bot"].is_number()));
    scraper_queues.delete(&client).await;
    classifier_queues.delete(&client).await;
}
