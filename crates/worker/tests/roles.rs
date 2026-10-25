#![cfg(feature = "integration")]

use std::time::Duration;

use aws_sdk_sqs::{Client, types::QueueAttributeName};
use classifier::Fake as FakeClassifier;
use db::{PgPool, Versions};
use embedder::Fake as FakeEmbedder;
use queue::Queue;
use scraper::Fixture;
use serde_json::json;
use types::{
    ClassifierJob, MetadataVersion, ModelVersion, QueryStatus, QuestionVersion, ScraperJob,
};
use uuid::Uuid;
use worker::{ClassifierRole, Context, ScraperRole};

const WAIT: Duration = Duration::from_secs(60);

/// A queue and its dead-letter queue, with a 1-second visibility timeout and `maxReceiveCount` 5.
struct TestQueues {
    client: Client,
    jobs_url: String,
    dlq_url: String,
}

impl TestQueues {
    async fn create(client: &Client) -> Self {
        let name = format!("test-{}", Uuid::new_v4());
        let create = |name: String| async {
            client
                .create_queue()
                .queue_name(name)
                .attributes(QueueAttributeName::VisibilityTimeout, "1")
        };
        let dlq_url = create(format!("{name}-dlq"))
            .await
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
        let jobs_url = create(name)
            .await
            .attributes(QueueAttributeName::RedrivePolicy, policy)
            .send()
            .await
            .unwrap()
            .queue_url
            .unwrap();
        Self {
            client: client.clone(),
            jobs_url,
            dlq_url,
        }
    }

    async fn delete(self) {
        for url in [self.jobs_url, self.dlq_url] {
            self.client
                .delete_queue()
                .queue_url(url)
                .send()
                .await
                .unwrap();
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

struct Stack {
    db: PgPool,
    scraper_queues: TestQueues,
    classifier_queues: TestQueues,
}

impl Stack {
    async fn create() -> Self {
        let client =
            Client::new(&aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await);
        Self {
            db: PgPool::connect(&std::env::var("DATABASE_URL").unwrap())
                .await
                .unwrap(),
            scraper_queues: TestQueues::create(&client).await,
            classifier_queues: TestQueues::create(&client).await,
        }
    }

    async fn delete(self) {
        self.scraper_queues.delete().await;
        self.classifier_queues.delete().await;
    }

    async fn scraper_role(&self, scraper: Fixture) -> ScraperRole<Fixture> {
        ScraperRole {
            context: self.scraper_queues.context(&self.db).await,
            scraper,
            classifier_jobs: Queue::connect(&self.classifier_queues.jobs_url).await,
            batch_size: 2,
        }
    }

    async fn classifier_role(
        &self,
        classifier: FakeClassifier,
    ) -> ClassifierRole<FakeClassifier, FakeEmbedder> {
        ClassifierRole {
            context: self.classifier_queues.context(&self.db).await,
            classifier,
            embedder: FakeEmbedder,
            metadata_version: MetadataVersion("v1".into()),
            question_version: QuestionVersion("v1".into()),
        }
    }

    async fn send_scraper_job(&self, query_id: Uuid) {
        Queue::connect(&self.scraper_queues.jobs_url)
            .await
            .send(ScraperJob {
                query_id,
                trace_context: None,
            })
            .await
            .unwrap();
    }

    async fn status(&self, query_id: Uuid) -> QueryStatus {
        db::fetch_query(&self.db, query_id)
            .await
            .unwrap()
            .unwrap()
            .status
    }

    async fn wait_for(&self, query_id: Uuid, status: QueryStatus) {
        tokio::time::timeout(WAIT, async {
            while self.status(query_id).await != status {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("query never reached {status:?}"));
    }

    async fn predictions(&self, query_id: Uuid) -> usize {
        let versions = Versions {
            model: ModelVersion("fake".into()),
            metadata: MetadataVersion("v1".into()),
            question: QuestionVersion("v1".into()),
        };
        db::result_page(&self.db, query_id, &versions, None, 100)
            .await
            .unwrap()
            .iter()
            .filter(|row| row.prediction.is_some())
            .count()
    }

    async fn batch_rows(&self, query_id: Uuid) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM query_batches WHERE query_id = $1")
            .bind(query_id)
            .fetch_one(&self.db)
            .await
            .unwrap()
    }

    /// Inserts a query that the scraper worker has already scraped into 5 tweets and 3 batches.
    async fn classifying_query(&self) -> (Uuid, Vec<ClassifierJob>) {
        let query = db::insert_query(&self.db, "ai").await.unwrap();
        let scraper_role = self.scraper_role(fixture()).await;
        scraper_role
            .handle(&ScraperJob {
                query_id: query.id,
                trace_context: None,
            })
            .await
            .unwrap();
        let ids = db::query_tweet_ids(&self.db, query.id).await.unwrap();
        let jobs = ids
            .chunks(2)
            .enumerate()
            .map(|(number, ids)| ClassifierJob {
                query_id: query.id,
                batch_number: number as i32,
                tweet_ids: ids.to_vec(),
                trace_context: None,
            })
            .collect();
        (query.id, jobs)
    }
}

/// Loads the shape of the default fixture, 5 tweets, with IDs that no other test run shares.
fn fixture() -> Fixture {
    let run = Uuid::new_v4();
    let tweets: Vec<_> = (0..5)
        .map(|n| json!({ "id": format!("{run}-{n}"), "account_id": run.to_string(), "text": format!("tweet {n}"), "raw": {} }))
        .collect();
    load(json!({
        "accounts": [{ "id": run.to_string(), "handle": "fixture", "raw": {} }],
        "tweets": tweets,
    }))
}

fn load(scraped: serde_json::Value) -> Fixture {
    let path = std::env::temp_dir().join(format!("{}.json", Uuid::new_v4()));
    std::fs::write(&path, scraped.to_string()).unwrap();
    let fixture = Fixture::load(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    fixture
}

#[tokio::test]
async fn one_scraper_job_ends_done_with_a_prediction_for_every_tweet() {
    let stack = Stack::create().await;
    let query = db::insert_query(&stack.db, "ai").await.unwrap();
    let scraper_role = stack.scraper_role(fixture()).await;
    let classifier_role = stack.classifier_role(FakeClassifier::new(0.5, 0)).await;
    stack.send_scraper_job(query.id).await;

    tokio::select! {
        () = scraper_role.run() => {}
        () = classifier_role.run() => {}
        () = stack.wait_for(query.id, QueryStatus::Done) => {}
    }

    assert_eq!(stack.predictions(query.id).await, 5);
    assert_eq!(stack.batch_rows(query.id).await, 3);
    stack.delete().await;
}

#[tokio::test]
async fn a_duplicated_classifier_job_changes_nothing() {
    let stack = Stack::create().await;
    let (query_id, jobs) = stack.classifying_query().await;
    let classifier_role = stack.classifier_role(FakeClassifier::new(0.5, 0)).await;

    for job in jobs.iter().chain(&jobs) {
        classifier_role.handle(job).await.unwrap();
    }

    assert_eq!(stack.status(query_id).await, QueryStatus::Done);
    assert_eq!(stack.predictions(query_id).await, 5);
    assert_eq!(stack.batch_rows(query_id).await, 3);
    stack.delete().await;
}

#[tokio::test]
async fn a_scraper_job_redelivered_after_classifying_still_ends_done() {
    let stack = Stack::create().await;
    let (query_id, jobs) = stack.classifying_query().await;
    assert_eq!(stack.status(query_id).await, QueryStatus::Classifying);
    let scraper_role = stack.scraper_role(fixture()).await;
    scraper_role
        .handle(&ScraperJob {
            query_id,
            trace_context: None,
        })
        .await
        .unwrap();
    let classifier_role = stack.classifier_role(FakeClassifier::new(0.5, 0)).await;
    classifier_role.handle(&jobs[0]).await.unwrap();

    tokio::select! {
        () = classifier_role.run() => {}
        () = stack.wait_for(query_id, QueryStatus::Done) => {}
    }

    assert_eq!(stack.batch_rows(query_id).await, 3);
    assert_eq!(stack.predictions(query_id).await, 5);
    stack.delete().await;
}

#[tokio::test]
async fn zero_scraped_tweets_end_done() {
    let stack = Stack::create().await;
    let query = db::insert_query(&stack.db, "ai").await.unwrap();
    let scraper_role = stack
        .scraper_role(load(json!({ "accounts": [], "tweets": [] })))
        .await;
    stack.send_scraper_job(query.id).await;

    tokio::select! {
        () = scraper_role.run() => {}
        () = stack.wait_for(query.id, QueryStatus::Done) => {}
    }

    assert_eq!(stack.batch_rows(query.id).await, 0);
    stack.delete().await;
}

#[tokio::test]
async fn a_job_that_always_fails_reaches_the_dlq_and_fails_the_query() {
    let stack = Stack::create().await;
    let (query_id, jobs) = stack.classifying_query().await;
    let jobs_queue = Queue::connect(&stack.classifier_queues.jobs_url).await;
    jobs_queue.send(jobs[0].clone()).await.unwrap();
    let classifier_role = stack
        .classifier_role(FakeClassifier::new(0.5, usize::MAX))
        .await;

    // `work` only: the dead-letter queue reader would delete the job before the test sees it.
    tokio::select! {
        never = classifier_role.work() => match never {},
        () = stack.wait_for(query_id, QueryStatus::Failed) => {}
    }

    let dead = Queue::connect(&stack.classifier_queues.dlq_url).await;
    let found = tokio::time::timeout(WAIT, async {
        loop {
            tokio::select! {
                never = classifier_role.work() => match never {},
                found = dead.receive::<ClassifierJob>(1) => {
                    if let Some(found) = found.unwrap().pop() {
                        break found;
                    }
                }
            }
        }
    })
    .await
    .expect("the job never reached the dead-letter queue");
    assert_eq!(found.job.query_id, query_id);
    assert_eq!(stack.predictions(query_id).await, 0);
    stack.delete().await;
}

#[tokio::test]
async fn the_dlq_reader_fails_a_query_that_the_worker_left_open() {
    let stack = Stack::create().await;
    let (query_id, jobs) = stack.classifying_query().await;
    Queue::connect(&stack.classifier_queues.dlq_url)
        .await
        .send(jobs[0].clone())
        .await
        .unwrap();
    let classifier_role = stack.classifier_role(FakeClassifier::new(0.5, 0)).await;
    assert_eq!(stack.status(query_id).await, QueryStatus::Classifying);

    tokio::select! {
        () = classifier_role.run() => {}
        () = stack.wait_for(query_id, QueryStatus::Failed) => {}
    }

    let query = db::fetch_query(&stack.db, query_id).await.unwrap().unwrap();
    assert!(query.error.is_some());
    stack.delete().await;
}
