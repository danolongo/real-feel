use std::{
    convert::Infallible,
    time::{Duration, Instant},
};

use futures::future::join_all;
use queue::{Job, Queue, Received};
use tracing::Instrument;
use types::{ClassifierJob, QueryStatus, ScraperJob};
use uuid::Uuid;

mod classify;
mod scrape;

pub use classify::ClassifierRole;
pub use scrape::ScraperRole;

/// The `maxReceiveCount` of every queue. A failure on this receive sets the query `failed`.
const FINAL_RECEIVE: u32 = 5;
const WAIT_SECONDS: i32 = 5;
const RETRY_PAUSE: Duration = Duration::from_secs(1);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Postgres failed: {0}")]
    Db(#[from] db::Error),
    #[error("SQS failed: {0}")]
    Queue(#[from] queue::Error),
    #[error("scraper failed: {0}")]
    Scraper(#[from] scraper::Error),
    #[error("classifier failed: {0}")]
    Classifier(#[from] classifier::Error),
    #[error("metadata failed: {0}")]
    Metadata(#[from] metadata::Error),
    #[error("query {0} does not exist")]
    UnknownQuery(Uuid),
    #[error("job exceeded its deadline")]
    Deadline,
}

/// A job that names the query it works on.
pub trait QueryJob: Job {
    fn query_id(&self) -> Uuid;
}

impl QueryJob for ScraperJob {
    fn query_id(&self) -> Uuid {
        self.query_id
    }
}

impl QueryJob for ClassifierJob {
    fn query_id(&self) -> Uuid {
        self.query_id
    }
}

/// What both roles share: Postgres, the role's queue, its dead-letter queue, and the job deadline.
pub struct Context {
    pub db: db::PgPool,
    pub jobs: Queue,
    pub dlq: Queue,
    pub deadline: Duration,
}

impl Context {
    /// Takes jobs from the queue without end. Each job runs under the job deadline.
    /// The worker deletes a job after `handle` returns. `handle` commits to Postgres first.
    pub async fn work<J: QueryJob>(
        &self,
        handle: impl AsyncFn(&J) -> Result<(), Error>,
    ) -> Infallible {
        loop {
            match self.jobs.receive::<J>(WAIT_SECONDS).await {
                Ok(batch) => {
                    for broken in &batch.broken {
                        tracing::error!(
                            message_id = ?broken.message_id,
                            error = %broken.error,
                            "broken message left in the queue for the dead-letter queue"
                        );
                    }
                    join_all(
                        batch
                            .jobs
                            .iter()
                            .map(|received| self.process(received, &handle)),
                    )
                    .await;
                }
                Err(error) => {
                    tracing::error!(%error, "cannot receive jobs");
                    tokio::time::sleep(RETRY_PAUSE).await;
                }
            }
        }
    }

    async fn process<J: QueryJob>(
        &self,
        received: &Received<J>,
        handle: &impl AsyncFn(&J) -> Result<(), Error>,
    ) {
        let span = tracing::info_span!(
            "job",
            query_id = %received.job.query_id(),
            receive_count = received.receive_count,
            trace_id = tracing::field::Empty
        );
        received.continue_trace(&span);
        if let Some(trace_id) = telemetry::trace_id(&span) {
            span.record("trace_id", trace_id);
        }
        span.in_scope(|| tracing::info!("job received"));
        let started = Instant::now();
        let outcome = async {
            tokio::time::timeout(self.deadline, handle(&received.job))
                .await
                .unwrap_or(Err(Error::Deadline))
        }
        .instrument(span.clone())
        .await;
        let _entered = span.enter();
        match outcome {
            Ok(()) => {
                let elapsed_ms = started.elapsed().as_millis() as u64;
                match self.jobs.delete(&received.receipt_handle).await {
                    Ok(()) => tracing::info!(elapsed_ms, "job done"),
                    Err(error) => tracing::error!(%error, "cannot delete the job"),
                }
            }
            Err(error) => {
                tracing::warn!(%error, elapsed_ms = started.elapsed().as_millis() as u64, "job failed");
                if received.receive_count >= FINAL_RECEIVE {
                    self.fail(received.job.query_id(), &error.to_string()).await;
                }
            }
        }
    }

    /// Reads the dead-letter queue without end. Sets each job's query `failed`, then deletes the job.
    /// Deletes each broken message after it logs the message.
    pub async fn read_dlq<J: QueryJob>(&self) -> Infallible {
        loop {
            match self.dlq.receive::<J>(WAIT_SECONDS).await {
                Ok(batch) => {
                    for broken in &batch.broken {
                        tracing::error!(
                            message_id = ?broken.message_id,
                            error = %broken.error,
                            "broken dead-letter message, deleting it"
                        );
                        if let Err(error) = self.dlq.delete(&broken.receipt_handle).await {
                            tracing::error!(%error, "cannot delete the broken dead-letter message");
                        }
                    }
                    for received in batch.jobs {
                        tracing::warn!(
                            query_id = %received.job.query_id(),
                            "dead-letter job found, failing the query"
                        );
                        let failed = db::set_status(
                            &self.db,
                            received.job.query_id(),
                            QueryStatus::Failed,
                            Some("the job reached the dead-letter queue"),
                        )
                        .await;
                        match failed {
                            Ok(_) => {
                                if let Err(error) = self.dlq.delete(&received.receipt_handle).await
                                {
                                    tracing::error!(%error, "cannot delete the dead-letter job");
                                }
                            }
                            Err(error) => tracing::error!(%error, "cannot set the query failed"),
                        }
                    }
                }
                Err(error) => {
                    tracing::error!(%error, "cannot receive dead-letter jobs");
                    tokio::time::sleep(RETRY_PAUSE).await;
                }
            }
        }
    }

    /// Logs `queued_to_done_ms`, the time from the creation of the query to now.
    async fn log_queued_to_done(&self, query_id: Uuid) {
        match db::fetch_query(&self.db, query_id).await {
            Ok(Some(query)) => tracing::info!(
                %query_id,
                queued_to_done_ms = (chrono::Utc::now() - query.created_at).num_milliseconds(),
                "query done"
            ),
            Ok(None) => {}
            Err(error) => tracing::warn!(%error, "cannot read the query to log its time to done"),
        }
    }

    async fn fail(&self, query_id: Uuid, error: &str) {
        if let Err(error) =
            db::set_status(&self.db, query_id, QueryStatus::Failed, Some(error)).await
        {
            tracing::error!(%error, "cannot set the query failed");
        }
    }
}
