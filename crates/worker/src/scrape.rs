use std::convert::Infallible;

use queue::Queue;
use scraper::Scraper;
use types::{ClassifierJob, QueryStatus, ScraperJob};

use crate::{Context, Error};

/// Carries out steps 5 to 9 of the request flow.
pub struct ScraperRole<S> {
    pub context: Context,
    pub scraper: S,
    pub classifier_jobs: Queue,
    pub batch_size: usize,
}

impl<S: Scraper + Sync> ScraperRole<S> {
    pub async fn run(&self) {
        tokio::select! {
            never = self.work() => match never {},
            never = self.context.read_dlq::<ScraperJob>() => match never {},
        }
    }

    pub async fn work(&self) -> Infallible {
        self.context.work(async |job| self.handle(job).await).await
    }

    #[tracing::instrument(name = "scrape_job", skip_all)]
    pub async fn handle(&self, job: &ScraperJob) -> Result<(), Error> {
        let db = &self.context.db;
        let id = job.query_id;
        let query = db::fetch_query(db, id)
            .await?
            .ok_or(Error::UnknownQuery(id))?;
        let classifying = query.status == QueryStatus::Classifying;
        if classifying {
            tracing::info!("query already classifying, sending its batches again");
        } else {
            if !db::set_status(db, id, QueryStatus::Scraping, None).await? {
                tracing::info!(status = ?query.status, "query is past scraping, skipping the job");
                return Ok(());
            }
            tracing::info!(text = %query.text, "scraping");
            let scraped = self.scraper.scrape(&query.text).await?;
            for account in &scraped.accounts {
                db::upsert_account(db, account).await?;
            }
            for tweet in &scraped.tweets {
                db::upsert_tweet(db, tweet).await?;
                db::upsert_query_tweet(db, id, &tweet.id).await?;
            }
            tracing::info!(
                tweets = scraped.tweets.len(),
                accounts = scraped.accounts.len(),
                "tweets and accounts saved"
            );
        }
        let tweet_ids = db::query_tweet_ids(db, id).await?;
        if tweet_ids.is_empty() {
            db::set_status(db, id, QueryStatus::Done, None).await?;
            tracing::info!("no tweets, query done");
            return Ok(());
        }
        let batches: Vec<_> = tweet_ids.chunks(self.batch_size).collect();
        let batch_count = batches.len();
        if !classifying && !db::set_batch_count(db, id, batch_count as i32).await? {
            tracing::info!("another job set the batch count, skipping the job");
            return Ok(());
        }
        for (batch_number, tweet_ids) in batches.into_iter().enumerate() {
            self.classifier_jobs
                .send(ClassifierJob {
                    query_id: id,
                    batch_number: batch_number as i32,
                    tweet_ids: tweet_ids.to_vec(),
                    trace_context: None,
                })
                .await?;
        }
        tracing::info!(
            batch_count,
            batch_size = self.batch_size,
            "classifier jobs sent, query classifying"
        );
        Ok(())
    }
}
