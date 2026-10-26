use std::convert::Infallible;

use classifier::Classifier;
use db::Versions;
use types::{ClassifierJob, MetadataVersion, QuestionVersion};

use crate::{Context, Error};

/// Carries out steps 10 to 13 of the request flow.
pub struct ClassifierRole<C> {
    pub context: Context,
    pub classifier: C,
    pub metadata_version: MetadataVersion,
    pub question_version: QuestionVersion,
}

impl<C: Classifier + Sync> ClassifierRole<C> {
    pub async fn run(&self) {
        tokio::select! {
            never = self.work() => match never {},
            never = self.context.read_dlq::<ClassifierJob>() => match never {},
        }
    }

    pub async fn work(&self) -> Infallible {
        self.context.work(async |job| self.handle(job).await).await
    }

    #[tracing::instrument(name = "classify_job", skip_all, fields(batch_number = job.batch_number))]
    pub async fn handle(&self, job: &ClassifierJob) -> Result<(), Error> {
        let db = &self.context.db;
        if db::query_batch_exists(db, job.query_id, job.batch_number).await? {
            tracing::info!("batch already finished, skipping the job");
            return Ok(());
        }
        let versions = Versions {
            model: self.classifier.model_version(),
            metadata: self.metadata_version.clone(),
            question: self.question_version.clone(),
        };
        let pending = db::tweet_ids_without_prediction(db, &job.tweet_ids, &versions).await?;
        let rows = db::tweets_with_accounts(db, &pending).await?;
        let inputs = rows
            .iter()
            .map(|(tweet, account)| {
                metadata::build(tweet, account, &[], &versions.metadata, &versions.question)
            })
            .collect::<Result<Vec<_>, _>>()?;
        tracing::info!(
            tweets = job.tweet_ids.len(),
            to_classify = inputs.len(),
            "classifying batch"
        );
        let predictions = if inputs.is_empty() {
            Vec::new()
        } else {
            self.classifier.classify(&inputs).await?
        };
        let done = db::finish_batch(db, job.query_id, job.batch_number, &predictions).await?;
        tracing::info!(predictions = predictions.len(), "batch finished");
        if done {
            self.context.log_queued_to_done(job.query_id).await;
        }
        Ok(())
    }
}
