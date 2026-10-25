use std::convert::Infallible;

use classifier::Classifier;
use db::{Vector, Versions};
use embedder::Embedder;
use types::{ClassifierJob, MetadataVersion, QuestionVersion};

use crate::{Context, Error};

/// Carries out steps 10 to 14 of the request flow.
pub struct ClassifierRole<C, E> {
    pub context: Context,
    pub classifier: C,
    pub embedder: E,
    pub metadata_version: MetadataVersion,
    pub question_version: QuestionVersion,
}

impl<C: Classifier + Sync, E: Embedder + Sync> ClassifierRole<C, E> {
    pub async fn run(&self) {
        tokio::select! {
            never = self.work() => match never {},
            never = self.context.read_dlq::<ClassifierJob>() => match never {},
        }
    }

    pub async fn work(&self) -> Infallible {
        self.context.work(async |job| self.handle(job).await).await
    }

    pub async fn handle(&self, job: &ClassifierJob) -> Result<(), Error> {
        let db = &self.context.db;
        if db::query_batch_exists(db, job.query_id, job.batch_number).await? {
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
                metadata::build(tweet, account, &versions.metadata, &versions.question)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (predictions, embeddings) = if inputs.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let predictions = self.classifier.classify(&inputs).await?;
            let texts: Vec<String> = rows.iter().map(|(tweet, _)| tweet.text.clone()).collect();
            let embeddings = self.embedder.embed(&texts).await?;
            let embeddings = rows
                .iter()
                .zip(embeddings)
                .map(|((tweet, _), embedding)| (tweet.id.clone(), Vector::from(embedding)))
                .collect();
            (predictions, embeddings)
        };
        db::finish_batch(
            db,
            job.query_id,
            job.batch_number,
            &predictions,
            &embeddings,
        )
        .await?;
        Ok(())
    }
}
