use sqlx::{PgExecutor, migrate::Migrator};
use types::{
    Account, MetadataVersion, ModelVersion, Prediction, Query, QueryStatus, QuestionVersion, Tweet,
};
use uuid::Uuid;

pub use sqlx::{Error, PgPool};

pub static MIGRATOR: Migrator = sqlx::migrate!();

pub struct Versions {
    pub model: ModelVersion,
    pub metadata: MetadataVersion,
    pub question: QuestionVersion,
}

pub struct ResultTweet {
    pub tweet: Tweet,
    pub account: Account,
    pub prediction: Option<Prediction>,
}

pub async fn ping<'e>(db: impl PgExecutor<'e>) -> Result<(), Error> {
    sqlx::query!("SELECT 1 AS ok").fetch_one(db).await?;
    Ok(())
}

pub async fn insert_query<'e>(db: impl PgExecutor<'e>, text: &str) -> Result<Query, Error> {
    sqlx::query_as!(
        Query,
        r#"INSERT INTO queries (id, text) VALUES ($1, $2)
        RETURNING id, text, status AS "status: QueryStatus", error, batch_count, created_at, updated_at"#,
        Uuid::new_v4(),
        text
    )
    .fetch_one(db)
    .await
}

pub async fn fetch_query<'e>(db: impl PgExecutor<'e>, id: Uuid) -> Result<Option<Query>, Error> {
    sqlx::query_as!(
        Query,
        r#"SELECT id, text, status AS "status: QueryStatus", error, batch_count, created_at, updated_at
        FROM queries WHERE id = $1"#,
        id
    )
    .fetch_optional(db)
    .await
}

/// Returns whether Postgres applied the change. `error` is stored as given.
pub async fn set_status<'e>(
    db: impl PgExecutor<'e>,
    id: Uuid,
    status: QueryStatus,
    error: Option<&str>,
) -> Result<bool, Error> {
    use QueryStatus::*;
    let allowed: &[QueryStatus] = match status {
        Queued => &[],
        Scraping => &[Queued, Scraping],
        Classifying => &[Scraping],
        Done => &[Scraping, Classifying],
        Failed => &[Queued, Scraping, Classifying],
    };
    let result = sqlx::query!(
        "UPDATE queries SET status = $2, error = $3, updated_at = NOW()
        WHERE id = $1 AND status = ANY($4::query_status[])",
        id,
        status as QueryStatus,
        error,
        allowed as &[QueryStatus]
    )
    .execute(db)
    .await?;
    Ok(result.rows_affected() == 1)
}

pub async fn upsert_account<'e>(db: impl PgExecutor<'e>, account: &Account) -> Result<u64, Error> {
    let result = sqlx::query!(
        "INSERT INTO accounts (id, handle, raw) VALUES ($1, $2, $3)
        ON CONFLICT (id) DO UPDATE
        SET handle = EXCLUDED.handle, raw = EXCLUDED.raw, updated_at = NOW()
        WHERE (accounts.handle, accounts.raw) IS DISTINCT FROM (EXCLUDED.handle, EXCLUDED.raw)",
        account.id,
        account.handle,
        account.raw
    )
    .execute(db)
    .await?;
    Ok(result.rows_affected())
}

pub async fn upsert_tweet<'e>(db: impl PgExecutor<'e>, tweet: &Tweet) -> Result<u64, Error> {
    let result = sqlx::query!(
        "INSERT INTO tweets (id, account_id, text, posted_at, raw) VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (id) DO UPDATE
        SET account_id = EXCLUDED.account_id, text = EXCLUDED.text,
            posted_at = EXCLUDED.posted_at, raw = EXCLUDED.raw, updated_at = NOW()
        WHERE (tweets.account_id, tweets.text, tweets.posted_at, tweets.raw)
            IS DISTINCT FROM (EXCLUDED.account_id, EXCLUDED.text, EXCLUDED.posted_at, EXCLUDED.raw)",
        tweet.id,
        tweet.account_id,
        tweet.text,
        tweet.posted_at,
        tweet.raw
    )
    .execute(db)
    .await?;
    Ok(result.rows_affected())
}

pub async fn upsert_query_tweet<'e>(
    db: impl PgExecutor<'e>,
    query_id: Uuid,
    tweet_id: &str,
) -> Result<u64, Error> {
    let result = sqlx::query!(
        "INSERT INTO query_tweets (query_id, tweet_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        query_id,
        tweet_id
    )
    .execute(db)
    .await?;
    Ok(result.rows_affected())
}

pub async fn upsert_query_batch<'e>(
    db: impl PgExecutor<'e>,
    query_id: Uuid,
    batch_number: i32,
) -> Result<u64, Error> {
    let result = sqlx::query!(
        "INSERT INTO query_batches (query_id, batch_number) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        query_id,
        batch_number
    )
    .execute(db)
    .await?;
    Ok(result.rows_affected())
}

pub async fn upsert_prediction<'e>(
    db: impl PgExecutor<'e>,
    prediction: &Prediction,
) -> Result<u64, Error> {
    let result = sqlx::query!(
        "INSERT INTO predictions
            (tweet_id, model_version, metadata_version, question_version, p_bot, raw)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (tweet_id, model_version, metadata_version, question_version) DO UPDATE
        SET p_bot = EXCLUDED.p_bot, raw = EXCLUDED.raw
        WHERE (predictions.p_bot, predictions.raw) IS DISTINCT FROM (EXCLUDED.p_bot, EXCLUDED.raw)",
        prediction.tweet_id,
        &prediction.model_version as &ModelVersion,
        &prediction.metadata_version as &MetadataVersion,
        &prediction.question_version as &QuestionVersion,
        prediction.p_bot,
        prediction.raw
    )
    .execute(db)
    .await?;
    Ok(result.rows_affected())
}

/// Returns up to `limit` tweets of the query, sorted by tweet ID, that follow the tweet ID `after`.
pub async fn result_page<'e>(
    db: impl PgExecutor<'e>,
    query_id: Uuid,
    versions: &Versions,
    after: Option<&str>,
    limit: i64,
) -> Result<Vec<ResultTweet>, Error> {
    let rows = sqlx::query!(
        r#"SELECT t.id, t.account_id, t.text, t.posted_at, t.raw,
            a.handle, a.raw AS account_raw,
            p.p_bot AS "p_bot?", p.raw AS "prediction_raw?"
        FROM query_tweets qt
        JOIN tweets t ON t.id = qt.tweet_id
        JOIN accounts a ON a.id = t.account_id
        LEFT JOIN predictions p ON p.tweet_id = t.id
            AND p.model_version = $2 AND p.metadata_version = $3 AND p.question_version = $4
        WHERE qt.query_id = $1 AND ($5::TEXT IS NULL OR t.id > $5)
        ORDER BY t.id
        LIMIT $6"#,
        query_id,
        &versions.model as &ModelVersion,
        &versions.metadata as &MetadataVersion,
        &versions.question as &QuestionVersion,
        after,
        limit
    )
    .fetch_all(db)
    .await?;
    Ok(rows
        .into_iter()
        .map(|row| ResultTweet {
            prediction: row
                .p_bot
                .zip(row.prediction_raw)
                .map(|(p_bot, raw)| Prediction {
                    tweet_id: row.id.clone(),
                    model_version: versions.model.clone(),
                    metadata_version: versions.metadata.clone(),
                    question_version: versions.question.clone(),
                    p_bot,
                    raw,
                }),
            account: Account {
                id: row.account_id.clone(),
                handle: row.handle,
                raw: row.account_raw,
            },
            tweet: Tweet {
                id: row.id,
                account_id: row.account_id,
                text: row.text,
                posted_at: row.posted_at,
                raw: row.raw,
            },
        })
        .collect())
}
