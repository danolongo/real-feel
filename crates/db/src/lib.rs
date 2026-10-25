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
        "INSERT INTO accounts
            (id, handle, joined_at, followers, following, tweet_count, is_verified, is_automated, raw)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (id) DO UPDATE
        SET handle = EXCLUDED.handle, joined_at = EXCLUDED.joined_at,
            followers = EXCLUDED.followers, following = EXCLUDED.following,
            tweet_count = EXCLUDED.tweet_count, is_verified = EXCLUDED.is_verified,
            is_automated = EXCLUDED.is_automated, raw = EXCLUDED.raw, updated_at = NOW()
        WHERE (accounts.handle, accounts.joined_at, accounts.followers, accounts.following,
                accounts.tweet_count, accounts.is_verified, accounts.is_automated, accounts.raw)
            IS DISTINCT FROM (EXCLUDED.handle, EXCLUDED.joined_at, EXCLUDED.followers,
                EXCLUDED.following, EXCLUDED.tweet_count, EXCLUDED.is_verified,
                EXCLUDED.is_automated, EXCLUDED.raw)",
        account.id,
        account.handle,
        account.joined_at,
        account.followers,
        account.following,
        account.tweet_count,
        account.is_verified,
        account.is_automated,
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
            a.handle, a.joined_at, a.followers, a.following, a.tweet_count,
            a.is_verified, a.is_automated, a.raw AS account_raw,
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
                joined_at: row.joined_at,
                followers: row.followers,
                following: row.following,
                tweet_count: row.tweet_count,
                is_verified: row.is_verified,
                is_automated: row.is_automated,
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

/// Sets the batch count and the status `classifying` in one `UPDATE`.
/// Postgres applies it only while the status is `scraping` and the batch count is null or equal to `batch_count`.
/// Returns whether Postgres applied it.
pub async fn set_batch_count<'e>(
    db: impl PgExecutor<'e>,
    id: Uuid,
    batch_count: i32,
) -> Result<bool, Error> {
    let result = sqlx::query!(
        "UPDATE queries SET batch_count = $2, status = 'classifying', updated_at = NOW()
        WHERE id = $1 AND status = 'scraping' AND (batch_count IS NULL OR batch_count = $2)",
        id,
        batch_count
    )
    .execute(db)
    .await?;
    Ok(result.rows_affected() == 1)
}

/// Returns the tweet IDs of the query, sorted by tweet ID.
pub async fn query_tweet_ids<'e>(
    db: impl PgExecutor<'e>,
    query_id: Uuid,
) -> Result<Vec<String>, Error> {
    sqlx::query_scalar!(
        "SELECT tweet_id FROM query_tweets WHERE query_id = $1 ORDER BY tweet_id",
        query_id
    )
    .fetch_all(db)
    .await
}

/// Returns the tweets with these IDs and their accounts, sorted by tweet ID.
pub async fn tweets_with_accounts<'e>(
    db: impl PgExecutor<'e>,
    tweet_ids: &[String],
) -> Result<Vec<(Tweet, Account)>, Error> {
    let rows = sqlx::query!(
        "SELECT t.id, t.account_id, t.text, t.posted_at, t.raw, a.handle, a.joined_at,
            a.followers, a.following, a.tweet_count, a.is_verified, a.is_automated,
            a.raw AS account_raw
        FROM tweets t
        JOIN accounts a ON a.id = t.account_id
        WHERE t.id = ANY($1)
        ORDER BY t.id",
        tweet_ids
    )
    .fetch_all(db)
    .await?;
    Ok(rows
        .into_iter()
        .map(|row| {
            let account = Account {
                id: row.account_id.clone(),
                handle: row.handle,
                joined_at: row.joined_at,
                followers: row.followers,
                following: row.following,
                tweet_count: row.tweet_count,
                is_verified: row.is_verified,
                is_automated: row.is_automated,
                raw: row.account_raw,
            };
            let tweet = Tweet {
                id: row.id,
                account_id: row.account_id,
                text: row.text,
                posted_at: row.posted_at,
                raw: row.raw,
            };
            (tweet, account)
        })
        .collect())
}

pub async fn query_batch_exists<'e>(
    db: impl PgExecutor<'e>,
    query_id: Uuid,
    batch_number: i32,
) -> Result<bool, Error> {
    sqlx::query_scalar!(
        r#"SELECT EXISTS (
            SELECT 1 FROM query_batches WHERE query_id = $1 AND batch_number = $2
        ) AS "exists!""#,
        query_id,
        batch_number
    )
    .fetch_one(db)
    .await
}

/// Returns the tweet IDs that have no prediction for `versions`, in the order given.
pub async fn tweet_ids_without_prediction<'e>(
    db: impl PgExecutor<'e>,
    tweet_ids: &[String],
    versions: &Versions,
) -> Result<Vec<String>, Error> {
    sqlx::query_scalar!(
        r#"SELECT t.id AS "id!"
        FROM UNNEST($1::TEXT[]) WITH ORDINALITY AS t (id, position)
        WHERE NOT EXISTS (
            SELECT 1 FROM predictions p
            WHERE p.tweet_id = t.id
                AND p.model_version = $2 AND p.metadata_version = $3 AND p.question_version = $4
        )
        ORDER BY t.position"#,
        tweet_ids,
        &versions.model as &ModelVersion,
        &versions.metadata as &MetadataVersion,
        &versions.question as &QuestionVersion
    )
    .fetch_all(db)
    .await
}

/// Commits one batch in one transaction. The row lock on the query makes two workers that
/// finish the last two batches at once count one after the other.
/// Returns whether this call set the status `done`.
pub async fn finish_batch(
    pool: &PgPool,
    query_id: Uuid,
    batch_number: i32,
    predictions: &[Prediction],
) -> Result<bool, Error> {
    let mut tx = pool.begin().await?;
    let batch_count = sqlx::query_scalar!(
        "SELECT batch_count FROM queries WHERE id = $1 FOR UPDATE",
        query_id
    )
    .fetch_one(&mut *tx)
    .await?;
    for prediction in predictions {
        upsert_prediction(&mut *tx, prediction).await?;
    }
    upsert_query_batch(&mut *tx, query_id, batch_number).await?;
    let finished = sqlx::query_scalar!(
        r#"SELECT COUNT(*) AS "count!" FROM query_batches WHERE query_id = $1"#,
        query_id
    )
    .fetch_one(&mut *tx)
    .await?;
    let done = match batch_count {
        Some(batch_count) if finished == i64::from(batch_count) => {
            set_status(&mut *tx, query_id, QueryStatus::Done, None).await?
        }
        _ => false,
    };
    tx.commit().await?;
    Ok(done)
}
