use std::time::Duration;

use chrono::{DateTime, Utc};
use serde_json::Value;
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

#[derive(Clone, Debug)]
pub struct ApiKey {
    pub id: Uuid,
    pub name: String,
    pub query_quota_per_hour: i32,
    pub created_at: DateTime<Utc>,
    pub revoked_at: Option<DateTime<Utc>>,
}

/// Stores the SHA-256 hash of a key. The table never holds the key.
pub async fn insert_api_key<'e>(
    db: impl PgExecutor<'e>,
    name: &str,
    key_sha256: &[u8],
    query_quota_per_hour: i32,
) -> Result<ApiKey, Error> {
    sqlx::query_as!(
        ApiKey,
        "INSERT INTO api_keys (id, name, key_sha256, query_quota_per_hour)
        VALUES ($1, $2, $3, $4)
        RETURNING id, name, query_quota_per_hour, created_at, revoked_at",
        Uuid::new_v4(),
        name,
        key_sha256,
        query_quota_per_hour
    )
    .fetch_one(db)
    .await
}

/// Returns whether Postgres revoked the key. A key that is unknown or already revoked returns
/// `false`.
pub async fn revoke_api_key<'e>(db: impl PgExecutor<'e>, id: Uuid) -> Result<bool, Error> {
    let result = sqlx::query!(
        "UPDATE api_keys SET revoked_at = NOW() WHERE id = $1 AND revoked_at IS NULL",
        id
    )
    .execute(db)
    .await?;
    Ok(result.rows_affected() == 1)
}

/// Returns the key with this hash, or `None` when the hash is unknown or the key is revoked.
pub async fn find_active_api_key<'e>(
    db: impl PgExecutor<'e>,
    key_sha256: &[u8],
) -> Result<Option<ApiKey>, Error> {
    sqlx::query_as!(
        ApiKey,
        "SELECT id, name, query_quota_per_hour, created_at, revoked_at
        FROM api_keys WHERE key_sha256 = $1 AND revoked_at IS NULL",
        key_sha256
    )
    .fetch_optional(db)
    .await
}

pub enum CreatedQuery {
    New(Query),
    Reused(Query),
    /// The key has used its query quota. The oldest counted query leaves the hour after
    /// `retry_after`.
    OverQuota {
        retry_after: Duration,
    },
}

/// Returns the newest query younger than `reuse_window` with the same lower-cased text that has
/// not failed. Otherwise inserts a new query for `api_key`. A zero `reuse_window` always inserts.
///
/// With `quota_enabled`, one transaction takes the advisory lock of the key, then the advisory
/// lock of the text, counts the key's queries of the past hour, and inserts. The key lock puts
/// parallel requests of one key in a line, so none passes the quota. The text lock gives two
/// identical queries sent at once one query. A reused query does not count.
pub async fn create_or_reuse_query(
    pool: &PgPool,
    text: &str,
    reuse_window: Duration,
    api_key: &ApiKey,
    quota_enabled: bool,
) -> Result<CreatedQuery, Error> {
    if reuse_window.is_zero() && !quota_enabled {
        return insert_query_for_key(pool, text, api_key.id)
            .await
            .map(CreatedQuery::New);
    }
    let mut tx = pool.begin().await?;
    if quota_enabled {
        sqlx::query!(
            r#"SELECT 1 AS "locked!" FROM PG_ADVISORY_XACT_LOCK(HASHTEXTEXTENDED($1::UUID::TEXT, 0))"#,
            api_key.id
        )
        .fetch_one(&mut *tx)
        .await?;
    }
    sqlx::query!(
        r#"SELECT 1 AS "locked!" FROM PG_ADVISORY_XACT_LOCK(HASHTEXTEXTENDED(LOWER($1), 0))"#,
        text
    )
    .fetch_one(&mut *tx)
    .await?;
    if !reuse_window.is_zero() {
        let earlier = sqlx::query_as!(
            Query,
            r#"SELECT id, text, status AS "status: QueryStatus", error, batch_count, created_at, updated_at
            FROM queries
            WHERE LOWER(text) = LOWER($1)
                AND status <> 'failed'
                AND created_at > NOW() - MAKE_INTERVAL(secs => $2)
            ORDER BY created_at DESC
            LIMIT 1"#,
            text,
            reuse_window.as_secs_f64()
        )
        .fetch_optional(&mut *tx)
        .await?;
        if let Some(query) = earlier {
            tx.commit().await?;
            return Ok(CreatedQuery::Reused(query));
        }
    }
    if quota_enabled {
        let used = sqlx::query!(
            r#"SELECT COUNT(*) AS "count!",
                EXTRACT(EPOCH FROM (MIN(created_at) + INTERVAL '1 hour' - NOW()))::DOUBLE PRECISION AS "retry_secs: f64"
            FROM queries
            WHERE api_key_id = $1 AND created_at > NOW() - INTERVAL '1 hour'"#,
            api_key.id
        )
        .fetch_one(&mut *tx)
        .await?;
        if used.count >= i64::from(api_key.query_quota_per_hour) {
            let retry_secs = used.retry_secs.unwrap_or(3600.0).max(0.0);
            return Ok(CreatedQuery::OverQuota {
                retry_after: Duration::from_secs_f64(retry_secs),
            });
        }
    }
    let query = insert_query_for_key(&mut *tx, text, api_key.id).await?;
    tx.commit().await?;
    Ok(CreatedQuery::New(query))
}

async fn insert_query_for_key<'e>(
    db: impl PgExecutor<'e>,
    text: &str,
    api_key_id: Uuid,
) -> Result<Query, Error> {
    sqlx::query_as!(
        Query,
        r#"INSERT INTO queries (id, text, api_key_id) VALUES ($1, $2, $3)
        RETURNING id, text, status AS "status: QueryStatus", error, batch_count, created_at, updated_at"#,
        Uuid::new_v4(),
        text,
        api_key_id
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

/// Returns how many batches of the query the classifier has finished.
pub async fn batches_done<'e>(db: impl PgExecutor<'e>, query_id: Uuid) -> Result<i64, Error> {
    sqlx::query_scalar!(
        r#"SELECT COUNT(*) AS "count!" FROM query_batches WHERE query_id = $1"#,
        query_id
    )
    .fetch_one(db)
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
            (tweet_id, model_version, metadata_version, question_version, p_bot, p_llm_text, raw)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (tweet_id, model_version, metadata_version, question_version) DO UPDATE
        SET p_bot = EXCLUDED.p_bot, p_llm_text = EXCLUDED.p_llm_text, raw = EXCLUDED.raw
        WHERE (predictions.p_bot, predictions.p_llm_text, predictions.raw)
            IS DISTINCT FROM (EXCLUDED.p_bot, EXCLUDED.p_llm_text, EXCLUDED.raw)",
        prediction.tweet_id,
        &prediction.model_version as &ModelVersion,
        &prediction.metadata_version as &MetadataVersion,
        &prediction.question_version as &QuestionVersion,
        prediction.p_bot,
        prediction.p_llm_text,
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
            p.p_bot AS "p_bot?", p.p_llm_text, p.raw AS "prediction_raw?"
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
                    p_llm_text: row.p_llm_text,
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

pub struct NewExperiment {
    pub model_version: ModelVersion,
    pub metadata_version: MetadataVersion,
    pub question_version: QuestionVersion,
    pub eval_set_sha256: String,
    pub seed: i64,
    pub repeats: i32,
}

pub struct Experiment {
    pub id: Uuid,
    pub model_version: ModelVersion,
    pub metadata_version: MetadataVersion,
    pub question_version: QuestionVersion,
    pub eval_set_sha256: String,
    pub seed: i64,
    pub repeats: i32,
    pub created_at: DateTime<Utc>,
}

/// `label` holds `human`, `bot`, or `agent`. `split` holds `tune` or `test`. Postgres rejects any other value.
pub struct ExperimentResult {
    pub account_id: String,
    pub tweet_id: String,
    pub repeat: i32,
    pub label: String,
    pub split: String,
    pub is_llm_text: Option<bool>,
    pub p_bot: f64,
    pub p_llm_text: Option<f64>,
    pub raw: Value,
}

pub async fn insert_experiment<'e>(
    db: impl PgExecutor<'e>,
    experiment: &NewExperiment,
) -> Result<Experiment, Error> {
    sqlx::query_as!(
        Experiment,
        r#"INSERT INTO experiments
            (id, model_version, metadata_version, question_version, eval_set_sha256, seed, repeats)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id, model_version AS "model_version: ModelVersion",
            metadata_version AS "metadata_version: MetadataVersion",
            question_version AS "question_version: QuestionVersion",
            eval_set_sha256, seed, repeats, created_at"#,
        Uuid::new_v4(),
        &experiment.model_version as &ModelVersion,
        &experiment.metadata_version as &MetadataVersion,
        &experiment.question_version as &QuestionVersion,
        experiment.eval_set_sha256,
        experiment.seed,
        experiment.repeats
    )
    .fetch_one(db)
    .await
}

/// Inserts all `results` in one statement. A repeated `(account_id, repeat)` pair for the same
/// experiment fails the primary key.
pub async fn insert_experiment_results<'e>(
    db: impl PgExecutor<'e>,
    experiment_id: Uuid,
    results: &[ExperimentResult],
) -> Result<u64, Error> {
    let account_ids: Vec<&str> = results.iter().map(|r| r.account_id.as_str()).collect();
    let tweet_ids: Vec<&str> = results.iter().map(|r| r.tweet_id.as_str()).collect();
    let repeats: Vec<i32> = results.iter().map(|r| r.repeat).collect();
    let labels: Vec<&str> = results.iter().map(|r| r.label.as_str()).collect();
    let splits: Vec<&str> = results.iter().map(|r| r.split.as_str()).collect();
    let is_llm_texts: Vec<Option<bool>> = results.iter().map(|r| r.is_llm_text).collect();
    let p_bots: Vec<f64> = results.iter().map(|r| r.p_bot).collect();
    let p_llm_texts: Vec<Option<f64>> = results.iter().map(|r| r.p_llm_text).collect();
    let raws: Vec<&Value> = results.iter().map(|r| &r.raw).collect();
    let result = sqlx::query!(
        "INSERT INTO experiment_results
            (experiment_id, account_id, tweet_id, repeat, label, split, is_llm_text, p_bot,
            p_llm_text, raw)
        SELECT $1, * FROM UNNEST($2::TEXT[], $3::TEXT[], $4::INTEGER[], $5::TEXT[], $6::TEXT[],
            $7::BOOLEAN[], $8::DOUBLE PRECISION[], $9::DOUBLE PRECISION[], $10::JSONB[])",
        experiment_id,
        &account_ids as &[&str],
        &tweet_ids as &[&str],
        &repeats,
        &labels as &[&str],
        &splits as &[&str],
        &is_llm_texts as &[Option<bool>],
        &p_bots,
        &p_llm_texts as &[Option<f64>],
        &raws as &[&Value]
    )
    .execute(db)
    .await?;
    Ok(result.rows_affected())
}

pub async fn fetch_experiment<'e>(
    db: impl PgExecutor<'e>,
    id: Uuid,
) -> Result<Option<Experiment>, Error> {
    sqlx::query_as!(
        Experiment,
        r#"SELECT id, model_version AS "model_version: ModelVersion",
            metadata_version AS "metadata_version: MetadataVersion",
            question_version AS "question_version: QuestionVersion",
            eval_set_sha256, seed, repeats, created_at
        FROM experiments WHERE id = $1"#,
        id
    )
    .fetch_optional(db)
    .await
}

/// Returns all experiments, oldest first.
pub async fn list_experiments<'e>(db: impl PgExecutor<'e>) -> Result<Vec<Experiment>, Error> {
    sqlx::query_as!(
        Experiment,
        r#"SELECT id, model_version AS "model_version: ModelVersion",
            metadata_version AS "metadata_version: MetadataVersion",
            question_version AS "question_version: QuestionVersion",
            eval_set_sha256, seed, repeats, created_at
        FROM experiments ORDER BY created_at, id"#
    )
    .fetch_all(db)
    .await
}

/// Returns the results of one experiment, sorted by account ID and repeat.
pub async fn experiment_results<'e>(
    db: impl PgExecutor<'e>,
    experiment_id: Uuid,
) -> Result<Vec<ExperimentResult>, Error> {
    sqlx::query_as!(
        ExperimentResult,
        "SELECT account_id, tweet_id, repeat, label, split, is_llm_text, p_bot, p_llm_text, raw
        FROM experiment_results WHERE experiment_id = $1
        ORDER BY account_id, repeat",
        experiment_id
    )
    .fetch_all(db)
    .await
}
