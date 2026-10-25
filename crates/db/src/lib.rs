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
    sqlx::query!("select 1 as ok").fetch_one(db).await?;
    Ok(())
}

pub async fn insert_query<'e>(db: impl PgExecutor<'e>, text: &str) -> Result<Query, Error> {
    sqlx::query_as!(
        Query,
        r#"insert into queries (id, text) values ($1, $2)
        returning id, text, status as "status: QueryStatus", error, batch_count, created_at, updated_at"#,
        Uuid::new_v4(),
        text
    )
    .fetch_one(db)
    .await
}

pub async fn fetch_query<'e>(db: impl PgExecutor<'e>, id: Uuid) -> Result<Option<Query>, Error> {
    sqlx::query_as!(
        Query,
        r#"select id, text, status as "status: QueryStatus", error, batch_count, created_at, updated_at
        from queries where id = $1"#,
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
        "update queries set status = $2, error = $3, updated_at = now()
        where id = $1 and status = any($4::query_status[])",
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
        "insert into accounts (id, handle, raw) values ($1, $2, $3)
        on conflict (id) do update
        set handle = excluded.handle, raw = excluded.raw, updated_at = now()
        where (accounts.handle, accounts.raw) is distinct from (excluded.handle, excluded.raw)",
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
        "insert into tweets (id, account_id, text, posted_at, raw) values ($1, $2, $3, $4, $5)
        on conflict (id) do update
        set account_id = excluded.account_id, text = excluded.text,
            posted_at = excluded.posted_at, raw = excluded.raw, updated_at = now()
        where (tweets.account_id, tweets.text, tweets.posted_at, tweets.raw)
            is distinct from (excluded.account_id, excluded.text, excluded.posted_at, excluded.raw)",
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
        "insert into query_tweets (query_id, tweet_id) values ($1, $2) on conflict do nothing",
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
        "insert into query_batches (query_id, batch_number) values ($1, $2) on conflict do nothing",
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
        "insert into predictions
            (tweet_id, model_version, metadata_version, question_version, p_bot, raw)
        values ($1, $2, $3, $4, $5, $6)
        on conflict (tweet_id, model_version, metadata_version, question_version) do update
        set p_bot = excluded.p_bot, raw = excluded.raw
        where (predictions.p_bot, predictions.raw) is distinct from (excluded.p_bot, excluded.raw)",
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
        r#"select t.id, t.account_id, t.text, t.posted_at, t.raw,
            a.handle, a.raw as account_raw,
            p.p_bot as "p_bot?", p.raw as "prediction_raw?"
        from query_tweets qt
        join tweets t on t.id = qt.tweet_id
        join accounts a on a.id = t.account_id
        left join predictions p on p.tweet_id = t.id
            and p.model_version = $2 and p.metadata_version = $3 and p.question_version = $4
        where qt.query_id = $1 and ($5::text is null or t.id > $5)
        order by t.id
        limit $6"#,
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
