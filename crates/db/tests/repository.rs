#![cfg(feature = "integration")]

use db::{PgPool, Versions};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Barrier;
use types::{
    Account, MetadataVersion, ModelVersion, Prediction, QueryStatus, QuestionVersion, Tweet,
};
use uuid::Uuid;

fn versions(model: &str) -> Versions {
    Versions {
        model: ModelVersion(model.into()),
        metadata: MetadataVersion("v1".into()),
        question: QuestionVersion("v1".into()),
    }
}

fn account() -> Account {
    Account {
        id: "a1".into(),
        handle: "alice".into(),
        joined_at: Some("2025-01-02T03:04:05Z".parse().unwrap()),
        followers: Some(120),
        following: Some(45),
        tweet_count: Some(900),
        is_verified: Some(true),
        is_automated: Some(false),
        raw: json!({"k": 1}),
    }
}

fn tweet(id: &str) -> Tweet {
    Tweet {
        id: id.into(),
        account_id: "a1".into(),
        text: format!("tweet {id}"),
        posted_at: None,
        raw: json!({}),
    }
}

fn prediction(tweet_id: &str, model: &str) -> Prediction {
    let v = versions(model);
    Prediction {
        tweet_id: tweet_id.into(),
        model_version: v.model,
        metadata_version: v.metadata,
        question_version: v.question,
        p_bot: 0.5,
        raw: json!({}),
    }
}

async fn seed(pool: &PgPool, tweets: usize) -> Uuid {
    let query = db::insert_query(pool, "q").await.unwrap();
    db::upsert_account(pool, &account()).await.unwrap();
    for n in 0..tweets {
        let id = format!("{n:04}");
        db::upsert_tweet(pool, &tweet(&id)).await.unwrap();
        db::upsert_query_tweet(pool, query.id, &id).await.unwrap();
        db::upsert_prediction(pool, &prediction(&id, "m1"))
            .await
            .unwrap();
    }
    query.id
}

#[sqlx::test(migrations = "./migrations")]
async fn ping_succeeds(pool: PgPool) {
    db::ping(&pool).await.unwrap();
}

#[sqlx::test(migrations = "./migrations")]
async fn insert_then_fetch_returns_a_queued_query(pool: PgPool) {
    let query = db::insert_query(&pool, "hello").await.unwrap();
    assert_eq!(query.status, QueryStatus::Queued);
    assert_eq!(db::fetch_query(&pool, query.id).await.unwrap(), Some(query));
    assert_eq!(db::fetch_query(&pool, Uuid::new_v4()).await.unwrap(), None);
}

#[sqlx::test(migrations = "./migrations")]
async fn second_upsert_with_same_input_changes_no_row(pool: PgPool) {
    let query = db::insert_query(&pool, "q").await.unwrap();
    assert_eq!(db::upsert_account(&pool, &account()).await.unwrap(), 1);
    assert_eq!(db::upsert_account(&pool, &account()).await.unwrap(), 0);
    assert_eq!(db::upsert_tweet(&pool, &tweet("1")).await.unwrap(), 1);
    assert_eq!(db::upsert_tweet(&pool, &tweet("1")).await.unwrap(), 0);
    assert_eq!(
        db::upsert_query_tweet(&pool, query.id, "1").await.unwrap(),
        1
    );
    assert_eq!(
        db::upsert_query_tweet(&pool, query.id, "1").await.unwrap(),
        0
    );
    assert_eq!(db::upsert_query_batch(&pool, query.id, 0).await.unwrap(), 1);
    assert_eq!(db::upsert_query_batch(&pool, query.id, 0).await.unwrap(), 0);
    let p = prediction("1", "m1");
    assert_eq!(db::upsert_prediction(&pool, &p).await.unwrap(), 1);
    assert_eq!(db::upsert_prediction(&pool, &p).await.unwrap(), 0);
}

#[sqlx::test(migrations = "./migrations")]
async fn upsert_with_changed_content_changes_the_row(pool: PgPool) {
    db::upsert_account(&pool, &account()).await.unwrap();
    let renamed = Account {
        handle: "alicia".into(),
        ..account()
    };
    assert_eq!(db::upsert_account(&pool, &renamed).await.unwrap(), 1);
}

#[sqlx::test(migrations = "./migrations")]
async fn upsert_with_a_changed_account_field_changes_the_row(pool: PgPool) {
    db::upsert_account(&pool, &account()).await.unwrap();
    let grown = Account {
        followers: Some(121),
        ..account()
    };
    assert_eq!(db::upsert_account(&pool, &grown).await.unwrap(), 1);
    assert_eq!(db::upsert_account(&pool, &grown).await.unwrap(), 0);
}

#[sqlx::test(migrations = "./migrations")]
async fn account_fields_round_trip_through_tweets_with_accounts(pool: PgPool) {
    db::upsert_account(&pool, &account()).await.unwrap();
    db::upsert_tweet(&pool, &tweet("0001")).await.unwrap();
    let rows = db::tweets_with_accounts(&pool, &["0001".to_string()])
        .await
        .unwrap();
    assert_eq!(rows[0].1, account());
}

#[sqlx::test(migrations = "./migrations")]
async fn status_changes_follow_the_table(pool: PgPool) {
    use QueryStatus::*;
    let all = [Queued, Scraping, Classifying, Done, Failed];
    let allowed = |from: QueryStatus, to: QueryStatus| {
        matches!(
            (from, to),
            (Queued | Scraping, Scraping)
                | (Scraping, Classifying)
                | (Scraping | Classifying, Done)
                | (Queued | Scraping | Classifying, Failed)
        )
    };
    for from in all {
        for to in all {
            let query = db::insert_query(&pool, "q").await.unwrap();
            let path = match from {
                Queued => vec![],
                Scraping => vec![Scraping],
                Classifying => vec![Scraping, Classifying],
                Done => vec![Scraping, Done],
                Failed => vec![Failed],
            };
            for step in path {
                assert!(db::set_status(&pool, query.id, step, None).await.unwrap());
            }
            let applied = db::set_status(&pool, query.id, to, None).await.unwrap();
            let status = db::fetch_query(&pool, query.id)
                .await
                .unwrap()
                .unwrap()
                .status;
            assert_eq!(applied, allowed(from, to), "{from:?} -> {to:?}");
            assert_eq!(
                status,
                if applied { to } else { from },
                "{from:?} -> {to:?}"
            );
        }
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn failed_records_the_error(pool: PgPool) {
    let query = db::insert_query(&pool, "q").await.unwrap();
    assert!(
        db::set_status(&pool, query.id, QueryStatus::Failed, Some("boom"))
            .await
            .unwrap()
    );
    let fetched = db::fetch_query(&pool, query.id).await.unwrap().unwrap();
    assert_eq!(fetched.error.as_deref(), Some("boom"));
}

#[sqlx::test(migrations = "./migrations")]
async fn result_page_pages_by_tweet_id(pool: PgPool) {
    let id = seed(&pool, 120).await;
    let v = versions("m1");
    let mut sizes = vec![];
    let mut after: Option<String> = None;
    loop {
        let page = db::result_page(&pool, id, &v, after.as_deref(), 50)
            .await
            .unwrap();
        let Some(last) = page.last() else { break };
        assert!(page.iter().all(|row| row.prediction.is_some()));
        after = Some(last.tweet.id.clone());
        sizes.push(page.len());
    }
    assert_eq!(sizes, [50, 50, 20]);
}

#[sqlx::test(migrations = "./migrations")]
async fn result_page_returns_only_the_requested_versions(pool: PgPool) {
    let id = seed(&pool, 1).await;
    db::upsert_prediction(
        &pool,
        &Prediction {
            p_bot: 0.9,
            ..prediction("0000", "m2")
        },
    )
    .await
    .unwrap();
    let page = db::result_page(&pool, id, &versions("m2"), None, 50)
        .await
        .unwrap();
    assert_eq!(page.len(), 1);
    assert_eq!(page[0].prediction.as_ref().unwrap().p_bot, 0.9);
    let page = db::result_page(&pool, id, &versions("m3"), None, 50)
        .await
        .unwrap();
    assert_eq!(page.len(), 1);
    assert!(page[0].prediction.is_none());
}

/// Seeds a query with `tweets` tweets, no predictions, and the status `scraping`.
async fn seed_scraping(pool: &PgPool, tweets: usize) -> Uuid {
    let query = db::insert_query(pool, "q").await.unwrap();
    db::upsert_account(pool, &account()).await.unwrap();
    for n in 0..tweets {
        let id = format!("{n:04}");
        db::upsert_tweet(pool, &tweet(&id)).await.unwrap();
        db::upsert_query_tweet(pool, query.id, &id).await.unwrap();
    }
    db::set_status(pool, query.id, QueryStatus::Scraping, None)
        .await
        .unwrap();
    query.id
}

async fn finish(pool: &PgPool, id: Uuid, batch_number: i32) -> bool {
    db::finish_batch(pool, id, batch_number, &[prediction("0000", "m1")])
        .await
        .unwrap()
}

async fn batch_rows(pool: &PgPool, id: Uuid) -> i64 {
    sqlx::query_scalar("SELECT COUNT(*) FROM query_batches WHERE query_id = $1")
        .bind(id)
        .fetch_one(pool)
        .await
        .unwrap()
}

#[sqlx::test(migrations = "./migrations")]
async fn set_batch_count_moves_scraping_to_classifying(pool: PgPool) {
    let id = seed_scraping(&pool, 1).await;
    assert!(db::set_batch_count(&pool, id, 3).await.unwrap());
    let query = db::fetch_query(&pool, id).await.unwrap().unwrap();
    assert_eq!(query.status, QueryStatus::Classifying);
    assert_eq!(query.batch_count, Some(3));
}

#[sqlx::test(migrations = "./migrations")]
async fn set_batch_count_refuses_a_different_count(pool: PgPool) {
    let id = seed_scraping(&pool, 1).await;
    sqlx::query("UPDATE queries SET batch_count = 3 WHERE id = $1")
        .bind(id)
        .execute(&pool)
        .await
        .unwrap();
    assert!(!db::set_batch_count(&pool, id, 4).await.unwrap());
    let query = db::fetch_query(&pool, id).await.unwrap().unwrap();
    assert_eq!(query.status, QueryStatus::Scraping);
    assert_eq!(query.batch_count, Some(3));
    assert!(db::set_batch_count(&pool, id, 3).await.unwrap());
}

#[sqlx::test(migrations = "./migrations")]
async fn set_batch_count_refuses_other_statuses(pool: PgPool) {
    let id = seed_scraping(&pool, 1).await;
    assert!(db::set_batch_count(&pool, id, 3).await.unwrap());
    assert!(!db::set_batch_count(&pool, id, 3).await.unwrap());
    let queued = db::insert_query(&pool, "q").await.unwrap();
    assert!(!db::set_batch_count(&pool, queued.id, 3).await.unwrap());
}

#[sqlx::test(migrations = "./migrations")]
async fn query_tweet_ids_are_sorted_and_scoped_to_the_query(pool: PgPool) {
    let id = seed_scraping(&pool, 3).await;
    let other = db::insert_query(&pool, "other").await.unwrap();
    db::upsert_tweet(&pool, &tweet("9999")).await.unwrap();
    db::upsert_query_tweet(&pool, other.id, "9999")
        .await
        .unwrap();
    db::upsert_query_tweet(&pool, id, "0000").await.unwrap();
    let ids = db::query_tweet_ids(&pool, id).await.unwrap();
    assert_eq!(ids, ["0000", "0001", "0002"]);
}

#[sqlx::test(migrations = "./migrations")]
async fn tweets_with_accounts_returns_the_requested_tweets(pool: PgPool) {
    seed_scraping(&pool, 3).await;
    let ids = vec!["0002".to_string(), "0000".to_string()];
    let rows = db::tweets_with_accounts(&pool, &ids).await.unwrap();
    assert_eq!(
        rows,
        [(tweet("0000"), account()), (tweet("0002"), account())]
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn query_batch_exists_matches_one_batch(pool: PgPool) {
    let id = seed_scraping(&pool, 1).await;
    assert!(!db::query_batch_exists(&pool, id, 0).await.unwrap());
    db::upsert_query_batch(&pool, id, 0).await.unwrap();
    assert!(db::query_batch_exists(&pool, id, 0).await.unwrap());
    assert!(!db::query_batch_exists(&pool, id, 1).await.unwrap());
    assert!(
        !db::query_batch_exists(&pool, Uuid::new_v4(), 0)
            .await
            .unwrap()
    );
}

#[sqlx::test(migrations = "./migrations")]
async fn tweet_ids_without_prediction_filters_by_versions(pool: PgPool) {
    seed_scraping(&pool, 3).await;
    db::upsert_prediction(&pool, &prediction("0001", "m1"))
        .await
        .unwrap();
    let ids: Vec<String> = ["0002", "0001", "0000"].map(String::from).into();
    let missing = db::tweet_ids_without_prediction(&pool, &ids, &versions("m1"))
        .await
        .unwrap();
    assert_eq!(missing, ["0002", "0000"]);
    let missing = db::tweet_ids_without_prediction(&pool, &ids, &versions("m2"))
        .await
        .unwrap();
    assert_eq!(missing, ["0002", "0001", "0000"]);
}

#[sqlx::test(migrations = "./migrations")]
async fn finish_batch_writes_predictions_and_the_batch(pool: PgPool) {
    let id = seed_scraping(&pool, 1).await;
    assert!(!finish(&pool, id, 0).await);
    assert!(db::query_batch_exists(&pool, id, 0).await.unwrap());
    let ids = vec!["0000".to_string()];
    let missing = db::tweet_ids_without_prediction(&pool, &ids, &versions("m1"))
        .await
        .unwrap();
    assert!(missing.is_empty());
}

#[sqlx::test(migrations = "./migrations")]
async fn finishing_the_same_batch_twice_leaves_one_row(pool: PgPool) {
    let id = seed_scraping(&pool, 1).await;
    assert!(db::set_batch_count(&pool, id, 2).await.unwrap());
    assert!(!finish(&pool, id, 0).await);
    assert!(!finish(&pool, id, 0).await);
    assert_eq!(batch_rows(&pool, id).await, 1);
    let query = db::fetch_query(&pool, id).await.unwrap().unwrap();
    assert_eq!(query.status, QueryStatus::Classifying);
}

#[sqlx::test(migrations = "./migrations")]
async fn two_tasks_finishing_the_last_batches_at_once_set_done(pool: PgPool) {
    let id = seed_scraping(&pool, 1).await;
    assert!(db::set_batch_count(&pool, id, 2).await.unwrap());
    let barrier = Arc::new(Barrier::new(2));
    let tasks: Vec<_> = (0..2)
        .map(|batch_number| {
            let (pool, barrier) = (pool.clone(), barrier.clone());
            tokio::spawn(async move {
                barrier.wait().await;
                finish(&pool, id, batch_number).await
            })
        })
        .collect();
    let mut set_done = 0;
    for task in tasks {
        set_done += usize::from(task.await.unwrap());
    }
    assert_eq!(set_done, 1);
    assert_eq!(batch_rows(&pool, id).await, 2);
    let query = db::fetch_query(&pool, id).await.unwrap().unwrap();
    assert_eq!(query.status, QueryStatus::Done);
}

#[sqlx::test(migrations = "./migrations")]
async fn finishing_every_batch_before_the_batch_count_leaves_the_status(pool: PgPool) {
    let id = seed_scraping(&pool, 1).await;
    assert!(!finish(&pool, id, 0).await);
    assert!(!finish(&pool, id, 1).await);
    let query = db::fetch_query(&pool, id).await.unwrap().unwrap();
    assert_eq!(query.status, QueryStatus::Scraping);
    assert_eq!(batch_rows(&pool, id).await, 2);
}
