#![cfg(feature = "integration")]

use db::{PgPool, Versions};
use serde_json::json;
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
