#![cfg(feature = "integration")]

use std::path::Path;

use classifier::Fake;
use eval::experiment::{Params, load_summary, order, report_markdown, run};
use sqlx::postgres::PgPoolOptions;
use types::{MetadataVersion, QuestionVersion};

#[tokio::test]
async fn experiment_then_report_on_the_fixture_with_the_fake() {
    let pool = PgPoolOptions::new()
        .connect(&std::env::var("DATABASE_URL").expect("DATABASE_URL is not set"))
        .await
        .unwrap();
    let set = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/eval-set.jsonl");
    let manifest = eval::manifest::from_file(&set).unwrap();
    let items = order(eval::item::read_jsonl(&set).unwrap(), None);

    let id = run(
        &pool,
        &Fake::by_tweet_id(0),
        &Params {
            items: &items,
            metadata_version: MetadataVersion("v5".into()),
            question_version: QuestionVersion("v4".into()),
            repeats: 2,
            eval_set_sha256: manifest.sha256.clone(),
        },
    )
    .await
    .unwrap();

    let experiment = db::fetch_experiment(&pool, id).await.unwrap().unwrap();
    assert_eq!(experiment.repeats, 2);
    assert_eq!(experiment.seed, eval::experiment::SPLIT_SEED as i64);
    assert_eq!(experiment.eval_set_sha256, manifest.sha256);
    let results = db::experiment_results(&pool, id).await.unwrap();
    assert_eq!(results.len(), 20);
    assert!(
        results
            .iter()
            .all(|r| r.split == "tune" || r.split == "test")
    );

    let summary = load_summary(&pool, id).await.unwrap();
    let split_total = |c: &eval::manifest::LabelCounts| c.human + c.bot + c.agent;
    assert_eq!(
        split_total(&summary.labels.tune) + split_total(&summary.labels.test),
        10
    );
    let repeatability = summary.repeatability.unwrap();
    assert_eq!(repeatability.items, 10);
    // The fake gives one tweet the same bot probability on every repeat.
    assert_eq!(repeatability.max_sd, 0.0);

    let markdown = report_markdown(&pool, &[id]).await.unwrap();
    for section in [
        "## Comparison",
        "## Label counts",
        "## Repeatability",
        "## Calibration curve",
        "```mermaid",
    ] {
        assert!(markdown.contains(section), "report lacks {section}");
    }

    sqlx::query("DELETE FROM experiment_results WHERE experiment_id = $1")
        .bind(id)
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("DELETE FROM experiments WHERE id = $1")
        .bind(id)
        .execute(&pool)
        .await
        .unwrap();
}
