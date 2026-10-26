use classifier::Classifier;
use db::{ExperimentResult, NewExperiment, PgPool};
use types::{MetadataVersion, ModelInput, QuestionVersion};
use uuid::Uuid;

use crate::Error;
use crate::analysis::summarize;
use crate::item::EvalItem;
use crate::report::{ExperimentSummary, render};
use crate::split::{Split, assign};

/// The seed of every split. The experiments row records it.
pub const SPLIT_SEED: u64 = 20_261_030;

/// Model inputs in one backend call and one insert.
const CHUNK: usize = 20;

/// Workers AI price of Clef-flash input tokens, in US dollars per million tokens.
/// Source: <https://developers.cloudflare.com/workers-ai/platform/pricing/>, read on 2026-10-26.
/// The page lists no output price.
pub const USD_PER_MILLION_INPUT_TOKENS: f64 = 0.038;

#[derive(Debug, Clone, PartialEq)]
pub struct CostEstimate {
    pub input_tokens: usize,
    pub usd: f64,
}

/// One token is about four characters of the JSON of `state` and `questions`.
pub fn estimate_cost(inputs: &[ModelInput], repeats: u32) -> CostEstimate {
    let chars: usize = inputs
        .iter()
        .map(|input| {
            let state = serde_json::to_string(&input.state).map_or(0, |s| s.len());
            let questions = serde_json::to_string(&input.questions).map_or(0, |s| s.len());
            state + questions
        })
        .sum();
    let input_tokens = chars / 4 * repeats as usize;
    CostEstimate {
        input_tokens,
        usd: input_tokens as f64 / 1_000_000.0 * USD_PER_MILLION_INPUT_TOKENS,
    }
}

/// Sorts items by account ID, so every run sees the same order. `sample` keeps the first n.
pub fn order(mut items: Vec<EvalItem>, sample: Option<usize>) -> Vec<EvalItem> {
    items.sort_by(|a, b| a.account.id.cmp(&b.account.id));
    if let Some(n) = sample {
        items.truncate(n);
    }
    items
}

pub fn build_inputs(
    items: &[EvalItem],
    metadata_version: &MetadataVersion,
    question_version: &QuestionVersion,
) -> Result<Vec<ModelInput>, Error> {
    items
        .iter()
        .map(|item| {
            Ok(metadata::build(
                &item.tweet,
                &item.account,
                &item.recent_tweets,
                metadata_version,
                question_version,
            )?)
        })
        .collect()
}

pub struct Params<'a> {
    pub items: &'a [EvalItem],
    pub metadata_version: MetadataVersion,
    pub question_version: QuestionVersion,
    pub repeats: u32,
    pub eval_set_sha256: String,
}

/// Inserts the experiments row, then classifies every item `repeats` times.
/// Inserts the results of each chunk as they arrive, so a failed run keeps them.
pub async fn run(
    pool: &PgPool,
    classifier: &impl Classifier,
    params: &Params<'_>,
) -> Result<Uuid, Error> {
    if params.items.is_empty() {
        return Err(Error::EmptyEvalSet);
    }
    let inputs = build_inputs(
        params.items,
        &params.metadata_version,
        &params.question_version,
    )?;
    let experiment = db::insert_experiment(
        pool,
        &NewExperiment {
            model_version: classifier.model_version(),
            metadata_version: params.metadata_version.clone(),
            question_version: params.question_version.clone(),
            eval_set_sha256: params.eval_set_sha256.clone(),
            seed: SPLIT_SEED as i64,
            repeats: params.repeats as i32,
        },
    )
    .await?;

    for repeat in 0..params.repeats {
        let mut done = 0;
        for (item_chunk, input_chunk) in params.items.chunks(CHUNK).zip(inputs.chunks(CHUNK)) {
            let predictions = classifier.classify(input_chunk).await?;
            if predictions.len() != input_chunk.len() {
                return Err(Error::PredictionCount {
                    sent: input_chunk.len(),
                    got: predictions.len(),
                });
            }
            let results: Vec<ExperimentResult> = item_chunk
                .iter()
                .zip(predictions)
                .map(|(item, prediction)| ExperimentResult {
                    account_id: item.account.id.clone(),
                    tweet_id: item.tweet.id.clone(),
                    repeat: repeat as i32,
                    label: item.label.as_str().into(),
                    split: match assign(SPLIT_SEED, &item.account.id) {
                        Split::Tune => "tune".into(),
                        Split::Test => "test".into(),
                    },
                    is_llm_text: item.is_llm_text,
                    p_bot: prediction.p_bot,
                    p_llm_text: prediction.p_llm_text,
                    raw: prediction.raw,
                })
                .collect();
            db::insert_experiment_results(pool, experiment.id, &results).await?;
            done += item_chunk.len();
            eprintln!(
                "repeat {} of {}: {done} of {} items",
                repeat + 1,
                params.repeats,
                params.items.len()
            );
        }
    }
    Ok(experiment.id)
}

/// Reads one stored experiment and summarizes it.
pub async fn load_summary(pool: &PgPool, id: Uuid) -> Result<ExperimentSummary, Error> {
    let experiment = db::fetch_experiment(pool, id)
        .await?
        .ok_or(Error::NoSuchExperiment(id))?;
    let results = db::experiment_results(pool, id).await?;
    summarize(&experiment, &results)
}

/// Renders the eval report for `ids`, or for every stored experiment when `ids` is empty.
pub async fn report_markdown(pool: &PgPool, ids: &[Uuid]) -> Result<String, Error> {
    let ids: Vec<Uuid> = if ids.is_empty() {
        db::list_experiments(pool)
            .await?
            .into_iter()
            .map(|e| e.id)
            .collect()
    } else {
        ids.to_vec()
    };
    let mut summaries = Vec::new();
    for id in ids {
        summaries.push(load_summary(pool, id).await?);
    }
    Ok(format!(
        "# Eval report\n\nSplit seed: {SPLIT_SEED}. The threshold comes from the tune split. Every metric uses the test split.\n\n{}",
        render(&summaries)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use types::Question;

    fn input(state: Value) -> ModelInput {
        ModelInput {
            tweet_id: "1".into(),
            metadata_version: MetadataVersion("v1".into()),
            question_version: QuestionVersion("v1".into()),
            state,
            questions: vec![Question {
                id: "bot".into(),
                instructions: "ab".into(),
            }],
        }
    }

    #[test]
    fn cost_counts_one_token_for_four_characters_of_state_and_questions() {
        // state "\"abcdefgh\"" is 10 characters. questions JSON is
        // [{"id":"bot","instructions":"ab"}] = 34 characters. 44 / 4 = 11 tokens.
        let estimate = estimate_cost(&[input(json!("abcdefgh"))], 1);
        assert_eq!(estimate.input_tokens, 11);
        assert!((estimate.usd - 11.0 * 0.038 / 1_000_000.0).abs() < 1e-15);
    }

    #[test]
    fn order_sorts_by_account_id_and_sample_keeps_the_first() {
        let mut items = crate::item::parse_jsonl::<EvalItem>(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/fixtures/eval-set.jsonl"
            ))
            .unwrap(),
        )
        .unwrap();
        items.reverse();
        let ids = |items: &[EvalItem]| {
            items
                .iter()
                .map(|i| i.account.id.clone())
                .collect::<Vec<_>>()
        };
        let all = ids(&order(items.clone(), None));
        let mut sorted = all.clone();
        sorted.sort();
        assert_eq!(all, sorted);
        assert_eq!(ids(&order(items, Some(3))), sorted[..3].to_vec());
    }

    #[test]
    fn cost_scales_with_repeats_and_items() {
        let one = estimate_cost(&[input(json!("abcdefgh"))], 1);
        let many = estimate_cost(&[input(json!("abcdefgh")), input(json!("abcdefgh"))], 5);
        assert_eq!(many.input_tokens, one.input_tokens * 10);
    }
}
