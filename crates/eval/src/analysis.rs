use std::collections::BTreeMap;

use db::{Experiment, ExperimentResult};

use crate::Error;
use crate::item::Label;
use crate::manifest::LabelCounts;
use crate::metrics::{Scored, best_threshold, evaluate};
use crate::report::{ExperimentSummary, Repeatability, SplitLabelCounts};

/// The bot threshold when the tune split is empty.
const FALLBACK_THRESHOLD: f64 = 0.5;

/// All repeats of one eval set item.
#[derive(Debug, Clone, PartialEq)]
pub struct ItemScores {
    pub label: Label,
    pub is_tune: bool,
    pub p_bots: Vec<f64>,
}

impl ItemScores {
    fn mean(&self) -> f64 {
        self.p_bots.iter().sum::<f64>() / self.p_bots.len() as f64
    }

    fn scored(&self) -> Scored {
        Scored {
            p_bot: self.mean(),
            label: self.label,
        }
    }
}

/// Groups the results by account. An item's `p_bot` is the mean over its repeats.
pub fn group(results: &[ExperimentResult]) -> Result<Vec<ItemScores>, Error> {
    let mut by_account: BTreeMap<&str, ItemScores> = BTreeMap::new();
    for result in results {
        let label =
            Label::parse(&result.label).ok_or_else(|| Error::BadLabel(result.label.clone()))?;
        by_account
            .entry(&result.account_id)
            .or_insert_with(|| ItemScores {
                label,
                is_tune: result.split == "tune",
                p_bots: Vec::new(),
            })
            .p_bots
            .push(result.p_bot);
    }
    Ok(by_account.into_values().collect())
}

/// The sample standard deviation. Fewer than two values give 0.
pub fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let squares: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    (squares / (values.len() - 1) as f64).sqrt()
}

/// Returns `None` when no item has two results.
pub fn repeatability(items: &[ItemScores], threshold: f64, repeats: i32) -> Option<Repeatability> {
    let repeated: Vec<&ItemScores> = items.iter().filter(|i| i.p_bots.len() >= 2).collect();
    if repeated.is_empty() {
        return None;
    }
    let sds: Vec<f64> = repeated.iter().map(|i| sample_sd(&i.p_bots)).collect();
    let flips = repeated
        .iter()
        .filter(|item| {
            let answers = item.p_bots.iter().map(|p| *p >= threshold);
            answers.clone().any(|a| a) && answers.clone().any(|a| !a)
        })
        .count();
    Some(Repeatability {
        repeats,
        items: repeated.len(),
        mean_sd: sds.iter().sum::<f64>() / sds.len() as f64,
        max_sd: sds.iter().copied().fold(0.0, f64::max),
        flip_share: flips as f64 / repeated.len() as f64,
    })
}

fn count(items: &[&ItemScores]) -> LabelCounts {
    let of = |label: Label| items.iter().filter(|i| i.label == label).count();
    LabelCounts {
        human: of(Label::Human),
        bot: of(Label::Bot),
        agent: of(Label::Agent),
    }
}

/// Picks the bot threshold on the tune split, then measures the test split at that threshold.
pub fn summarize(
    experiment: &Experiment,
    results: &[ExperimentResult],
) -> Result<ExperimentSummary, Error> {
    let items = group(results)?;
    let (tune, test): (Vec<&ItemScores>, Vec<&ItemScores>) = items.iter().partition(|i| i.is_tune);
    let tune_scored: Vec<Scored> = tune.iter().map(|i| i.scored()).collect();
    let test_scored: Vec<Scored> = test.iter().map(|i| i.scored()).collect();
    let threshold = best_threshold(&tune_scored).unwrap_or(FALLBACK_THRESHOLD);
    Ok(ExperimentSummary {
        model_version: experiment.model_version.clone(),
        metadata_version: experiment.metadata_version.clone(),
        question_version: experiment.question_version.clone(),
        labels: SplitLabelCounts {
            tune: count(&tune),
            test: count(&test),
        },
        repeatability: repeatability(&items, threshold, experiment.repeats),
        metrics: evaluate(&test_scored, threshold),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(label: Label, p_bots: &[f64]) -> ItemScores {
        ItemScores {
            label,
            is_tune: false,
            p_bots: p_bots.to_vec(),
        }
    }

    #[test]
    fn sample_sd_uses_n_minus_one() {
        // Mean 0.5. Squares 0.01 and 0.01. Sum 0.02 over n-1 = 1. SD sqrt(0.02).
        assert!((sample_sd(&[0.4, 0.6]) - 0.02_f64.sqrt()).abs() < 1e-12);
        assert_eq!(sample_sd(&[0.4]), 0.0);
        assert_eq!(sample_sd(&[0.3, 0.3, 0.3]), 0.0);
    }

    #[test]
    fn repeatability_reports_mean_and_largest_sd_and_flips() {
        let items = vec![
            item(Label::Bot, &[0.4, 0.6]),   // SD sqrt(0.02), flips at 0.5
            item(Label::Human, &[0.1, 0.1]), // SD 0, no flip
            item(Label::Human, &[0.9]),      // one result, left out
        ];
        let r = repeatability(&items, 0.5, 2).unwrap();
        assert_eq!(r.items, 2);
        assert!((r.max_sd - 0.02_f64.sqrt()).abs() < 1e-12);
        assert!((r.mean_sd - 0.02_f64.sqrt() / 2.0).abs() < 1e-12);
        assert!((r.flip_share - 0.5).abs() < 1e-12);
    }

    #[test]
    fn repeatability_is_empty_without_repeats() {
        assert_eq!(repeatability(&[item(Label::Bot, &[0.5])], 0.5, 1), None);
    }

    #[test]
    fn group_averages_the_repeats_of_one_account() {
        let result = |repeat: i32, p_bot: f64| ExperimentResult {
            account_id: "1".into(),
            tweet_id: "t".into(),
            repeat,
            label: "agent".into(),
            split: "test".into(),
            is_llm_text: None,
            p_bot,
            p_llm_text: None,
            raw: serde_json::json!({}),
        };
        let items = group(&[result(0, 0.2), result(1, 0.4)]).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].label, Label::Agent);
        assert!((items[0].mean() - 0.3).abs() < 1e-12);
    }
}
