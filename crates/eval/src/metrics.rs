use serde::{Deserialize, Serialize};

use crate::item::Label;

/// One eval set item as the metrics see it: the bot probability and the label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scored {
    pub p_bot: f64,
    pub label: Label,
}

impl Scored {
    pub fn is_bot(&self) -> bool {
        self.label.is_bot()
    }
}

pub const BIN_COUNT: usize = 10;

/// One reliability bin. A bin covers `[lower, upper)`. The last bin also covers 1.0.
/// An empty bin has `count` 0 and zeros in the two means.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bin {
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    pub mean_p_bot: f64,
    pub bot_share: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    pub threshold: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    /// Empty when the slice holds only one class.
    pub roc_auc: Option<f64>,
    pub brier: f64,
    pub ece: f64,
    /// Empty when the slice holds no item with that label.
    pub recall_bot: Option<f64>,
    pub recall_agent: Option<f64>,
    pub bins: Vec<Bin>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Confusion {
    tp: usize,
    fp: usize,
    tn: usize,
    fn_: usize,
}

/// An item counts as predicted bot when `p_bot >= threshold`.
fn confusion(samples: &[Scored], threshold: f64) -> Confusion {
    let mut c = Confusion::default();
    for s in samples {
        match (s.p_bot >= threshold, s.is_bot()) {
            (true, true) => c.tp += 1,
            (true, false) => c.fp += 1,
            (false, false) => c.tn += 1,
            (false, true) => c.fn_ += 1,
        }
    }
    c
}

/// Division that gives 0 for an empty denominator.
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn f1_of(c: Confusion) -> f64 {
    let precision = ratio(c.tp, c.tp + c.fp);
    let recall = ratio(c.tp, c.tp + c.fn_);
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

pub fn accuracy(samples: &[Scored], threshold: f64) -> f64 {
    let c = confusion(samples, threshold);
    ratio(c.tp + c.tn, samples.len())
}

pub fn precision(samples: &[Scored], threshold: f64) -> f64 {
    let c = confusion(samples, threshold);
    ratio(c.tp, c.tp + c.fp)
}

pub fn recall(samples: &[Scored], threshold: f64) -> f64 {
    let c = confusion(samples, threshold);
    ratio(c.tp, c.tp + c.fn_)
}

pub fn f1(samples: &[Scored], threshold: f64) -> f64 {
    f1_of(confusion(samples, threshold))
}

/// The share of items with `label` that have `p_bot >= threshold`.
/// Returns `None` when the slice holds no item with `label`.
pub fn recall_for(samples: &[Scored], label: Label, threshold: f64) -> Option<f64> {
    let of_label: Vec<_> = samples.iter().filter(|s| s.label == label).collect();
    if of_label.is_empty() {
        return None;
    }
    let hits = of_label.iter().filter(|s| s.p_bot >= threshold).count();
    Some(ratio(hits, of_label.len()))
}

/// Rank-based ROC-AUC: the share of (bot, human) pairs where the bot has the higher
/// bot probability. A tie counts half. Returns `None` when one class is missing.
pub fn roc_auc(samples: &[Scored]) -> Option<f64> {
    let bots: Vec<f64> = samples
        .iter()
        .filter(|s| s.is_bot())
        .map(|s| s.p_bot)
        .collect();
    let humans: Vec<f64> = samples
        .iter()
        .filter(|s| !s.is_bot())
        .map(|s| s.p_bot)
        .collect();
    if bots.is_empty() || humans.is_empty() {
        return None;
    }
    let mut wins = 0.0;
    for bot in &bots {
        for human in &humans {
            if bot > human {
                wins += 1.0;
            } else if bot == human {
                wins += 0.5;
            }
        }
    }
    Some(wins / (bots.len() * humans.len()) as f64)
}

/// Mean squared difference between `p_bot` and the outcome (1 for a bot, 0 for a human).
pub fn brier(samples: &[Scored]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let total: f64 = samples
        .iter()
        .map(|s| {
            let outcome = if s.is_bot() { 1.0 } else { 0.0 };
            (s.p_bot - outcome).powi(2)
        })
        .sum();
    total / samples.len() as f64
}

/// Ten equal-width bins over `[0, 1]`.
pub fn reliability_bins(samples: &[Scored]) -> Vec<Bin> {
    let mut sums = [(0usize, 0.0f64, 0usize); BIN_COUNT];
    for s in samples {
        let index = ((s.p_bot * BIN_COUNT as f64).floor().max(0.0) as usize).min(BIN_COUNT - 1);
        let entry = &mut sums[index];
        entry.0 += 1;
        entry.1 += s.p_bot;
        entry.2 += usize::from(s.is_bot());
    }
    sums.iter()
        .enumerate()
        .map(|(index, &(count, p_sum, bots))| Bin {
            lower: index as f64 / BIN_COUNT as f64,
            upper: (index + 1) as f64 / BIN_COUNT as f64,
            count,
            mean_p_bot: if count == 0 {
                0.0
            } else {
                p_sum / count as f64
            },
            bot_share: ratio(bots, count),
        })
        .collect()
}

/// Expected calibration error: the count-weighted mean gap between each bin's
/// mean bot probability and its share of bots.
pub fn expected_calibration_error(bins: &[Bin]) -> f64 {
    let total: usize = bins.iter().map(|b| b.count).sum();
    if total == 0 {
        return 0.0;
    }
    bins.iter()
        .map(|b| b.count as f64 * (b.mean_p_bot - b.bot_share).abs())
        .sum::<f64>()
        / total as f64
}

/// Returns the threshold that maximises F1 on the slice. Candidates are the
/// bot probabilities in the slice. A tie goes to the lowest threshold.
/// Returns `None` for an empty slice.
pub fn best_threshold(samples: &[Scored]) -> Option<f64> {
    let mut candidates: Vec<f64> = samples.iter().map(|s| s.p_bot).collect();
    candidates.sort_by(|a, b| a.total_cmp(b));
    candidates.dedup();
    let mut best: Option<(f64, f64)> = None;
    for threshold in candidates {
        let score = f1(samples, threshold);
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((threshold, score));
        }
    }
    best.map(|(threshold, _)| threshold)
}

/// Computes every metric on one slice at one threshold.
pub fn evaluate(samples: &[Scored], threshold: f64) -> Metrics {
    let bins = reliability_bins(samples);
    Metrics {
        threshold,
        accuracy: accuracy(samples, threshold),
        precision: precision(samples, threshold),
        recall: recall(samples, threshold),
        f1: f1(samples, threshold),
        roc_auc: roc_auc(samples),
        brier: brier(samples),
        ece: expected_calibration_error(&bins),
        recall_bot: recall_for(samples, Label::Bot, threshold),
        recall_agent: recall_for(samples, Label::Agent, threshold),
        bins,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(p_bot: f64, label: Label) -> Scored {
        Scored { p_bot, label }
    }

    /// Bots: 0.9 (agent 0.8, bot 0.9, bot 0.4). Humans: 0.6, 0.2, 0.1.
    fn sample() -> Vec<Scored> {
        vec![
            s(0.9, Label::Bot),
            s(0.8, Label::Agent),
            s(0.6, Label::Human),
            s(0.4, Label::Bot),
            s(0.2, Label::Human),
            s(0.1, Label::Human),
        ]
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn threshold_metrics_match_hand_counts() {
        // At 0.5: TP 2 (0.9, 0.8), FP 1 (0.6), FN 1 (0.4), TN 2.
        let data = sample();
        close(accuracy(&data, 0.5), 4.0 / 6.0);
        close(precision(&data, 0.5), 2.0 / 3.0);
        close(recall(&data, 0.5), 2.0 / 3.0);
        close(f1(&data, 0.5), 2.0 / 3.0);
    }

    #[test]
    fn the_threshold_is_inclusive() {
        let data = vec![s(0.5, Label::Bot), s(0.1, Label::Human)];
        close(recall(&data, 0.5), 1.0);
    }

    #[test]
    fn roc_auc_counts_winning_pairs() {
        // 9 pairs. Bot 0.9 wins 3, bot 0.8 wins 3, bot 0.4 wins 2. 8/9.
        close(roc_auc(&sample()).unwrap(), 8.0 / 9.0);
    }

    #[test]
    fn roc_auc_counts_a_tie_as_half() {
        // Bots 0.5, 0.7. Humans 0.5, 0.3. Pairs: tie, win, win, win = 3.5 of 4.
        let data = vec![
            s(0.5, Label::Bot),
            s(0.7, Label::Bot),
            s(0.5, Label::Human),
            s(0.3, Label::Human),
        ];
        close(roc_auc(&data).unwrap(), 0.875);
    }

    #[test]
    fn roc_auc_is_empty_with_one_class() {
        assert_eq!(roc_auc(&[s(0.5, Label::Bot)]), None);
        assert_eq!(roc_auc(&[]), None);
    }

    #[test]
    fn brier_is_the_mean_squared_gap() {
        // Gaps: 0.1, 0.2, 0.6, 0.6, 0.2, 0.1. Squares sum to 0.82. Mean 0.82 / 6.
        close(brier(&sample()), 0.82 / 6.0);
    }

    #[test]
    fn bins_hold_counts_and_means() {
        let bins = reliability_bins(&sample());
        assert_eq!(bins.len(), 10);
        assert_eq!(bins.iter().map(|b| b.count).sum::<usize>(), 6);
        assert_eq!(bins[0].count, 0);
        assert_eq!(bins[9].count, 1);
        close(bins[9].mean_p_bot, 0.9);
        close(bins[9].bot_share, 1.0);
        close(bins[6].bot_share, 0.0);
        close(bins[2].lower, 0.2);
        close(bins[2].upper, 0.3);
    }

    #[test]
    fn a_bin_averages_its_items() {
        let data = vec![s(0.61, Label::Bot), s(0.69, Label::Human)];
        let bins = reliability_bins(&data);
        assert_eq!(bins[6].count, 2);
        close(bins[6].mean_p_bot, 0.65);
        close(bins[6].bot_share, 0.5);
    }

    #[test]
    fn a_probability_of_one_lands_in_the_last_bin() {
        let bins = reliability_bins(&[s(1.0, Label::Bot)]);
        assert_eq!(bins[9].count, 1);
    }

    #[test]
    fn ece_weights_each_bin_gap_by_its_count() {
        // One item per bin. Gaps: 0.1, 0.2, 0.6, 0.6, 0.2, 0.1. Sum 1.8. Mean 0.3.
        let bins = reliability_bins(&sample());
        close(expected_calibration_error(&bins), 0.3);
    }

    #[test]
    fn recall_is_split_by_label() {
        let data = sample();
        // Bot items: 0.9 and 0.4. At 0.5 one of two.
        close(recall_for(&data, Label::Bot, 0.5).unwrap(), 0.5);
        // Agent items: 0.8. At 0.5 one of one.
        close(recall_for(&data, Label::Agent, 0.5).unwrap(), 1.0);
        assert_eq!(recall_for(&[s(0.5, Label::Human)], Label::Agent, 0.5), None);
    }

    #[test]
    fn best_threshold_maximises_f1() {
        // F1 by threshold: 0.1 .667, 0.2 .75, 0.4 .857, 0.6 .667, 0.8 .8, 0.9 .5.
        close(best_threshold(&sample()).unwrap(), 0.4);
        assert_eq!(best_threshold(&[]), None);
    }

    #[test]
    fn best_threshold_breaks_a_tie_toward_the_lowest() {
        // Both thresholds give F1 1.0.
        let data = vec![s(0.7, Label::Bot), s(0.7, Label::Bot), s(0.1, Label::Human)];
        close(best_threshold(&data).unwrap(), 0.7);
    }

    #[test]
    fn evaluate_fills_every_field() {
        let m = evaluate(&sample(), 0.5);
        close(m.accuracy, 4.0 / 6.0);
        close(m.roc_auc.unwrap(), 8.0 / 9.0);
        close(m.ece, 0.3);
        assert_eq!(m.bins.len(), 10);
    }

    #[test]
    fn an_empty_slice_gives_zeros_not_a_panic() {
        let m = evaluate(&[], 0.5);
        assert_eq!(m.accuracy, 0.0);
        assert_eq!(m.roc_auc, None);
        assert_eq!(m.brier, 0.0);
        assert_eq!(m.ece, 0.0);
    }
}
