use std::fmt::Write;

use serde::{Deserialize, Serialize};
use types::{MetadataVersion, ModelVersion, QuestionVersion};

use crate::manifest::LabelCounts;
use crate::metrics::Metrics;

/// How far `p_bot` moves when one item passes through one experiment more than once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Repeatability {
    pub repeats: i32,
    /// Items with two or more results.
    pub items: usize,
    /// The mean over items of the standard deviation of `p_bot` across repeats.
    pub mean_sd: f64,
    /// The largest standard deviation of `p_bot` across repeats over all items.
    pub max_sd: f64,
    /// The share of items whose `is_bot` answer differs between repeats.
    pub flip_share: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitLabelCounts {
    pub tune: LabelCounts,
    pub test: LabelCounts,
}

/// The result of one experiment on the test split, ready to render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub model_version: ModelVersion,
    pub metadata_version: MetadataVersion,
    pub question_version: QuestionVersion,
    pub labels: SplitLabelCounts,
    pub repeatability: Option<Repeatability>,
    pub metrics: Metrics,
}

impl ExperimentSummary {
    fn items(&self) -> usize {
        let (tune, test) = (&self.labels.tune, &self.labels.test);
        tune.human + tune.bot + tune.agent + test.human + test.bot + test.agent
    }

    fn name(&self) -> String {
        format!(
            "{} / {} / {} / {} items",
            self.model_version.0,
            self.metadata_version.0,
            self.question_version.0,
            self.items()
        )
    }
}

fn number(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}"))
}

/// Renders one comparison table and one calibration curve for each experiment.
pub fn render(summaries: &[ExperimentSummary]) -> String {
    let mut out = String::new();
    out.push_str("## Comparison\n\n");
    out.push_str(
        "| Model version | Metadata version | Question version | Items | Bot threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier | ECE | Recall (bot) | Recall (agent) |\n",
    );
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for summary in summaries {
        let m = &summary.metrics;
        let _ = writeln!(
            out,
            "| `{}` | `{}` | `{}` | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {:.3} | {:.3} | {} | {} |",
            summary.model_version.0,
            summary.metadata_version.0,
            summary.question_version.0,
            summary.items(),
            m.threshold,
            m.accuracy,
            m.precision,
            m.recall,
            m.f1,
            number(m.roc_auc),
            m.brier,
            m.ece,
            number(m.recall_bot),
            number(m.recall_agent),
        );
    }

    out.push_str("\n## Label counts\n\n");
    out.push_str("| Experiment | Tune human | Tune bot | Tune agent | Test human | Test bot | Test agent |\n");
    out.push_str("|---|---|---|---|---|---|---|\n");
    for summary in summaries {
        let (tune, test) = (&summary.labels.tune, &summary.labels.test);
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} |",
            summary.name(),
            tune.human,
            tune.bot,
            tune.agent,
            test.human,
            test.bot,
            test.agent
        );
    }

    if summaries.iter().any(|s| s.repeatability.is_some()) {
        out.push_str("\n## Repeatability\n\n");
        out.push_str("Every item passes through the experiment more than once. The standard deviation uses the sample formula.\n\n");
        out.push_str("| Experiment | Repeats | Items | Mean SD of p_bot | Largest SD of p_bot | Share of items that flip is_bot |\n");
        out.push_str("|---|---|---|---|---|---|\n");
        for summary in summaries {
            let Some(r) = &summary.repeatability else {
                continue;
            };
            let _ = writeln!(
                out,
                "| {} | {} | {} | {:.4} | {:.4} | {:.3} |",
                summary.name(),
                r.repeats,
                r.items,
                r.mean_sd,
                r.max_sd,
                r.flip_share
            );
        }
    }

    for summary in summaries {
        let _ = write!(
            out,
            "\n## Calibration curve: {}\n\n{}",
            summary.name(),
            calibration_chart(summary)
        );
    }
    out
}

/// A Mermaid `xychart-beta`. The x axis lists the mean bot probability of each
/// non-empty bin. One line plots the share of bots. The other line is the diagonal:
/// a calibrated experiment follows it.
fn calibration_chart(summary: &ExperimentSummary) -> String {
    let bins: Vec<_> = summary
        .metrics
        .bins
        .iter()
        .filter(|b| b.count > 0)
        .collect();
    let join = |values: Vec<f64>| {
        values
            .iter()
            .map(|v| format!("{v:.2}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let means = join(bins.iter().map(|b| b.mean_p_bot).collect());
    let shares = join(bins.iter().map(|b| b.bot_share).collect());

    let mut out = String::new();
    out.push_str("```mermaid\nxychart-beta\n");
    let _ = writeln!(out, "    title \"{}\"", summary.name());
    let _ = writeln!(out, "    x-axis \"Mean p_bot\" [{means}]");
    out.push_str("    y-axis \"Share of bots\" 0 --> 1\n");
    let _ = writeln!(out, "    line [{shares}]");
    let _ = writeln!(out, "    line [{means}]");
    out.push_str("```\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::item::Label;
    use crate::metrics::{Scored, evaluate};

    fn summary(model: &str, metadata: &str) -> ExperimentSummary {
        let samples = vec![
            Scored {
                p_bot: 0.9,
                label: Label::Bot,
            },
            Scored {
                p_bot: 0.8,
                label: Label::Agent,
            },
            Scored {
                p_bot: 0.2,
                label: Label::Human,
            },
            Scored {
                p_bot: 0.1,
                label: Label::Human,
            },
        ];
        ExperimentSummary {
            model_version: ModelVersion(model.into()),
            metadata_version: MetadataVersion(metadata.into()),
            question_version: QuestionVersion("v1".into()),
            labels: SplitLabelCounts {
                tune: LabelCounts {
                    human: 1,
                    bot: 1,
                    agent: 0,
                },
                test: LabelCounts {
                    human: 1,
                    bot: 0,
                    agent: 1,
                },
            },
            repeatability: None,
            metrics: evaluate(&samples, 0.5),
        }
    }

    #[test]
    fn render_has_one_table_row_and_one_chart_for_each_experiment() {
        let md = render(&[summary("fake", "v1"), summary("fake", "v2")]);
        assert_eq!(md.matches("```mermaid").count(), 2);
        assert_eq!(md.matches("xychart-beta").count(), 2);
        assert_eq!(md.matches("| `fake` |").count(), 2);
        assert!(md.contains(
            "| `fake` | `v1` | `v1` | 4 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |"
        ));
    }

    #[test]
    fn the_chart_plots_non_empty_bins_and_the_diagonal() {
        let md = render(&[summary("fake", "v1")]);
        // Non-empty bins have mean 0.10, 0.20, 0.80, 0.90. Bots share: 0, 0, 1, 1.
        assert!(md.contains("x-axis \"Mean p_bot\" [0.10, 0.20, 0.80, 0.90]"));
        assert!(md.contains("    line [0.00, 0.00, 1.00, 1.00]"));
        assert!(md.contains("    line [0.10, 0.20, 0.80, 0.90]"));
    }

    #[test]
    fn render_adds_label_counts_and_a_repeatability_table_only_when_repeats_exist() {
        let plain = render(&[summary("fake", "v1")]);
        assert!(plain.contains("## Label counts"));
        assert!(plain.contains("| fake / v1 / v1 / 4 items | 1 | 1 | 0 | 1 | 0 | 1 |"));
        assert!(!plain.contains("## Repeatability"));

        let mut repeated = summary("fake", "v1");
        repeated.repeatability = Some(Repeatability {
            repeats: 5,
            items: 50,
            mean_sd: 0.01,
            max_sd: 0.2,
            flip_share: 0.04,
        });
        let md = render(&[repeated]);
        assert!(md.contains("## Repeatability"));
        assert!(md.contains("| fake / v1 / v1 / 4 items | 5 | 50 | 0.0100 | 0.2000 | 0.040 |"));
    }

    #[test]
    fn a_missing_value_renders_as_n_a() {
        let mut one = summary("fake", "v1");
        one.metrics.roc_auc = None;
        assert!(render(&[one]).contains("| n/a |"));
    }
}
