use std::fmt::Write;

use serde::{Deserialize, Serialize};
use types::{MetadataVersion, ModelVersion, QuestionVersion};

use crate::metrics::Metrics;

/// The result of one experiment on one slice, ready to render.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub model_version: ModelVersion,
    pub metadata_version: MetadataVersion,
    pub question_version: QuestionVersion,
    pub metrics: Metrics,
}

impl ExperimentSummary {
    fn name(&self) -> String {
        format!(
            "{} / {} / {}",
            self.model_version.0, self.metadata_version.0, self.question_version.0
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
        "| Model version | Metadata version | Question version | Bot threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier | ECE | Recall (bot) | Recall (agent) |\n",
    );
    out.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for summary in summaries {
        let m = &summary.metrics;
        let _ = writeln!(
            out,
            "| `{}` | `{}` | `{}` | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {:.3} | {:.3} | {} | {} |",
            summary.model_version.0,
            summary.metadata_version.0,
            summary.question_version.0,
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
            metrics: evaluate(&samples, 0.5),
        }
    }

    #[test]
    fn render_has_one_table_row_and_one_chart_for_each_experiment() {
        let md = render(&[summary("fake", "v1"), summary("fake", "v2")]);
        assert_eq!(md.matches("```mermaid").count(), 2);
        assert_eq!(md.matches("xychart-beta").count(), 2);
        assert_eq!(md.matches("| `fake` |").count(), 2);
        assert!(
            md.contains("| `fake` | `v1` | `v1` | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |")
        );
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
    fn a_missing_value_renders_as_n_a() {
        let mut one = summary("fake", "v1");
        one.metrics.roc_auc = None;
        assert!(render(&[one]).contains("| n/a |"));
    }
}
