use std::collections::HashMap;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use types::{Account, Tweet};

use crate::Error;

/// What `eval collect` writes: one account, the tweet that matched the query, and the recent tweets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectedItem {
    pub account: Account,
    pub tweet: Tweet,
    pub recent_tweets: Vec<Tweet>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordLabel {
    Human,
    Bot,
    Agent,
    Unsure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TweetFlag {
    pub tweet_id: String,
    pub is_llm_text: bool,
}

/// What the label agent writes for one account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelRecord {
    pub account_id: String,
    pub handle: String,
    pub label: RecordLabel,
    pub confidence: Confidence,
    pub tweets: Vec<TweetFlag>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    Human,
    Bot,
    Agent,
}

impl Label {
    /// An agent account is a bot that an LLM runs.
    pub fn is_bot(self) -> bool {
        matches!(self, Label::Bot | Label::Agent)
    }
}

/// One eval set item: a collected item with its label.
/// `is_llm_text` holds the flag of the tweet that matched the query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalItem {
    pub account: Account,
    pub tweet: Tweet,
    pub recent_tweets: Vec<Tweet>,
    pub label: Label,
    pub is_llm_text: Option<bool>,
    pub confidence: Confidence,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Joined {
    pub items: Vec<EvalItem>,
    pub dropped_unsure: usize,
    pub dropped_unlabeled: usize,
}

/// Matches collected items to label records by account ID.
/// Drops `unsure` labels and items with no label record.
pub fn join(collected: Vec<CollectedItem>, records: Vec<LabelRecord>) -> Result<Joined, Error> {
    let mut by_account: HashMap<String, LabelRecord> = HashMap::new();
    for record in records {
        let account_id = record.account_id.clone();
        if by_account.insert(account_id.clone(), record).is_some() {
            return Err(Error::DuplicateLabelRecord { account_id });
        }
    }

    let mut joined = Joined {
        items: Vec::new(),
        dropped_unsure: 0,
        dropped_unlabeled: 0,
    };
    for item in collected {
        let Some(record) = by_account.get(&item.account.id) else {
            joined.dropped_unlabeled += 1;
            continue;
        };
        let label = match record.label {
            RecordLabel::Human => Label::Human,
            RecordLabel::Bot => Label::Bot,
            RecordLabel::Agent => Label::Agent,
            RecordLabel::Unsure => {
                joined.dropped_unsure += 1;
                continue;
            }
        };
        let is_llm_text = record
            .tweets
            .iter()
            .find(|flag| flag.tweet_id == item.tweet.id)
            .map(|flag| flag.is_llm_text);
        joined.items.push(EvalItem {
            account: item.account,
            tweet: item.tweet,
            recent_tweets: item.recent_tweets,
            label,
            is_llm_text,
            confidence: record.confidence,
        });
    }
    Ok(joined)
}

/// Parses JSON Lines. Blank lines are skipped.
pub fn parse_jsonl<T: DeserializeOwned>(text: &str) -> Result<Vec<T>, Error> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|source| Error::Line {
                line: index + 1,
                source,
            })
        })
        .collect()
}

pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, Error> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_jsonl(&text)
}

pub fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), Error> {
    let mut text = String::new();
    for row in rows {
        text.push_str(&serde_json::to_string(row)?);
        text.push('\n');
    }
    std::fs::write(path, text).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn collected(account_id: &str, tweet_id: &str) -> CollectedItem {
        CollectedItem {
            account: Account {
                id: account_id.into(),
                handle: format!("handle_{account_id}"),
                joined_at: None,
                followers: None,
                following: None,
                tweet_count: None,
                is_verified: None,
                is_automated: None,
                raw: json!({}),
            },
            tweet: Tweet {
                id: tweet_id.into(),
                account_id: account_id.into(),
                text: "text".into(),
                posted_at: None,
                raw: json!({}),
            },
            recent_tweets: vec![],
        }
    }

    fn record(account_id: &str, label: RecordLabel, flags: &[(&str, bool)]) -> LabelRecord {
        LabelRecord {
            account_id: account_id.into(),
            handle: format!("handle_{account_id}"),
            label,
            confidence: Confidence::High,
            tweets: flags
                .iter()
                .map(|(id, flag)| TweetFlag {
                    tweet_id: (*id).into(),
                    is_llm_text: *flag,
                })
                .collect(),
            reason: "reason".into(),
        }
    }

    #[test]
    fn is_bot_is_true_for_bot_and_agent() {
        assert!(!Label::Human.is_bot());
        assert!(Label::Bot.is_bot());
        assert!(Label::Agent.is_bot());
    }

    #[test]
    fn join_drops_unsure_and_unlabeled_and_reports_both() {
        let collected = vec![
            collected("1", "t1"),
            collected("2", "t2"),
            collected("3", "t3"),
            collected("4", "t4"),
        ];
        let records = vec![
            record("1", RecordLabel::Human, &[("t1", false)]),
            record("2", RecordLabel::Unsure, &[]),
            record("3", RecordLabel::Agent, &[("other", false), ("t3", true)]),
        ];
        let joined = join(collected, records).unwrap();
        assert_eq!(joined.dropped_unsure, 1);
        assert_eq!(joined.dropped_unlabeled, 1);
        assert_eq!(joined.items.len(), 2);
        assert_eq!(joined.items[0].label, Label::Human);
        assert_eq!(joined.items[0].is_llm_text, Some(false));
        assert_eq!(joined.items[1].label, Label::Agent);
        assert_eq!(joined.items[1].is_llm_text, Some(true));
    }

    #[test]
    fn join_leaves_is_llm_text_empty_when_no_flag_matches_the_tweet() {
        let joined = join(
            vec![collected("1", "t1")],
            vec![record("1", RecordLabel::Bot, &[("other", true)])],
        )
        .unwrap();
        assert_eq!(joined.items[0].is_llm_text, None);
    }

    #[test]
    fn join_rejects_two_records_for_one_account() {
        let result = join(
            vec![collected("1", "t1")],
            vec![
                record("1", RecordLabel::Bot, &[]),
                record("1", RecordLabel::Human, &[]),
            ],
        );
        assert!(matches!(result, Err(Error::DuplicateLabelRecord { .. })));
    }

    #[test]
    fn jsonl_round_trips_and_skips_blank_lines() {
        let items = vec![collected("1", "t1"), collected("2", "t2")];
        let mut text = String::new();
        for item in &items {
            text.push_str(&serde_json::to_string(item).unwrap());
            text.push_str("\n\n");
        }
        let parsed: Vec<CollectedItem> = parse_jsonl(&text).unwrap();
        assert_eq!(parsed, items);
    }

    #[test]
    fn jsonl_error_names_the_line() {
        let result = parse_jsonl::<CollectedItem>("\n{not json}\n");
        assert!(matches!(result, Err(Error::Line { line: 2, .. })));
    }
}
