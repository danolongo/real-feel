use std::{future::Future, path::Path};

use serde::Deserialize;
use types::{Account, Tweet};

mod twitterapi;

pub use twitterapi::TwitterApi;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot read the fixture file: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON does not match its type: {0}")]
    Json(#[from] serde_json::Error),
    #[error("twitterapi.io request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("cannot parse a date: {0}")]
    Date(#[from] chrono::ParseError),
}

/// Every tweet's `account_id` matches an account in the same `Scraped`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Scraped {
    pub accounts: Vec<Account>,
    pub tweets: Vec<Tweet>,
}

pub trait Scraper {
    fn scrape(&self, query: &str) -> impl Future<Output = Result<Scraped, Error>> + Send;
}

/// Returns the content of one JSON file for any query.
#[derive(Debug, Clone)]
pub struct Fixture {
    scraped: Scraped,
}

impl Fixture {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let scraped = serde_json::from_slice(&std::fs::read(path)?)?;
        Ok(Self { scraped })
    }
}

impl Scraper for Fixture {
    async fn scrape(&self, _query: &str) -> Result<Scraped, Error> {
        Ok(self.scraped.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fixture_returns_default_json_for_any_query() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/default.json");
        let scraped = Fixture::load(path)
            .unwrap()
            .scrape("anything")
            .await
            .unwrap();
        assert_eq!(scraped.tweets.len(), 5);
        assert_eq!(scraped.accounts.len(), 3);
        for tweet in &scraped.tweets {
            assert!(scraped.accounts.iter().any(|a| a.id == tweet.account_id));
        }
    }
}
