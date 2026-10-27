use std::time::Duration;

use chrono::{DateTime, Duration as Span, TimeZone, Utc};
use serde_json::json;
use sha2::{Digest, Sha256};
use types::{Account, Tweet};

use crate::{Error, Scraped, Scraper};

const TWEETS_PER_ACCOUNT: usize = 5;

/// Makes `tweets` tweets for any query, without a network call. One query text always gives the same IDs.
#[derive(Debug, Clone)]
pub struct Synthetic {
    tweets: usize,
    latency: Duration,
}

impl Synthetic {
    /// `latency` is the pause before each `scrape` returns.
    pub fn new(tweets: usize, latency: Duration) -> Self {
        Self { tweets, latency }
    }
}

/// SHA-256 of the query (already lower-cased), a kind, and an index. The first 8 bytes become a number.
fn number(query: &str, kind: &str, index: usize) -> u64 {
    let digest = Sha256::new()
        .chain_update(query)
        .chain_update([0])
        .chain_update(kind)
        .chain_update(index.to_le_bytes())
        .finalize();
    u64::from_be_bytes(digest[..8].try_into().expect("a digest has 32 bytes"))
}

fn base_time() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
}

fn account(query: &str, index: usize) -> Account {
    let n = number(query, "account", index);
    // IDs hold numbers as text, like X's IDs. The shift keeps each one inside i64.
    let id = (n >> 1).to_string();
    let followers = (n % 5_000) as i32;
    let tweet_count = ((n >> 8) % 20_000) as i32;
    Account {
        handle: format!("synthetic_{n:016x}"),
        joined_at: Some(base_time() - Span::days(30 + ((n >> 16) % 3_000) as i64)),
        followers: Some(followers),
        following: Some(((n >> 24) % 2_000) as i32),
        tweet_count: Some(tweet_count),
        is_verified: Some(n.is_multiple_of(17)),
        is_automated: Some(n.is_multiple_of(11)),
        raw: json!({
            "id": id,
            "name": format!("Synthetic {n:016x}"),
            "favouritesCount": (n >> 32) % 30_000,
            "mediaCount": (n >> 40) % 500,
            "location": "Nowhere",
            "profile_bio": {"description": "A synthetic account for load tests."},
        }),
        id,
    }
}

fn tweet(query: &str, index: usize, account_id: String) -> Tweet {
    let n = number(query, "tweet", index);
    Tweet {
        id: (n >> 1).to_string(),
        account_id,
        text: format!(
            "Synthetic tweet {n:016x} about {query}. It has some words, so the model input has a realistic size."
        ),
        posted_at: Some(base_time() + Span::minutes(index as i64)),
        raw: json!({
            "likeCount": (n >> 8) % 1_000,
            "retweetCount": (n >> 20) % 200,
            "replyCount": (n >> 30) % 100,
            "quoteCount": (n >> 40) % 50,
            "viewCount": (n >> 44) % 100_000,
        }),
    }
}

impl Scraper for Synthetic {
    async fn scrape(&self, query: &str) -> Result<Scraped, Error> {
        tokio::time::sleep(self.latency).await;
        let query = &query.to_lowercase();
        let accounts: Vec<Account> = (0..self.tweets.div_ceil(TWEETS_PER_ACCOUNT))
            .map(|index| account(query, index))
            .collect();
        let tweets = (0..self.tweets)
            .map(|index| {
                tweet(
                    query,
                    index,
                    accounts[index / TWEETS_PER_ACCOUNT].id.clone(),
                )
            })
            .collect();
        Ok(Scraped { accounts, tweets })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use tokio::time::Instant;

    use super::*;

    fn ids(scraped: &Scraped) -> Vec<&str> {
        let tweets = scraped.tweets.iter().map(|t| t.id.as_str());
        let accounts = scraped.accounts.iter().map(|a| a.id.as_str());
        tweets.chain(accounts).collect()
    }

    #[tokio::test]
    async fn one_query_text_gives_the_same_ids_and_a_new_text_gives_new_ids() {
        let synthetic = Synthetic::new(50, Duration::ZERO);
        let first = synthetic.scrape("Rust Lang").await.unwrap();
        let again = synthetic.scrape("rust lang").await.unwrap();
        let other = synthetic.scrape("go lang").await.unwrap();
        assert_eq!(first, again);
        let first_ids: HashSet<_> = ids(&first).into_iter().collect();
        assert_eq!(first_ids.len(), 50 + 10);
        assert!(ids(&other).iter().all(|id| !first_ids.contains(id)));
    }

    #[tokio::test]
    async fn makes_the_requested_number_of_tweets_with_matching_accounts() {
        for tweets in [0, 1, 7, 50] {
            let scraped = Synthetic::new(tweets, Duration::ZERO)
                .scrape("q")
                .await
                .unwrap();
            assert_eq!(scraped.tweets.len(), tweets);
            assert_eq!(scraped.accounts.len(), tweets.div_ceil(TWEETS_PER_ACCOUNT));
            for tweet in &scraped.tweets {
                assert!(scraped.accounts.iter().any(|a| a.id == tweet.account_id));
            }
        }
    }

    #[tokio::test(start_paused = true)]
    async fn pauses_for_the_latency() {
        let synthetic = Synthetic::new(1, Duration::from_millis(250));
        let started = Instant::now();
        synthetic.scrape("q").await.unwrap();
        assert_eq!(started.elapsed(), Duration::from_millis(250));
    }
}
