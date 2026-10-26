use std::collections::HashSet;
use std::io::{BufRead, Write};

use scraper::{Scraper, TwitterApi, USD_PER_1000_TWEETS, USD_PER_CALL_MINIMUM};
use serde::Deserialize;
use types::Tweet;

use crate::Error;
use crate::item::{CollectedItem, parse_jsonl};

/// Tweets in one twitterapi.io search page.
pub const TWEETS_PER_PAGE: usize = 20;

/// What a run can cost at most, before any call.
#[derive(Debug, Clone, PartialEq)]
pub struct Estimate {
    pub search_tweets: usize,
    /// Every search tweet could come from a new account.
    pub max_accounts: usize,
    pub max_recent_tweets: usize,
    pub usd: f64,
}

/// The price of `calls` successful calls that return `tweets` tweets in total.
/// twitterapi.io charges at least `USD_PER_CALL_MINIMUM` for each call.
pub fn cost_usd(tweets: usize, calls: usize) -> f64 {
    let by_tweets = tweets as f64 / 1000.0 * USD_PER_1000_TWEETS;
    by_tweets.max(calls as f64 * USD_PER_CALL_MINIMUM)
}

/// Upper bound: every search page is full and every search tweet has a new author.
pub fn estimate(queries: usize, pages: u32, recent: usize) -> Estimate {
    let search_calls = queries * pages as usize;
    let search_tweets = search_calls * TWEETS_PER_PAGE;
    let max_accounts = search_tweets;
    let max_recent_tweets = max_accounts * recent;
    Estimate {
        search_tweets,
        max_accounts,
        max_recent_tweets,
        usd: cost_usd(search_tweets, search_calls) + cost_usd(max_recent_tweets, max_accounts),
    }
}

/// Reads one query for each line. Blank lines and lines that start with `#` are skipped.
pub fn parse_queries(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

#[derive(Deserialize)]
struct SeenItem {
    account: SeenAccount,
}

#[derive(Deserialize)]
struct SeenAccount {
    id: String,
}

/// Reads the account IDs of a collected file or an eval set file.
pub fn parse_seen_ids(text: &str) -> Result<HashSet<String>, Error> {
    let items: Vec<SeenItem> = parse_jsonl(text)?;
    Ok(items.into_iter().map(|item| item.account.id).collect())
}

/// Asks "Continue? [y/N]". Only `y` or `yes` answers yes.
pub fn confirm(input: &mut impl BufRead, output: &mut impl Write) -> std::io::Result<bool> {
    write!(output, "Continue? [y/N] ")?;
    output.flush()?;
    let mut answer = String::new();
    input.read_line(&mut answer)?;
    Ok(matches!(answer.trim().to_lowercase().as_str(), "y" | "yes"))
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Summary {
    pub queries_failed: usize,
    pub search_tweets: usize,
    pub collected: usize,
    pub skipped: usize,
    pub failed: usize,
    pub recent_tweets: usize,
    pub usd: f64,
}

/// Runs every query. For each new account it keeps the first matching tweet, fetches
/// `recent` recent tweets, and writes one `CollectedItem` line. `seen` holds the IDs of
/// accounts to skip. A failure for one query or one account logs a warning and moves on.
pub async fn run(
    api: &TwitterApi,
    queries: &[String],
    recent: usize,
    seen: &HashSet<String>,
    out: &mut impl Write,
) -> Result<Summary, Error> {
    let mut summary = Summary::default();
    let mut attempted: HashSet<String> = HashSet::new();
    let mut search_calls = 0;
    for query in queries {
        let scraped = match api.scrape(query).await {
            Ok(scraped) => scraped,
            Err(error) => {
                eprintln!("warning: query {query:?} failed: {error}");
                summary.queries_failed += 1;
                continue;
            }
        };
        search_calls += 1;
        summary.search_tweets += scraped.tweets.len();
        for tweet in scraped.tweets {
            if !attempted.insert(tweet.account_id.clone()) {
                continue;
            }
            if seen.contains(&tweet.account_id) {
                summary.skipped += 1;
                continue;
            }
            let Some(account) = scraped.accounts.iter().find(|a| a.id == tweet.account_id) else {
                continue;
            };
            let recent_tweets: Vec<Tweet> = match api.recent_tweets(&account.handle, recent).await {
                Ok(tweets) => tweets,
                Err(error) => {
                    eprintln!("warning: account {} failed: {error}", account.handle);
                    summary.failed += 1;
                    continue;
                }
            };
            summary.recent_tweets += recent_tweets.len();
            let item = CollectedItem {
                account: account.clone(),
                tweet,
                recent_tweets,
            };
            writeln!(out, "{}", serde_json::to_string(&item)?).map_err(write_error)?;
            out.flush().map_err(write_error)?;
            summary.collected += 1;
        }
    }
    summary.usd = cost_usd(summary.search_tweets, search_calls)
        + cost_usd(summary.recent_tweets, summary.collected);
    Ok(summary)
}

fn write_error(source: std::io::Error) -> Error {
    Error::Io {
        path: "the output file".into(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::{Value, json};
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[test]
    fn cost_is_the_larger_of_the_tweet_price_and_the_call_minimum() {
        assert!((cost_usd(1000, 1) - 0.15).abs() < 1e-12);
        assert!((cost_usd(0, 4) - 4.0 * USD_PER_CALL_MINIMUM).abs() < 1e-12);
    }

    #[test]
    fn estimate_counts_search_tweets_and_the_recent_tweet_bound() {
        let e = estimate(10, 2, 20);
        assert_eq!(e.search_tweets, 400);
        assert_eq!(e.max_accounts, 400);
        assert_eq!(e.max_recent_tweets, 8000);
        // 400 search tweets and 8,400 tweets in total at 0.15 USD per 1,000.
        assert!((e.usd - 8400.0 * 0.15 / 1000.0).abs() < 1e-9);
    }

    #[test]
    fn queries_skip_comments_and_blank_lines() {
        let text = "# bots\nfree crypto\n\n  ai agent  \n#last\n";
        assert_eq!(parse_queries(text), vec!["free crypto", "ai agent"]);
    }

    #[test]
    fn seen_ids_come_from_the_account_of_each_line() {
        let text = "{\"account\":{\"id\":\"1\",\"handle\":\"a\"},\"label\":\"bot\"}\n\n{\"account\":{\"id\":\"2\"}}\n";
        let ids = parse_seen_ids(text).unwrap();
        assert_eq!(ids, HashSet::from(["1".to_string(), "2".to_string()]));
    }

    #[test]
    fn confirm_accepts_only_y_and_yes() {
        for (answer, expected) in [
            ("y\n", true),
            ("YES\n", true),
            ("n\n", false),
            ("\n", false),
            ("", false),
        ] {
            let mut output = Vec::new();
            assert_eq!(
                confirm(&mut answer.as_bytes(), &mut output).unwrap(),
                expected
            );
            assert_eq!(String::from_utf8(output).unwrap(), "Continue? [y/N] ");
        }
    }

    fn search_tweet(id: &str, author_id: &str) -> Value {
        json!({
            "id": id,
            "text": format!("made-up tweet {id}"),
            "createdAt": "Thu Jan 01 10:00:00 +0000 2026",
            "author": {
                "id": author_id,
                "userName": format!("made_up_{author_id}"),
                "createdAt": "Wed Dec 31 08:30:00 +0000 2025",
                "followers": 120,
                "following": 45,
                "statusesCount": 900,
            },
        })
    }

    async fn mount_recent(server: &MockServer, handle: &str, tweets: Vec<Value>) {
        Mock::given(method("GET"))
            .and(path("/twitter/user/last_tweets"))
            .and(query_param("userName", handle))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": { "tweets": tweets },
                "has_next_page": false,
                "next_cursor": "",
            })))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn run_writes_one_item_for_each_new_account() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/twitter/tweet/advanced_search"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "tweets": [
                    search_tweet("101", "1"),
                    search_tweet("102", "1"),
                    search_tweet("103", "2"),
                    search_tweet("104", "3"),
                    search_tweet("105", "4"),
                ],
                "has_next_page": false,
                "next_cursor": "",
            })))
            .mount(&server)
            .await;
        mount_recent(
            &server,
            "made_up_1",
            vec![search_tweet("201", "1"), search_tweet("202", "1")],
        )
        .await;
        mount_recent(&server, "made_up_4", vec![search_tweet("401", "4")]).await;
        // Account 3 fails and account 2 is skipped, so no mock answers for either.
        Mock::given(method("GET"))
            .and(path("/twitter/user/last_tweets"))
            .and(query_param("userName", "made_up_3"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let api = TwitterApi::new(server.uri(), "made-up-key", 1, Duration::ZERO);
        let seen = HashSet::from(["2".to_string()]);
        let mut out = Vec::new();
        let summary = run(&api, &["made-up query".into()], 20, &seen, &mut out)
            .await
            .unwrap();

        assert_eq!(summary.collected, 2);
        assert_eq!(summary.skipped, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.search_tweets, 5);
        assert_eq!(summary.recent_tweets, 3);
        assert!(summary.usd > 0.0);

        let items: Vec<CollectedItem> = parse_jsonl(&String::from_utf8(out).unwrap()).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].account.id, "1");
        assert_eq!(items[0].tweet.id, "101");
        assert_eq!(items[0].recent_tweets.len(), 2);
        assert_eq!(items[1].account.handle, "made_up_4");
        assert_eq!(items[1].recent_tweets.len(), 1);
    }

    #[tokio::test]
    async fn run_moves_on_when_a_query_fails() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let api = TwitterApi::new(server.uri(), "made-up-key", 1, Duration::ZERO);
        let mut out = Vec::new();
        let summary = run(&api, &["q".into()], 20, &HashSet::new(), &mut out)
            .await
            .unwrap();
        assert_eq!(summary.queries_failed, 1);
        assert!(out.is_empty());
    }
}
