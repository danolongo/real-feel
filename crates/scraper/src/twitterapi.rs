use std::{collections::HashSet, time::Duration};

use chrono::{DateTime, Utc};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::Value;
use types::{Account, Tweet};

use crate::{Error, Scraped, Scraper};

/// The date format of search results, such as `Sun Oct 25 09:56:57 +0000 2026`.
const DATE_FORMAT: &str = "%a %b %d %H:%M:%S %z %Y";

/// The most requests for one page, counting the first.
const MAX_ATTEMPTS: u32 = 3;

#[derive(Deserialize)]
struct SearchPage {
    tweets: Vec<Value>,
    has_next_page: bool,
    next_cursor: String,
}

#[derive(Deserialize)]
struct SearchTweet {
    id: String,
    text: String,
    #[serde(rename = "createdAt")]
    created_at: String,
    author: Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchAuthor {
    id: String,
    user_name: String,
    created_at: Option<String>,
    followers: Option<i32>,
    following: Option<i32>,
    statuses_count: Option<i32>,
    is_blue_verified: Option<bool>,
    is_automated: Option<bool>,
}

fn parse_date(date: &str) -> Result<DateTime<Utc>, Error> {
    Ok(DateTime::parse_from_str(date, DATE_FORMAT)?.with_timezone(&Utc))
}

/// Searches X through twitterapi.io.
#[derive(Debug, Clone)]
pub struct TwitterApi {
    client: reqwest::Client,
    base_url: String,
    key: String,
    pages: u32,
    interval: Duration,
}

impl TwitterApi {
    /// `pages` is the most search pages for one query. `interval` is the pause between two requests.
    pub fn new(
        base_url: impl Into<String>,
        key: impl Into<String>,
        pages: u32,
        interval: Duration,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            key: key.into(),
            pages,
            interval,
        }
    }

    /// Sends the request again after `interval` when twitterapi.io answers `429`.
    #[tracing::instrument(name = "twitterapi_page", skip(self, query, cursor))]
    async fn fetch_page(&self, query: &str, cursor: &str, page: u32) -> Result<SearchPage, Error> {
        let mut attempt = 1;
        loop {
            let response = self
                .client
                .get(format!(
                    "{}/twitter/tweet/advanced_search",
                    self.base_url.trim_end_matches('/')
                ))
                .header("X-API-Key", &self.key)
                .query(&[
                    (
                        "query",
                        format!("{query} lang:en -filter:retweets").as_str(),
                    ),
                    ("queryType", "Latest"),
                    ("cursor", cursor),
                ])
                .send()
                .await?;
            if response.status() == StatusCode::TOO_MANY_REQUESTS && attempt < MAX_ATTEMPTS {
                tracing::warn!(
                    attempt,
                    "twitterapi.io answered 429, sending the request again"
                );
                attempt += 1;
                tokio::time::sleep(self.interval).await;
                continue;
            }
            return Ok(response.error_for_status()?.json().await?);
        }
    }
}

impl Scraper for TwitterApi {
    #[tracing::instrument(name = "twitterapi_scrape", skip_all)]
    async fn scrape(&self, query: &str) -> Result<Scraped, Error> {
        let mut accounts = Vec::new();
        let mut tweets = Vec::new();
        let mut account_ids = HashSet::new();
        let mut tweet_ids = HashSet::new();
        let mut cursor = String::new();
        for page_number in 0..self.pages {
            if page_number > 0 {
                tokio::time::sleep(self.interval).await;
            }
            let page = self.fetch_page(query, &cursor, page_number + 1).await?;
            tracing::info!(
                page = page_number + 1,
                tweets = page.tweets.len(),
                has_next_page = page.has_next_page,
                "twitterapi.io page fetched"
            );
            for raw in page.tweets {
                let tweet: SearchTweet = serde_json::from_value(raw.clone())?;
                let author: SearchAuthor = serde_json::from_value(tweet.author.clone())?;
                if account_ids.insert(author.id.clone()) {
                    accounts.push(Account {
                        id: author.id.clone(),
                        handle: author.user_name,
                        joined_at: author.created_at.as_deref().map(parse_date).transpose()?,
                        followers: author.followers,
                        following: author.following,
                        tweet_count: author.statuses_count,
                        is_verified: author.is_blue_verified,
                        is_automated: author.is_automated,
                        raw: tweet.author,
                    });
                }
                if tweet_ids.insert(tweet.id.clone()) {
                    tweets.push(Tweet {
                        id: tweet.id,
                        account_id: author.id,
                        text: tweet.text,
                        posted_at: Some(parse_date(&tweet.created_at)?),
                        raw,
                    });
                }
            }
            if !page.has_next_page {
                break;
            }
            cursor = page.next_cursor;
        }
        tracing::info!(
            tweets = tweets.len(),
            accounts = accounts.len(),
            "scrape finished"
        );
        Ok(Scraped { accounts, tweets })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, method, path, query_param},
    };

    use super::*;

    fn tweet(id: &str, author_id: &str) -> Value {
        json!({
            "id": id,
            "text": format!("made-up tweet {id}"),
            "createdAt": "Thu Jan 01 10:00:00 +0000 2026",
            "likeCount": 3,
            "author": {
                "id": author_id,
                "userName": format!("made_up_{author_id}"),
                "createdAt": "Wed Dec 31 08:30:00 +0000 2025",
                "followers": 120,
                "following": 45,
                "statusesCount": 900,
                "isBlueVerified": true,
                "isAutomated": false,
                "location": "nowhere",
            },
        })
    }

    fn page(tweets: Vec<Value>, next_cursor: Option<&str>) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({
            "tweets": tweets,
            "has_next_page": next_cursor.is_some(),
            "next_cursor": next_cursor.unwrap_or(""),
        }))
    }

    fn scraper(server: &MockServer, pages: u32) -> TwitterApi {
        TwitterApi::new(server.uri(), "made-up-key", pages, Duration::ZERO)
    }

    #[tokio::test]
    async fn sends_search_request_with_key_query_and_cursor() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/twitter/tweet/advanced_search"))
            .and(header("X-API-Key", "made-up-key"))
            .and(query_param("query", "ai news lang:en -filter:retweets"))
            .and(query_param("queryType", "Latest"))
            .and(query_param("cursor", ""))
            .respond_with(page(vec![], None))
            .expect(1)
            .mount(&server)
            .await;
        let scraped = scraper(&server, 5).scrape("ai news").await.unwrap();
        assert!(scraped.tweets.is_empty());
    }

    #[tokio::test]
    async fn follows_next_cursor_and_stops_when_has_next_page_is_false() {
        let server = MockServer::start().await;
        Mock::given(query_param("cursor", ""))
            .respond_with(page(vec![tweet("101", "1")], Some("cursor-one")))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(query_param("cursor", "cursor-one"))
            .respond_with(page(vec![tweet("102", "2")], None))
            .expect(1)
            .mount(&server)
            .await;
        let scraped = scraper(&server, 5).scrape("q").await.unwrap();
        assert_eq!(scraped.tweets.len(), 2);
    }

    #[tokio::test]
    async fn stops_at_the_page_limit() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(page(vec![tweet("101", "1")], Some("more")))
            .expect(2)
            .mount(&server)
            .await;
        scraper(&server, 2).scrape("q").await.unwrap();
    }

    #[tokio::test]
    async fn maps_tweets_and_authors_and_keeps_them_in_raw() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(page(vec![tweet("101", "1")], None))
            .mount(&server)
            .await;
        let scraped = scraper(&server, 1).scrape("q").await.unwrap();
        let account = &scraped.accounts[0];
        assert_eq!(account.id, "1");
        assert_eq!(account.handle, "made_up_1");
        assert_eq!(account.followers, Some(120));
        assert_eq!(account.following, Some(45));
        assert_eq!(account.tweet_count, Some(900));
        assert_eq!(account.is_verified, Some(true));
        assert_eq!(account.is_automated, Some(false));
        assert_eq!(account.raw["location"], "nowhere");
        let tweet = &scraped.tweets[0];
        assert_eq!(tweet.id, "101");
        assert_eq!(tweet.account_id, "1");
        assert_eq!(tweet.text, "made-up tweet 101");
        assert_eq!(tweet.raw["likeCount"], 3);
    }

    #[tokio::test]
    async fn keeps_one_account_for_a_repeated_author_and_one_tweet_for_a_repeated_id() {
        let server = MockServer::start().await;
        Mock::given(query_param("cursor", ""))
            .respond_with(page(
                vec![tweet("101", "1"), tweet("102", "1")],
                Some("next"),
            ))
            .mount(&server)
            .await;
        Mock::given(query_param("cursor", "next"))
            .respond_with(page(vec![tweet("102", "1"), tweet("103", "2")], None))
            .mount(&server)
            .await;
        let scraped = scraper(&server, 5).scrape("q").await.unwrap();
        let ids = |ids: Vec<&String>| ids.into_iter().cloned().collect::<Vec<_>>();
        assert_eq!(
            ids(scraped.accounts.iter().map(|a| &a.id).collect()),
            ["1", "2"]
        );
        assert_eq!(
            ids(scraped.tweets.iter().map(|t| &t.id).collect()),
            ["101", "102", "103"]
        );
    }

    #[tokio::test]
    async fn parses_search_dates() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(page(vec![tweet("101", "1")], None))
            .mount(&server)
            .await;
        let scraped = scraper(&server, 1).scrape("q").await.unwrap();
        assert_eq!(
            scraped.tweets[0].posted_at.unwrap().to_rfc3339(),
            "2026-01-01T10:00:00+00:00"
        );
        assert_eq!(
            scraped.accounts[0].joined_at.unwrap().to_rfc3339(),
            "2025-12-31T08:30:00+00:00"
        );
    }

    #[tokio::test]
    async fn fails_on_a_malformed_date() {
        let server = MockServer::start().await;
        let mut bad = tweet("101", "1");
        bad["createdAt"] = json!("yesterday");
        Mock::given(method("GET"))
            .respond_with(page(vec![bad], None))
            .mount(&server)
            .await;
        let error = scraper(&server, 1).scrape("q").await.unwrap_err();
        assert!(matches!(error, Error::Date(_)));
    }

    #[tokio::test]
    async fn fails_on_http_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .expect(1)
            .mount(&server)
            .await;
        let error = scraper(&server, 1).scrape("q").await.unwrap_err();
        assert!(matches!(error, Error::Http(_)));
    }

    #[tokio::test]
    async fn sends_the_request_again_after_a_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(page(vec![tweet("101", "1")], None))
            .expect(1)
            .mount(&server)
            .await;
        let scraped = scraper(&server, 1).scrape("q").await.unwrap();
        assert_eq!(scraped.tweets.len(), 1);
    }

    #[tokio::test]
    async fn fails_after_three_attempts_that_answer_429() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(429))
            .expect(3)
            .mount(&server)
            .await;
        let error = scraper(&server, 1).scrape("q").await.unwrap_err();
        assert!(matches!(error, Error::Http(_)));
    }
}
