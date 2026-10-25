use std::collections::HashSet;

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};
use types::{Account, Tweet};

use crate::{Error, Scraped, Scraper};

const PROMPT: &str = "Extract every tweet on the page and the account that posted each tweet.";

fn extraction_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "accounts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "handle": { "type": "string" }
                    },
                    "required": ["id", "handle"]
                }
            },
            "tweets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "account_id": { "type": "string" },
                        "text": { "type": "string" },
                        "posted_at": { "type": "string" }
                    },
                    "required": ["id", "account_id", "text"]
                }
            }
        },
        "required": ["accounts", "tweets"]
    })
}

#[derive(Deserialize)]
struct SearchResponse {
    data: SearchData,
}

#[derive(Deserialize)]
struct SearchData {
    web: Vec<SearchHit>,
}

#[derive(Deserialize)]
struct SearchHit {
    json: Option<Extracted>,
}

#[derive(Deserialize)]
struct Extracted {
    #[serde(default)]
    accounts: Vec<Value>,
    #[serde(default)]
    tweets: Vec<Value>,
}

#[derive(Deserialize)]
struct ExtractedAccount {
    id: String,
    handle: String,
}

#[derive(Deserialize)]
struct ExtractedTweet {
    id: String,
    account_id: String,
    text: String,
    posted_at: Option<DateTime<Utc>>,
}

/// Searches X through self-hosted Firecrawl.
#[derive(Debug, Clone)]
pub struct Firecrawl {
    client: reqwest::Client,
    url: String,
}

impl Firecrawl {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: url.into(),
        }
    }
}

impl Scraper for Firecrawl {
    async fn scrape(&self, query: &str) -> Result<Scraped, Error> {
        let body = json!({
            "query": format!("site:x.com {query}"),
            "scrapeOptions": {
                "formats": [{
                    "type": "json",
                    "prompt": PROMPT,
                    "schema": extraction_schema(),
                }],
            },
        });
        let response: SearchResponse = self
            .client
            .post(format!("{}/v2/search", self.url.trim_end_matches('/')))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        map(response)
    }
}

/// Keeps the first object for each ID and stores the full extracted object in `raw`.
fn map(response: SearchResponse) -> Result<Scraped, Error> {
    let mut accounts = Vec::new();
    let mut tweets = Vec::new();
    let mut account_ids = HashSet::new();
    let mut tweet_ids = HashSet::new();
    for extracted in response.data.web.into_iter().filter_map(|hit| hit.json) {
        for raw in extracted.accounts {
            let account: ExtractedAccount = serde_json::from_value(raw.clone())?;
            if account_ids.insert(account.id.clone()) {
                accounts.push(Account {
                    id: account.id,
                    handle: account.handle,
                    raw,
                });
            }
        }
        for raw in extracted.tweets {
            let tweet: ExtractedTweet = serde_json::from_value(raw.clone())?;
            if tweet_ids.insert(tweet.id.clone()) {
                tweets.push(Tweet {
                    id: tweet.id,
                    account_id: tweet.account_id,
                    text: tweet.text,
                    posted_at: tweet.posted_at,
                    raw,
                });
            }
        }
    }
    if let Some(tweet) = tweets.iter().find(|t| !account_ids.contains(&t.account_id)) {
        return Err(Error::UnknownAccount {
            tweet_id: tweet.id.clone(),
            account_id: tweet.account_id.clone(),
        });
    }
    Ok(Scraped { accounts, tweets })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_partial_json, method, path},
    };

    use super::*;

    fn hit(accounts: Value, tweets: Value) -> Value {
        json!({ "url": "https://x.com/search", "json": { "accounts": accounts, "tweets": tweets } })
    }

    async fn server_returning(web: Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/search"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "success": true, "data": { "web": web } })),
            )
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn sends_search_request_with_json_extraction() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v2/search"))
            .and(body_partial_json(json!({
                "query": "site:x.com ai news",
                "scrapeOptions": { "formats": [{ "type": "json" }] },
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "success": true, "data": { "web": [] } })),
            )
            .expect(1)
            .mount(&server)
            .await;
        let scraped = Firecrawl::new(server.uri())
            .scrape("ai news")
            .await
            .unwrap();
        assert!(scraped.tweets.is_empty());
    }

    #[tokio::test]
    async fn maps_extracted_objects_and_keeps_them_in_raw() {
        let server = server_returning(json!([
            hit(
                json!([{ "id": "9000000001", "handle": "fixture_alpha", "bio": "extra" }]),
                json!([{ "id": "8000000001", "account_id": "9000000001", "text": "hello", "posted_at": "2026-01-01T10:00:00Z" }]),
            ),
            hit(
                json!([{ "id": "9000000001", "handle": "fixture_alpha" }]),
                json!([{ "id": "8000000001", "account_id": "9000000001", "text": "hello" }]),
            ),
            json!({ "url": "https://x.com/none", "json": null }),
        ]))
        .await;
        let scraped = Firecrawl::new(server.uri()).scrape("q").await.unwrap();
        assert_eq!(scraped.accounts.len(), 1);
        assert_eq!(scraped.accounts[0].handle, "fixture_alpha");
        assert_eq!(scraped.accounts[0].raw["bio"], "extra");
        assert_eq!(scraped.tweets.len(), 1);
        assert_eq!(scraped.tweets[0].text, "hello");
        assert_eq!(scraped.tweets[0].account_id, "9000000001");
        assert!(scraped.tweets[0].posted_at.is_some());
        assert_eq!(scraped.tweets[0].raw["id"], "8000000001");
    }

    #[tokio::test]
    async fn rejects_tweet_from_unknown_account() {
        let server = server_returning(json!([hit(
            json!([]),
            json!([{ "id": "8000000001", "account_id": "9000000009", "text": "hello" }]),
        )]))
        .await;
        let error = Firecrawl::new(server.uri()).scrape("q").await.unwrap_err();
        assert!(matches!(error, Error::UnknownAccount { .. }));
    }

    #[tokio::test]
    async fn fails_on_http_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let error = Firecrawl::new(server.uri()).scrape("q").await.unwrap_err();
        assert!(matches!(error, Error::Http(_)));
    }
}
