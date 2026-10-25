use std::{future::Future, time::Duration};

use rand::Rng;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

/// The length of every embedding.
pub const DIMENSIONS: usize = 384;

const MODEL: &str = "@cf/baai/bge-small-en-v1.5";
const BASE_URL: &str = "https://api.cloudflare.com";
const MAX_ATTEMPTS: u32 = 5;
const BACKOFF_BASE: Duration = Duration::from_millis(500);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Workers AI request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Workers AI answered with status {0}")]
    Status(StatusCode),
    #[error("Workers AI returned {got} embeddings of {dimensions} numbers for {expected} texts")]
    Shape {
        expected: usize,
        got: usize,
        dimensions: usize,
    },
    #[error("environment variable {0} is not set")]
    MissingVariable(&'static str),
}

pub trait Embedder {
    /// Returns one embedding for each text, in order.
    fn embed(&self, texts: &[String]) -> impl Future<Output = Result<Vec<Vec<f32>>, Error>> + Send;
}

/// The hosted backend: the Workers AI embedding model.
pub struct Hosted {
    client: reqwest::Client,
    url: String,
    api_token: String,
    backoff_base: Duration,
}

#[derive(Serialize)]
struct Request<'a> {
    text: &'a [String],
}

#[derive(Deserialize)]
struct Envelope {
    result: Output,
}

#[derive(Deserialize)]
struct Output {
    data: Vec<Vec<f32>>,
}

impl Hosted {
    pub fn new(base_url: &str, account_id: &str, api_token: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            url: format!("{base_url}/client/v4/accounts/{account_id}/ai/run/{MODEL}"),
            api_token: api_token.to_owned(),
            backoff_base: BACKOFF_BASE,
        }
    }

    /// Reads `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.
    pub fn from_env() -> Result<Self, Error> {
        let var = |name| std::env::var(name).map_err(|_| Error::MissingVariable(name));
        Ok(Self::new(
            BASE_URL,
            &var("CLOUDFLARE_ACCOUNT_ID")?,
            &var("CLOUDFLARE_API_TOKEN")?,
        ))
    }

    /// Sets the base delay of the retry backoff.
    pub fn with_backoff_base(mut self, backoff_base: Duration) -> Self {
        self.backoff_base = backoff_base;
        self
    }

    async fn send(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        let response = self
            .client
            .post(&self.url)
            .bearer_auth(&self.api_token)
            .json(&Request { text: texts })
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(Error::Status(status));
        }
        Ok(response.json::<Envelope>().await?.result.data)
    }
}

impl Embedder for Hosted {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        let mut attempt = 0;
        let data = loop {
            attempt += 1;
            match self.send(texts).await {
                Err(Error::Status(status))
                    if attempt < MAX_ATTEMPTS
                        && (status == StatusCode::TOO_MANY_REQUESTS
                            || status.is_server_error()) =>
                {
                    let ceiling = self.backoff_base * 2u32.pow(attempt - 1);
                    let delay = rand::rng().random_range(Duration::ZERO..=ceiling);
                    tokio::time::sleep(delay).await;
                }
                result => break result?,
            }
        };
        if data.len() != texts.len() || data.iter().any(|row| row.len() != DIMENSIONS) {
            return Err(Error::Shape {
                expected: texts.len(),
                got: data.len(),
                dimensions: data.first().map_or(0, Vec::len),
            });
        }
        Ok(data)
    }
}

/// A test embedder. It derives each embedding from the text and makes no network calls.
pub struct Fake;

impl Embedder for Fake {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        Ok(texts.iter().map(|text| fake_embedding(text)).collect())
    }
}

/// Seeds SplitMix64 with the FNV-1a hash of the text.
fn fake_embedding(text: &str) -> Vec<f32> {
    let mut state = text.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    (0..DIMENSIONS)
        .map(|_| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, header, method, path},
    };

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|text| text.to_string()).collect()
    }

    fn body(rows: usize) -> serde_json::Value {
        json!({
            "result": { "shape": [rows, DIMENSIONS], "data": vec![vec![0.5; DIMENSIONS]; rows], "pooling": "mean" },
            "success": true, "errors": [], "messages": []
        })
    }

    fn hosted(server: &MockServer) -> Hosted {
        Hosted::new(&server.uri(), "acct", "token").with_backoff_base(Duration::from_millis(1))
    }

    #[tokio::test]
    async fn sends_the_request_and_parses_the_embeddings() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/client/v4/accounts/acct/ai/run/@cf/baai/bge-small-en-v1.5",
            ))
            .and(header("authorization", "Bearer token"))
            .and(body_json(json!({ "text": ["a", "b"] })))
            .respond_with(ResponseTemplate::new(200).set_body_json(body(2)))
            .expect(1)
            .mount(&server)
            .await;

        let embeddings = hosted(&server).embed(&texts(&["a", "b"])).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert!(embeddings.iter().all(|row| row.len() == DIMENSIONS));
    }

    #[tokio::test]
    async fn retries_429_and_5xx_then_succeeds() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body(1)))
            .mount(&server)
            .await;

        let embeddings = hosted(&server).embed(&texts(&["a"])).await.unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(server.received_requests().await.unwrap().len(), 4);
    }

    #[tokio::test]
    async fn gives_up_after_five_attempts() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(500))
            .expect(5)
            .mount(&server)
            .await;

        let error = hosted(&server).embed(&texts(&["a"])).await.unwrap_err();

        assert!(matches!(
            error,
            Error::Status(StatusCode::INTERNAL_SERVER_ERROR)
        ));
    }

    #[tokio::test]
    async fn does_not_retry_other_client_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1)
            .mount(&server)
            .await;

        let error = hosted(&server).embed(&texts(&["a"])).await.unwrap_err();

        assert!(matches!(error, Error::Status(StatusCode::UNAUTHORIZED)));
    }

    #[tokio::test]
    async fn rejects_a_wrong_number_of_embeddings() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body(1)))
            .mount(&server)
            .await;

        let error = hosted(&server)
            .embed(&texts(&["a", "b"]))
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            Error::Shape {
                expected: 2,
                got: 1,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn fake_is_deterministic_and_has_the_right_length() {
        let embeddings = Fake.embed(&texts(&["a", "b", "a"])).await.unwrap();

        assert!(embeddings.iter().all(|row| row.len() == DIMENSIONS));
        assert_eq!(embeddings[0], embeddings[2]);
        assert_ne!(embeddings[0], embeddings[1]);
        assert_eq!(embeddings[0], Fake.embed(&texts(&["a"])).await.unwrap()[0]);
    }
}
