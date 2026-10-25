use std::env;
use std::sync::Mutex;
use std::time::Duration;

use futures::future::try_join_all;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::{Instant, sleep};
use types::{ModelInput, ModelVersion, Prediction};

use crate::{Classifier, Error};

pub const CLOUDFLARE_URL: &str = "https://api.cloudflare.com";
const MAX_ATTEMPTS: u32 = 5;
const BACKOFF_CAP: Duration = Duration::from_secs(8);
const QUESTION_ID: &str = "bot";

/// Clef-flash on Cloudflare Workers AI.
pub struct Hosted {
    http: reqwest::Client,
    url: String,
    api_token: String,
    interval: Duration,
    next_slot: Mutex<Instant>,
    backoff_base: Duration,
}

#[derive(Deserialize)]
struct Envelope {
    result: Value,
}

#[derive(Deserialize)]
struct Output {
    answers: Answers,
}

#[derive(Deserialize)]
struct Answers {
    bot: Noul,
}

#[derive(Deserialize)]
struct Noul {
    noul: f64,
}

impl Hosted {
    pub fn new(base_url: &str, account_id: &str, api_token: &str, rps: u32) -> Self {
        Self {
            http: reqwest::Client::new(),
            url: format!(
                "{base_url}/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clef-flash"
            ),
            api_token: api_token.into(),
            interval: Duration::from_secs(1) / rps,
            next_slot: Mutex::new(Instant::now()),
            backoff_base: Duration::from_millis(500),
        }
    }

    /// Reads `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, and `CLASSIFIER_RPS` (default 5).
    pub fn from_env() -> Self {
        let rps = env::var("CLASSIFIER_RPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(5);
        Self::new(
            CLOUDFLARE_URL,
            &env::var("CLOUDFLARE_ACCOUNT_ID").expect("CLOUDFLARE_ACCOUNT_ID is not set"),
            &env::var("CLOUDFLARE_API_TOKEN").expect("CLOUDFLARE_API_TOKEN is not set"),
            rps,
        )
    }

    #[cfg(test)]
    fn with_backoff_base(mut self, base: Duration) -> Self {
        self.backoff_base = base;
        self
    }

    /// Waits for the next free slot. Concurrent calls take consecutive slots.
    async fn wait_for_slot(&self) {
        let slot = {
            let mut next = self.next_slot.lock().unwrap();
            let slot = (*next).max(Instant::now());
            *next = slot + self.interval;
            slot
        };
        sleep(slot.saturating_duration_since(Instant::now())).await;
    }

    /// Waits a random time between 0 and the exponential ceiling.
    async fn back_off(&self, attempt: u32) {
        let ceiling = (self.backoff_base * 2u32.pow(attempt - 1)).min(BACKOFF_CAP);
        sleep(ceiling.mul_f64(fastrand::f64())).await;
    }

    #[tracing::instrument(name = "clef_flash_request", skip_all, fields(tweet_id = %input.tweet_id))]
    async fn classify_one(&self, input: &ModelInput) -> Result<Prediction, Error> {
        let body = json!({
            "model": "clef-flash",
            "state": input.text,
            "questions": {
                QUESTION_ID: { "type": "noul", "instructions": input.question },
            },
        });
        let mut attempt = 1;
        let result = loop {
            self.wait_for_slot().await;
            let response = self
                .http
                .post(&self.url)
                .bearer_auth(&self.api_token)
                .json(&body)
                .send()
                .await?;
            let status = response.status();
            if status.is_success() {
                break response.json::<Envelope>().await?.result;
            }
            let retryable = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
            if !retryable || attempt == MAX_ATTEMPTS {
                tracing::warn!(
                    status = status.as_u16(),
                    attempt,
                    "Clef-flash request failed"
                );
                return Err(Error::Status(status.as_u16()));
            }
            tracing::warn!(
                status = status.as_u16(),
                attempt,
                "Clef-flash answered an error, sending the request again"
            );
            self.back_off(attempt).await;
            attempt += 1;
        };
        let output: Output =
            serde_json::from_value(result.clone()).map_err(|e| Error::Response(e.to_string()))?;
        tracing::info!(
            p_bot = output.answers.bot.noul,
            attempts = attempt,
            "Clef-flash answered"
        );
        Ok(Prediction {
            tweet_id: input.tweet_id.clone(),
            model_version: self.model_version(),
            metadata_version: input.metadata_version.clone(),
            question_version: input.question_version.clone(),
            p_bot: output.answers.bot.noul,
            raw: result,
        })
    }
}

impl Classifier for Hosted {
    fn model_version(&self) -> ModelVersion {
        ModelVersion("clef-flash-hosted".into())
    }

    /// Cloudflare takes one state for each request, so this sends one request for each model input.
    #[tracing::instrument(name = "clef_flash_classify", skip_all, fields(inputs = inputs.len()))]
    async fn classify(&self, inputs: &[ModelInput]) -> Result<Vec<Prediction>, Error> {
        try_join_all(inputs.iter().map(|input| self.classify_one(input))).await
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use types::{MetadataVersion, QuestionVersion};
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    const PATH: &str = "/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash";

    fn input(tweet_id: &str) -> ModelInput {
        ModelInput {
            tweet_id: tweet_id.into(),
            metadata_version: MetadataVersion("v1".into()),
            question_version: QuestionVersion("v1".into()),
            text: format!("text {tweet_id}"),
            question: "Is this account a bot?".into(),
        }
    }

    fn success(p: f64) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({
            "success": true,
            "result": {
                "model": "clef-flash",
                "answers": { "bot": { "type": "noul", "noul": p } },
                "usage": { "input_tokens": 10, "output_tokens": 0 },
            },
        }))
    }

    fn hosted(server: &MockServer, rps: u32) -> Hosted {
        Hosted::new(&server.uri(), "acct", "token", rps).with_backoff_base(Duration::from_millis(1))
    }

    #[tokio::test]
    async fn sends_the_documented_request_and_parses_the_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(PATH))
            .and(header("authorization", "Bearer token"))
            .and(body_json(json!({
                "model": "clef-flash",
                "state": "text 1",
                "questions": { "bot": { "type": "noul", "instructions": "Is this account a bot?" } },
            })))
            .respond_with(success(0.75))
            .expect(1)
            .mount(&server)
            .await;

        let predictions = hosted(&server, 100).classify(&[input("1")]).await.unwrap();

        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0].tweet_id, "1");
        assert_eq!(predictions[0].p_bot, 0.75);
        assert_eq!(
            predictions[0].model_version,
            ModelVersion("clef-flash-hosted".into())
        );
        assert_eq!(predictions[0].raw["answers"]["bot"]["noul"], 0.75);
    }

    #[tokio::test]
    async fn keeps_the_order_of_the_inputs() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(success(0.5))
            .mount(&server)
            .await;

        let predictions = hosted(&server, 100)
            .classify(&[input("3"), input("1"), input("2")])
            .await
            .unwrap();

        let ids: Vec<_> = predictions.iter().map(|p| p.tweet_id.as_str()).collect();
        assert_eq!(ids, ["3", "1", "2"]);
    }

    #[tokio::test]
    async fn retries_429_and_503_then_succeeds() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1)
            .with_priority(2)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .respond_with(success(0.2))
            .with_priority(3)
            .mount(&server)
            .await;

        let predictions = hosted(&server, 100).classify(&[input("1")]).await.unwrap();

        assert_eq!(predictions[0].p_bot, 0.2);
        assert_eq!(server.received_requests().await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn stops_after_5_attempts() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let result = hosted(&server, 100).classify(&[input("1")]).await;

        assert!(matches!(result, Err(Error::Status(503))));
        assert_eq!(server.received_requests().await.unwrap().len(), 5);
    }

    #[tokio::test]
    async fn does_not_retry_other_4xx() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let result = hosted(&server, 100).classify(&[input("1")]).await;

        assert!(matches!(result, Err(Error::Status(401))));
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn rejects_a_response_without_a_bot_probability() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "result": { "answers": {} } })),
            )
            .mount(&server)
            .await;

        let result = hosted(&server, 100).classify(&[input("1")]).await;

        assert!(matches!(result, Err(Error::Response(_))));
    }

    #[tokio::test]
    async fn stays_under_the_requests_per_second_limit() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(success(0.5))
            .mount(&server)
            .await;
        let inputs: Vec<_> = (0..6).map(|i| input(&i.to_string())).collect();

        let start = std::time::Instant::now();
        hosted(&server, 10).classify(&inputs).await.unwrap();

        // 6 requests at 10 per second leave 5 gaps of 100 ms.
        assert!(start.elapsed() >= Duration::from_millis(500));
    }
}
