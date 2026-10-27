use std::ops::{Deref, DerefMut};

use aws_sdk_sqs::{
    Client,
    types::{Message, MessageSystemAttributeName, QueueAttributeName},
};
use serde::{Serialize, de::DeserializeOwned};
use tracing::Span;
use types::{ClassifierJob, ScraperJob};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("SQS request failed: {0}")]
    Sqs(Box<aws_sdk_sqs::Error>),
    #[error("job is not valid JSON for its type: {0}")]
    Json(#[from] serde_json::Error),
    #[error("SQS message lacks its {0}")]
    MissingField(&'static str),
}

/// A job type that carries a trace context.
pub trait Job: Serialize + DeserializeOwned {
    fn trace_context(&self) -> Option<&str>;
    fn set_trace_context(&mut self, trace_context: Option<String>);
}

impl Job for ScraperJob {
    fn trace_context(&self) -> Option<&str> {
        self.trace_context.as_deref()
    }

    fn set_trace_context(&mut self, trace_context: Option<String>) {
        self.trace_context = trace_context;
    }
}

impl Job for ClassifierJob {
    fn trace_context(&self) -> Option<&str> {
        self.trace_context.as_deref()
    }

    fn set_trace_context(&mut self, trace_context: Option<String>) {
        self.trace_context = trace_context;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Received<J> {
    pub job: J,
    pub receive_count: u32,
    pub receipt_handle: String,
}

impl<J: Job> Received<J> {
    /// Makes `span` continue the trace that the sender wrote into the job.
    pub fn continue_trace(&self, span: &Span) {
        if let Some(trace_context) = self.job.trace_context() {
            telemetry::continue_trace(span, trace_context);
        }
    }
}

/// A message that is not a valid job. Its receipt handle lets the caller delete it.
#[derive(Debug, Clone, PartialEq)]
pub struct Broken {
    pub receipt_handle: String,
    pub message_id: Option<String>,
    pub error: String,
}

/// The messages of one receive. A broken message never hides the valid jobs next to it.
/// Dereferences to `jobs`.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch<J> {
    pub jobs: Vec<Received<J>>,
    pub broken: Vec<Broken>,
}

impl<J> Deref for Batch<J> {
    type Target = Vec<Received<J>>;

    fn deref(&self) -> &Self::Target {
        &self.jobs
    }
}

impl<J> DerefMut for Batch<J> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.jobs
    }
}

/// A message without a receipt handle returns an error: nobody can delete it.
/// A message without a receive count gets count 1.
fn parse_message<J: Job>(message: Message) -> Result<Result<Received<J>, Broken>, Error> {
    let receipt_handle = message
        .receipt_handle
        .ok_or(Error::MissingField("receipt handle"))?;
    let parsed = message
        .body
        .ok_or_else(|| "message has no body".to_string())
        .and_then(|body| serde_json::from_str(&body).map_err(|error| error.to_string()));
    Ok(match parsed {
        Ok(job) => Ok(Received {
            job,
            receive_count: message
                .attributes
                .as_ref()
                .and_then(|attributes| {
                    attributes.get(&MessageSystemAttributeName::ApproximateReceiveCount)
                })
                .and_then(|count| count.parse().ok())
                .unwrap_or(1),
            receipt_handle,
        }),
        Err(error) => Err(Broken {
            receipt_handle,
            message_id: message.message_id,
            error,
        }),
    })
}

#[derive(Debug, Clone)]
pub struct Queue {
    client: Client,
    url: String,
}

impl Queue {
    /// `url` is the queue URL. `AWS_ENDPOINT_URL` redirects the client to LocalStack.
    pub async fn connect(url: impl Into<String>) -> Self {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Self {
            client: Client::new(&config),
            url: url.into(),
        }
    }

    /// Writes the current trace context into `job`, then sends it.
    pub async fn send<J: Job>(&self, mut job: J) -> Result<(), Error> {
        job.set_trace_context(telemetry::current_traceparent());
        self.client
            .send_message()
            .queue_url(&self.url)
            .message_body(serde_json::to_string(&job)?)
            .send()
            .await
            .map_err(|error| Error::Sqs(Box::new(error.into())))?;
        Ok(())
    }

    /// Long-polls for up to `wait_seconds` and returns at most 10 messages.
    /// Parses each message on its own and puts the ones that are not valid jobs in `broken`.
    // LocalStack ignores the newer `message_system_attribute_names`.
    #[allow(deprecated)]
    pub async fn receive<J: Job>(&self, wait_seconds: i32) -> Result<Batch<J>, Error> {
        let output = self
            .client
            .receive_message()
            .queue_url(&self.url)
            .max_number_of_messages(10)
            .wait_time_seconds(wait_seconds)
            .attribute_names(QueueAttributeName::from("ApproximateReceiveCount"))
            .send()
            .await
            .map_err(|error| Error::Sqs(Box::new(error.into())))?;

        let mut batch = Batch {
            jobs: Vec::new(),
            broken: Vec::new(),
        };
        for message in output.messages.unwrap_or_default() {
            match parse_message(message)? {
                Ok(received) => batch.jobs.push(received),
                Err(broken) => batch.broken.push(broken),
            }
        }
        Ok(batch)
    }

    /// Succeeds when SQS answers for this queue.
    pub async fn ping(&self) -> Result<(), Error> {
        self.client
            .get_queue_attributes()
            .queue_url(&self.url)
            .send()
            .await
            .map_err(|error| Error::Sqs(Box::new(error.into())))?;
        Ok(())
    }

    pub async fn delete(&self, receipt_handle: &str) -> Result<(), Error> {
        self.client
            .delete_message()
            .queue_url(&self.url)
            .receipt_handle(receipt_handle)
            .send()
            .await
            .map_err(|error| Error::Sqs(Box::new(error.into())))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;

    fn message(body: Option<&str>, receipt_handle: Option<&str>) -> Message {
        Message::builder()
            .set_body(body.map(Into::into))
            .set_receipt_handle(receipt_handle.map(Into::into))
            .message_id("m-1")
            .attributes(MessageSystemAttributeName::ApproximateReceiveCount, "3")
            .build()
    }

    #[test]
    fn splits_a_valid_message_from_an_invalid_one() {
        let query_id = Uuid::nil();
        let valid = format!(r#"{{"query_id":"{query_id}","trace_context":null}}"#);

        let good = parse_message::<ScraperJob>(message(Some(&valid), Some("r-1")))
            .unwrap()
            .unwrap();
        assert_eq!(good.job.query_id, query_id);
        assert_eq!(good.receive_count, 3);
        assert_eq!(good.receipt_handle, "r-1");

        let bad = parse_message::<ScraperJob>(message(Some("not json"), Some("r-2")))
            .unwrap()
            .unwrap_err();
        assert_eq!(bad.receipt_handle, "r-2");
        assert_eq!(bad.message_id.as_deref(), Some("m-1"));
        assert!(!bad.error.is_empty());
    }

    #[test]
    fn a_message_without_a_body_is_broken() {
        let bad = parse_message::<ScraperJob>(message(None, Some("r-1")))
            .unwrap()
            .unwrap_err();
        assert_eq!(bad.error, "message has no body");
    }

    #[test]
    fn a_message_without_a_receipt_handle_is_an_error() {
        let result = parse_message::<ScraperJob>(message(Some("{}"), None));
        assert!(matches!(result, Err(Error::MissingField("receipt handle"))));
    }

    #[test]
    fn a_missing_receive_count_becomes_one() {
        let query_id = Uuid::nil();
        let message = Message::builder()
            .body(format!(
                r#"{{"query_id":"{query_id}","trace_context":null}}"#
            ))
            .receipt_handle("r-1")
            .build();
        let received = parse_message::<ScraperJob>(message).unwrap().unwrap();
        assert_eq!(received.receive_count, 1);
    }
}
