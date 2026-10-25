use aws_sdk_sqs::{
    Client,
    types::{MessageSystemAttributeName, QueueAttributeName},
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

    /// Long-polls for up to `wait_seconds` and returns at most 10 jobs.
    // LocalStack ignores the newer `message_system_attribute_names`.
    #[allow(deprecated)]
    pub async fn receive<J: Job>(&self, wait_seconds: i32) -> Result<Vec<Received<J>>, Error> {
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

        output
            .messages
            .unwrap_or_default()
            .into_iter()
            .map(|message| {
                let body = message.body.ok_or(Error::MissingField("body"))?;
                let receive_count = message
                    .attributes
                    .as_ref()
                    .and_then(|attributes| {
                        attributes.get(&MessageSystemAttributeName::ApproximateReceiveCount)
                    })
                    .and_then(|count| count.parse().ok())
                    .ok_or(Error::MissingField("receive count"))?;
                Ok(Received {
                    job: serde_json::from_str(&body)?,
                    receive_count,
                    receipt_handle: message
                        .receipt_handle
                        .ok_or(Error::MissingField("receipt handle"))?,
                })
            })
            .collect()
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
