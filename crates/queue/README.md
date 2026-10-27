# queue

`queue` sends, receives, and deletes typed jobs on an SQS queue, and carries the trace context inside each job.

## Where it sits

The API sends scraper jobs. The scraper worker receives scraper jobs and sends classifier jobs. The classifier worker receives classifier jobs. `queue` depends on `types` and `telemetry`.

## Terms

See [glossary.md](../../docs/glossary.md): queue, job, scraper job, classifier job, dead-letter queue, worker.

## How it works

1. The caller reads the queue URL from `SCRAPER_QUEUE_URL` or `CLASSIFIER_QUEUE_URL` and calls `Queue::connect(url)`.
2. `send(job)` writes the current `traceparent` into the job's `trace_context`. It then sends the job as JSON.
3. `receive::<J>(wait_seconds)` long-polls the queue. It parses each message on its own and returns a `Batch` of up to 10 messages. `Batch.jobs` holds the valid `Received` values: the job, the receive count, and the receipt handle. `Batch.broken` holds each message that is not a valid job: the receipt handle, the message ID, and the parse error. `Batch` dereferences to `jobs`.
4. The receive count comes from the `ApproximateReceiveCount` attribute. It is 1 on the first receive and grows each time the visibility timeout expires without a delete. A message without the attribute gets count 1.
5. The worker calls `received.continue_trace(&span)`. The span then joins the trace that the sender started.
6. The worker calls `delete(&received.receipt_handle)` after the job succeeds.
7. The API calls `ping()` for its readiness check. `ping` succeeds when SQS answers for the queue.

`ScraperJob` and `ClassifierJob` implement the `Job` trait. A new job type needs the same two methods.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `AWS_ENDPOINT_URL` | unset | Points the client at LocalStack, for example `http://localhost:4566`. |
| `AWS_REGION` | none | The AWS region. |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | none | The credentials. LocalStack accepts any value. |

`aws-config` reads these variables. `queue` has no endpoint code. The queue URL comes from the caller.

## Run it

`queue` is a library. It has no binary.

## Test it

From the repository root:

```bash
cargo test -p queue
```

This command needs no Docker. The integration tests need LocalStack from `docker compose up -d --wait`:

```bash
AWS_ENDPOINT_URL=http://localhost:4566 AWS_REGION=us-east-1 \
AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test \
cargo test -p queue --features integration
```

Each integration test creates a queue with a unique name and deletes the queue at the end. The tests never touch `scraper-jobs` or `classifier-jobs`.

## Failure modes

- A failed SQS call returns `Error::Sqs`. The caller decides whether to retry.
- A message that is not valid JSON for its type, or has no body, goes to `Batch.broken`. The other messages of the receive stay in `Batch.jobs`. The caller decides what to do with a broken message.
- A message without a receipt handle returns `Error::MissingField`, because nobody can delete it.
- A worker that crashes before it deletes a job gets the job again. The receive count rises by one.
- `receive` asks for `ApproximateReceiveCount` with the older `AttributeNames` field. LocalStack ignores the newer `MessageSystemAttributeNames` field.
- The AWS SDK uses rustls with `aws-lc-rs`, which builds a C library. No pure Rust provider is ready for production.
