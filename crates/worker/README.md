# worker

`worker` takes jobs from SQS and runs one role: `scraper` or `classifier`.

## Where it sits

The API sends scraper jobs. The scraper worker sends classifier jobs. The classifier worker finishes the query.

```text
scraper-jobs ─▶ worker --role scraper ─▶ classifier-jobs ─▶ worker --role classifier
                     │                                            │
                     └──────────────▶ Postgres ◀──────────────────┘
```

The scraper role calls `scraper`, `db`, and `queue`. The classifier role calls `metadata`, `classifier`, `embedder`, and `db`. Both roles call `telemetry`.

## Terms

See [glossary.md](../../docs/glossary.md): worker, role, scraper, classifier, embedder, job, scraper job, classifier job, queue, dead-letter queue, DLQ reader, job deadline, receive count, trace context, batch, batch number, batch count, query status, upsert, fake, fixture.

## How it works

Scraper role, steps 5 to 9 of the request flow:

1. The scraper worker takes a scraper job and reads the query from Postgres.
2. If the query status is `classifying`, the scraper worker skips to step 5.
3. The scraper worker sets the query status to `scraping` and calls the scraper.
4. The scraper worker upserts the accounts, the tweets, and the `query_tweets` rows.
5. The scraper worker reads the tweet IDs, sorted. With zero tweets, it sets the status to `done` and stops.
6. The scraper worker splits the tweet IDs into batches of `BATCH_SIZE`, numbered from 0.
7. The scraper worker records the batch count and sets the status to `classifying` in one `UPDATE`. A redelivered job skips this write.
8. The scraper worker sends one classifier job for each batch.

Classifier role, steps 10 to 14:

1. The classifier worker takes a classifier job. If a `query_batches` row exists for the batch, the worker deletes the job and stops.
2. The classifier worker finds the tweets that have no prediction for the model version, `DEFAULT_METADATA_VERSION`, and `DEFAULT_QUESTION_VERSION`.
3. The classifier worker builds one model input for each of those tweets with `metadata::build`.
4. The classifier sends the model inputs to the inference backend. The embedder embeds the tweet text.
5. The classifier worker calls `db::finish_batch`. It commits the predictions, the embeddings, and the `query_batches` row in one transaction, and sets `done` after the last batch.

Failure flow, in both roles:

1. The worker runs each job under `tokio::time::timeout` of `JOB_DEADLINE_SECS`. A timeout counts as a failure.
2. The worker deletes the job after the handler returns, which is after Postgres commits.
3. A failure leaves the job in the queue. SQS delivers the job again after the visibility timeout.
4. A failure on receive 5 or later sets the query status to `failed` and records the error. The job stays in the queue, and SQS moves it to the dead-letter queue.
5. The DLQ reader takes each job from the role's dead-letter queue. It sets the query status to `failed` and deletes the job.
6. The worker joins the trace of the API with `Received::continue_trace`.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | none | Postgres address. |
| `SCRAPER_QUEUE_URL`, `SCRAPER_DLQ_URL` | none | The scraper role's queue and dead-letter queue. |
| `CLASSIFIER_QUEUE_URL`, `CLASSIFIER_DLQ_URL` | none | The classifier role's queue and dead-letter queue. The scraper role sends to `CLASSIFIER_QUEUE_URL`. |
| `SCRAPER` | `twitterapi` | Scraper implementation: `twitterapi` or `fixture`. |
| `TWITTERAPI_KEY` | none | The twitterapi.io key. Required with `SCRAPER=twitterapi`. |
| `TWITTERAPI_PAGES` | `5` | The most search pages for each query, 20 tweets each. |
| `TWITTERAPI_INTERVAL_SECS` | `5` | The pause between two requests. The free tier allows one request every 5 seconds. |
| `FIXTURE_PATH` | `crates/scraper/fixtures/default.json` | The file the `fixture` scraper reads. |
| `CLASSIFIER` | `hosted` | Classifier implementation: `hosted` or `fake`. |
| `EMBEDDER` | `hosted` | Embedder implementation: `hosted` or `fake`. |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | none | Read by the hosted classifier and the hosted embedder. |
| `CLASSIFIER_RPS` | `5` | Request limit of the hosted classifier. |
| `DEFAULT_METADATA_VERSION`, `DEFAULT_QUESTION_VERSION` | none | Versions the classifier role builds model inputs with. |
| `BATCH_SIZE` | none | Tweets in one batch. |
| `JOB_DEADLINE_SECS` | none | Seconds a worker gives one job. Keep it below the visibility timeout. |

## Run it

Run from the repository root, with `.env` loaded.

```bash
cargo run -p worker -- --role scraper
cargo run -p worker -- --role classifier
```

## Test it

The integration tests need Postgres and LocalStack from `docker compose up -d`. They use the fixture scraper, the fake classifier, and the fake embedder. Each test creates its own queues with a 1-second visibility timeout and `maxReceiveCount` 5, and deletes them at the end.

```bash
cargo test -p worker --features integration
```

The failure test waits for five receives, so it takes several seconds.

## Failure modes

| What breaks | What the worker does | What you do |
|-------------|----------------------|-------------|
| Postgres, SQS, the scraper, the classifier, or the embedder returns an error | The worker keeps the job. SQS delivers it again. | Read the `job failed` log line. |
| A job runs past `JOB_DEADLINE_SECS` | The worker counts a failure. | Raise the deadline, or find the slow call. |
| The fifth receive fails | The worker sets the query status to `failed` with the error. | Read `queries.error`. |
| The worker crashes on the fifth receive | SQS moves the job to the dead-letter queue. The DLQ reader sets `failed`. | Restart the worker. |
| The scraper job names a query that does not exist | The job fails on every receive and reaches the dead-letter queue. | Delete the job from the dead-letter queue. |
| The worker crashes after `db::finish_batch` and before the delete | SQS delivers the job again. The `query_batches` row makes the repeat a no-op. | None. |
| The scraper worker crashes between the batch count write and the last send | SQS delivers the scraper job again. The worker sees `classifying` and sends every classifier job again. | None. |
| `BATCH_SIZE` changes while a query is `classifying` | A redelivered scraper job builds batches of the new size. The batch numbers can then mismatch the recorded batch count. | Keep `BATCH_SIZE` fixed while queries run. |
| A query is already `done` or `failed` | The worker deletes the job and changes nothing. | None. |
