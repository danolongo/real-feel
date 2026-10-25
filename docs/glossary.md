# Glossary

This file defines every Real Feel term. Each term has one definition and one part of speech.

- Use the term exactly as written here.
- Do not use the words in the "Do not use" column.
- Add a new term here in the same PR that introduces it.
- Name code after these terms. Write `struct Query`, not `struct Search`.

See [writing-rules.md](writing-rules.md) for the full rules.

## People and roles

| Term | Part of speech | Definition | Do not use |
|------|----------------|------------|------------|
| you | pronoun | The developer who reads a Real Feel document and does the work it describes. | we, one, the developer |
| user | noun | A person who sends queries to Real Feel. | customer, client |
| reviewer | noun | The developer who approves a PR. | approver |

## Data

| Term | Part of speech | Definition | Do not use |
|------|----------------|------------|------------|
| query | noun | A natural-language question that a user sends to the API. Example: "what are people saying about AI?" | search, request, prompt |
| query status | noun | The stage a query is in. Values: `queued`, `scraping`, `classifying`, `done`, `failed`. | state, progress |
| tweet | noun | One post on X. | post, status |
| account | noun | The X profile that posts a tweet. | user, profile, author |
| metadata | noun | The account fields and tweet fields, other than tweet text, that a model input carries. Examples: account age, follower count, posting times. | features, attributes, context |
| metadata version | noun | An identifier (`v1`, `v2`, …) for one exact set of metadata fields and one exact text layout. | schema, shape, feature set |
| question version | noun | An identifier (`v1`, `v2`, …) for one exact wording of the question that Clef-flash answers. | prompt version |
| model input | noun | Everything the classifier sends to Clef-flash for one tweet: tweet text, metadata, and the question. | payload, prompt |
| bot | noun | An account that software runs. | fake account |
| human | noun | An account that a person runs. | real account, organic account |
| bot probability | noun | A number from 0 to 1. It says how likely Clef-flash thinks the account behind a tweet is a bot. Field name: `p_bot`. | bot score, confidence, score |
| bot threshold | noun | The bot probability at or above which the API reports an account as a bot. Field name: `is_bot`. | cutoff |
| prediction | noun | The stored output for one model input: bot probability, model version, metadata version, question version. | decision, score, classification |
| result | noun | What the API returns for a query whose status is `done`: tweets, accounts, and predictions. | response, output |
| embedding | noun | A vector of numbers that captures the meaning of tweet text. Postgres stores it with pgvector. | vector, encoding |
| label | noun | The known truth for one eval set item: `bot` or `human`. Never a verb. | ground truth, tag |

## Components

| Term | Part of speech | Definition | Do not use |
|------|----------------|------------|------------|
| API | noun | The Rust HTTP server in `crates/api`. It accepts queries and returns results. | backend, server, service |
| worker | noun | A Rust process in `crates/worker`. It takes jobs from SQS. It runs one role: `scraper` or `classifier`. | consumer, processor |
| scraper | noun | The worker part that fetches tweets and accounts from X. Code: `trait Scraper`. | crawler, fetcher |
| classifier | noun | The worker part that sends batches to an inference backend and returns predictions. Code: `trait Classifier`. | detector, model client |
| embedder | noun | The worker part that turns tweet text into embeddings. Code: `trait Embedder`. | encoder |
| inference backend | noun | Any system that runs Clef-flash for the classifier. Kinds: hosted backend, self-hosted backend, fake. | provider, model server |
| hosted backend | noun | Clef-flash running on Cloudflare Workers AI. Cloudflare runs the GPUs. | Cloudflare API, managed model |
| self-hosted backend | noun | Clef-flash running on an inference server on a Real Feel GPU instance. | local model, own model |
| inference server | noun | An HTTP server on a GPU instance that serves Clef-flash. Kinds: reference server, candle server. | model server, inference service |
| reference server | noun | The Python inference server in `inference/reference`. It defines correct output. | Python server, oracle |
| candle server | noun | The Rust inference server in `crates/inference`. It uses the `candle` library. | Rust server |
| Postgres | noun | The single database. It stores queries, tweets, accounts, predictions, and embeddings. On AWS, Postgres runs on RDS. | DB, the database |
| queue | noun | One SQS queue. Never a verb. Queues: `scraper-jobs`, `classifier-jobs`, and one dead-letter queue for each. | topic, stream |
| job | noun | The content of one SQS message. Kinds: scraper job, classifier job. | task, event, message |
| scraper job | noun | A job that tells a scraper worker to fetch tweets for one query. | scrape job, crawl job |
| classifier job | noun | A job that tells a classifier worker to classify one batch of tweets. | classify job |
| dead-letter queue | noun | The queue that receives a job after the job fails too many times. Short form: DLQ. | error queue |
| visibility timeout | noun | The time SQS hides a received job from other workers. After it ends, SQS delivers the job again. | lock time |
| job deadline | noun | The time a worker gives one job before it abandons the job. `JOB_DEADLINE_SECS` sets it. | time limit |
| batch number | noun | The position of one batch in its query, counted from 0. Field name: `batch_number`. | batch index, batch ID |
| batch count | noun | How many batches a query has. The scraper worker records it once. Field name: `batch_count`. | batch total |
| batch | noun | An ordered list of model inputs that the classifier sends to an inference backend in one HTTP request. Never a verb. | chunk, group |
| dynamic batching | noun | An inference server collecting HTTP requests for a short time and running them on the GPU together. | micro-batching |
| fake | noun | A test version of a trait that returns fixed output and makes no network calls. Never an adjective. | mock, stub, dummy |
| fixture | noun | Stored test data that replaces a network response. | sample data, mock data |
| upsert | verb | To insert a row, or to update the row when its key exists and its content differs. A second upsert with the same input changes no row. | insert-or-update, merge |
| repository function | noun | A function in `crates/db` that runs SQL. No other code runs SQL. | query function, data access function |
| receive count | noun | How many times SQS has delivered one job. The worker reads it from `ApproximateReceiveCount`. | delivery count, attempt count |
| trace context | noun | The W3C `traceparent` value that a job carries, so the spans of the API and the worker join one trace. | trace header, trace ID |
| page | noun | One slice of a result that the API returns for one request. | slice, section |
| cursor | noun | The last `tweet_id` of the previous page. You pass it as `after` to get the next page. | offset, page token |
| liveness check | noun | The `GET /health` route. It returns `200` while the process runs and checks no dependency. | health probe |
| readiness check | noun | The `GET /ready` route. It returns `200` when Postgres and SQS respond, and `503` when either does not. | ready probe |
| error code | noun | The `code` field of an API error response, such as `invalid_query`. | error type, error name |

## Models

| Term | Part of speech | Definition | Do not use |
|------|----------------|------------|------------|
| Clef-flash | noun | Cloudflare's 9B decision model. Real Feel uses it to classify tweets. | the LLM, the model |
| Clef | noun | Cloudflare's 27B decision model. Real Feel does not use it. | — |
| variant | noun | One numeric precision of Clef-flash's weights. Variants: `bf16`, `int8`, `int4`. | build, quant, version |
| model version | noun | An identifier for one inference backend running one variant. Examples: `clef-flash-hosted`, `clef-flash-int8-reference`, `clef-flash-int8-candle`. | model id, checkpoint |
| embedding model | noun | The model that the embedder calls. | encoder |

## Quality and process

| Term | Part of speech | Definition | Do not use |
|------|----------------|------------|------------|
| eval set | noun | A fixed collection of tweets and accounts, each with a label. | test set, benchmark data |
| experiment | noun | One pass of the eval set through one model version, one metadata version, and one question version. | eval run, trial |
| eval report | noun | A Markdown file in `docs/reports/` that compares experiments. | results, writeup |
| calibration | noun | How closely bot probability matches reality. A calibrated 0.8 is right 80% of the time. | confidence accuracy |
| benchmark | noun | A fixed measurement of inference server speed: throughput and latency at set batch sizes. Never a verb. | perf test |
| load test | noun | A k6 script that sends HTTP requests to the API at a set rate. | stress test, perf test |
| break point | noun | The request rate at which a load test first fails its targets. | limit, ceiling |
| milestone | noun | A numbered stage of the refactor plan (M0–M8). | phase, sprint |
| gate | noun | The checklist that must hold true before you close a milestone. | exit criteria, acceptance |
| spike | noun | Short, time-boxed research that answers one question and ends in a decision. | investigation, POC |
| decision record | noun | A file in `docs/decisions/` that records one decision and its reasons. Short form: ADR. | design doc |
