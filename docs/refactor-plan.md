# Refactor Plan

This plan rebuilds Real Feel as an all-Rust system on AWS.

- Milestones break the work into stages.
- Each milestone ends with a gate. You close a milestone only when every gate item holds true.
- This plan follows [writing-rules.md](writing-rules.md). [glossary.md](glossary.md) defines every term.

## Contents

1. [Goal](#goal)
2. [Target architecture](#target-architecture)
3. [Decisions](#decisions)
4. [Open decisions](#open-decisions)
5. [Repository layout](#repository-layout)
6. [Workflow](#workflow)
7. [CI](#ci)
8. [Test layers](#test-layers)
9. [Documentation](#documentation)
10. [Milestones](#milestones)
11. [Risks](#risks)

## Goal

A user sends a query. Real Feel finds matching tweets on X. Clef-flash classifies each tweet's account as bot or human. The API returns the result.

You build this system to learn. You also build it to run at serious load. Two outcomes matter equally:

- **The system.** It holds up under load tests. It autoscales. Datadog shows you every part.
- **The model operations.** You run Clef-flash on your own GPU. You measure accuracy with an eval set. You measure speed with benchmarks. You compare metadata versions.

## Target architecture

```text
                 (M5: ALB + Auto Scaling Group)
User ──HTTP──▶ API (Rust, axum, EC2) ──────────▶ Postgres + pgvector (RDS)
                  │                                   ▲
                  │ scraper job                       │ tweets, accounts,
                  ▼                                   │ predictions          
          queue: scraper-jobs                         │
                  │                                   │
                  ▼                                   │
        worker, role: scraper ────────────────────────┤
          └─ scraper ──▶ twitterapi.io ──▶ X          │
                  │                                   │
                  │ classifier jobs (one per batch)   │
                  ▼                                   │
          queue: classifier-jobs                      │
                  │                                   │
                  ▼                                   │
        worker, role: classifier ─────────────────────┘
          ├─ builds model inputs (metadata version, question version)
          └─ classifier ──▶ inference backend
                             ├─ hosted backend: Clef-flash on Workers AI
                             └─ self-hosted backend: inference server on GPU EC2
                                  ├─ reference server (Python)
                                  └─ candle server (Rust)

Datadog watches every component. OpenTelemetry carries the traces.
```

### Request flow

1. The user sends `POST /queries` with the query text.
2. The API checks for [query reuse](#query-reuse). On a match, it returns `202` and the earlier query's ID, and stops. Otherwise it inserts the query into Postgres with query status `queued`.
3. The API sends a scraper job to the `scraper-jobs` queue. The job carries the trace context.
4. The API returns `202` and the query ID.
5. A scraper worker takes the scraper job. It sets the query status to `scraping`.
6. The scraper fetches tweets and their accounts through twitterapi.io.
7. The scraper worker upserts the tweets and accounts into Postgres.
8. The scraper worker sorts the query's tweets by tweet ID. It splits them into batches of `BATCH_SIZE` and numbers them from 0. In one `UPDATE`, it records the batch count and sets the query status to `classifying`. If the scraper found no tweets, the scraper worker sets the status to `done` instead.
9. The scraper worker sends one classifier job for each batch. A classifier job carries tweet IDs, never tweet text.
10. A classifier worker takes a classifier job. If a `query_batches` row already exists for its batch, the worker deletes the job and stops. Otherwise it builds one model input for each tweet that has no prediction for the current versions.
11. The classifier sends the batch to the inference backend. It gets back one bot probability for each model input.
12. The classifier worker opens one Postgres transaction. It locks the query row, upserts the predictions, and upserts one `query_batches` row for its batch number.
13. In the same transaction, the classifier worker counts the `query_batches` rows. When the count equals the batch count, it sets the query status to `done`.
14. The user polls `GET /queries/{id}`. The API returns the result, one page at a time, once the status is `done`.

Step 8 comes before step 9 on purpose. If the scraper worker sent classifier jobs first, a fast classifier worker could finish the last batch while the batch count was still null. No worker would set `done`.

The row lock in step 12 matters. Without it, two workers that finish the last two batches at once could each count one batch short. Neither would set `done`.

### Query reuse

- A new query reuses an earlier query when both conditions hold:
  - The two texts match after trimming and lower-casing.
  - The earlier query is younger than `QUERY_REUSE_SECS` and its status is not `failed`.
- The earlier query can still be in progress. The new request then waits on the same scrape.
- The API runs the check and the insert in one transaction. It first takes a Postgres advisory lock on the hash of the lower-cased text. Two identical queries sent at once then get one query ID and one scraper job.
- A failed query is never reused. The next identical query scrapes again.

### Failure flow

- SQS delivers each job at least once. The worker upserts every row, so a second delivery changes nothing. A repeated classifier job upserts the same `query_batches` row, so it never counts twice.
- If the API cannot send the scraper job, the API sets the query status to `failed` and returns `503`.
- Postgres applies a status change only when the current status allows it. See [Status changes](#status-changes). A retried scraper job cannot move a `done` query back to `scraping`.
- The worker deletes a job only after Postgres commits.
- A redelivered scraper job can find the query in `classifying`. The scraper worker skips the scrape. It rebuilds the same batches from the stored tweets and sends the classifier jobs again. The `query_batches` rows make the repeats harmless.
- The scraper worker writes the batch count only while it is null or equal to the new count. A late scraper job never changes it.
- Each queue has a visibility timeout of 300 seconds. Each worker abandons a job after `JOB_DEADLINE_SECS`, which stays below the visibility timeout. SQS never redelivers a job that a worker still runs.
- When a job fails, SQS makes the job visible again after the visibility timeout. A worker retries it.
- After five failed receives, SQS moves the job to the dead-letter queue.
- On the fifth receive, the worker reads `ApproximateReceiveCount`. If that attempt fails, the worker sets the query status to `failed` and records the error.
- A worker that crashes on the fifth receive never sets `failed`. So each worker role also reads its own dead-letter queue. For each job there, the worker sets the query status to `failed` and deletes the job.
- One batch that reaches the dead-letter queue fails the whole query. The API returns no partial result.

## Decisions

Each row becomes a decision record in M0. "Agreed" rows come from the design discussion. "Proposed" rows are new in this plan. You accept or replace them in M0.

| # | Decision | Reason | Status |
|---|----------|--------|--------|
| D1 | The scraper uses twitterapi.io, a hosted X data API. The scraper hides it behind `trait Scraper`. | Writing an X scraper is a project of its own. Spike 2.1 showed that X blocks self-hosted Firecrawl. twitterapi.io returns tweets with their authors' metadata for about $0.15 per 1,000 tweets. The trait lets you swap sources. | Agreed (changed in M2) |
| D2 | Clef-flash classifies tweets. Clef is not used. | Clef-flash costs less for each input token and fits one cheap GPU. If the eval set shows it is not accurate enough, you revisit this decision. | Agreed |
| D3 | Metadata versions and question versions are first-class. Every prediction stores them. | Metadata decides accuracy as much as the model does. You must compare versions with evidence. | Agreed |
| D4 | SQS replaces Kafka. | Real Feel needs jobs that run later and retry. It does not need replay, many consumers, or high-volume streams. SQS needs no servers. | Agreed |
| D5 | Rust workers replace PySpark. | Each tweet costs one HTTP call and one database write. That is a job queue, not big-data work. | Agreed |
| D6 | Rust for the API, workers, eval tooling, and shared types. Python only for the reference server. | One language for every component you write. Rust types give compile-time checks across components. Python stays where Clef-flash's own code lives. | Agreed |
| D7 | The API and workers run on EC2. | Fixed capacity makes load tests measure Real Feel's code, not Lambda's scaling. | Agreed |
| D8 | Postgres holds query status. DynamoDB goes away. | One database is simpler than two. Postgres handles the load. | Agreed |
| D9 | One Postgres instance, with connection pooling in M5. Sharding waits for M8. | A million tweets with their accounts and predictions fits in a few GB. Load tests must prove a need before you shard. | Agreed |
| D10 | No load balancer until M5. Then an ALB with an Auto Scaling Group. | One API instance needs no load balancer. Scaling experiments need one. | Agreed |
| D11 | You run Clef-flash on your own GPU instance (M6). | Learning goal. You own the model operations end to end. | Agreed |
| D12 | Datadog for observability. The code emits OpenTelemetry. | The Datadog student plan is free for 2 years and covers 10 hosts. OpenTelemetry keeps the code vendor-neutral. | Agreed |
| D13 | Two queues: `scraper-jobs` and `classifier-jobs`. | Scraping and classifying have different speeds and costs. Two queues let each worker role autoscale on its own queue. | Proposed |
| D14 | Postgres access through `sqlx`. | Pure Rust. `sqlx` checks SQL against the schema at compile time. | Proposed |
| D15 | Terraform for AWS infrastructure. Terraform stores its state file in S3 with native S3 locking. | Widely used. Plain configuration files. No extra lock table. | Proposed |
| D16 | The inference server contract types in `crates/types` generate JSON Schema (task 6.2). The reference server generates its Python models from that JSON Schema. No other type derives `schemars`. | Rust and Python share one contract. CI catches drift between the two. | Proposed |
| D17 | GitHub Actions authenticates to AWS with OIDC. | No long-lived AWS keys in GitHub secrets. | Proposed |

## Open decisions

You make each decision before its deadline. Each decision ends in a decision record.

| # | Question | Options | Deadline |
|---|----------|---------|----------|
| O1 | Can self-hosted Firecrawl fetch X search pages, profiles, and tweets? | Self-hosted Firecrawl; Apify actors; a third-party X data API | Settled in task 2.1: no. The scraper uses twitterapi.io. |
| O2 | Which embedding model, and how many dimensions? | A Workers AI embedding model; a small embedding model that runs inside the worker | M8, before the embeddings experiment |
| O3 | Which eval set? | Accounts you hand-label from scraped tweets; Cresci-2017; both | M4, task 4.1 |
| O4 | Which bot threshold? | 0.5 by default; tuned from the eval report | M4, task 4.7 |
| O5 | How does Clef-flash's schema head run outside Cloudflare? | Hugging Face `transformers` with Cloudflare's code; custom code | M6, task 6.1 |
| O6 | Which variant does the self-hosted backend run? | `bf16`, `int8`, or `int4`, all on g6.xlarge | M6, task 6.7 |
| O7 | Does `candle` support Clef-flash's base architecture? | Use `candle` as is; add the architecture to `candle`; stop M7 | M7, task 7.1 |
| O8 | Which server becomes the default self-hosted backend? | reference server; candle server | M7, task 7.5 |

## Repository layout

This is the layout after M7. Each milestone adds its own parts.

```text
real-feel/
├── Cargo.toml                 # workspace root
├── Cargo.lock                 # committed (binaries need a lock file)
├── rust-toolchain.toml
├── crates/
│   ├── types/                 # shared contracts: Query, Tweet, Account, ModelInput, Prediction, jobs
│   ├── telemetry/             # tracing, OpenTelemetry, and trace context, shared by all binaries
│   ├── db/                    # sqlx pool, migrations, repository functions
│   ├── queue/                 # SQS send, receive, delete for typed jobs
│   ├── scraper/               # trait Scraper; twitterapi.io and fixture implementations
│   ├── metadata/              # builds model inputs for each metadata version and question version
│   ├── classifier/            # trait Classifier; hosted, self-hosted, and fake implementations
│   ├── api/                   # binary: axum HTTP server
│   ├── worker/                # binary: --role scraper | classifier
│   ├── eval/                  # binary: experiments and eval reports
│   └── inference/             # binary: candle server (M7)
├── inference/
│   └── reference/             # Python reference server (uv)
├── infra/
│   └── terraform/             # AWS infrastructure
├── loadtest/
│   └── k6/                    # load test scripts
├── docs/
│   ├── refactor-plan.md       # this file
│   ├── glossary.md
│   ├── writing-rules.md
│   ├── local-dev.md
│   ├── decisions/             # decision records (local only, git ignores it)
│   └── reports/               # eval reports, benchmarks, load test reports, spikes
├── docker-compose.yml         # Postgres (pgvector) and LocalStack (SQS)
└── .github/
    ├── workflows/
    └── pull_request_template.md
```

Each crate depends only on crates above it in this list. `types` depends on nothing in the workspace.

## Workflow

### Branches

- `main` holds working code. Nobody pushes to `main` directly.
- You create one branch for each task.
- Branch names: `<kind>/m<N>-<short-name>`. Example: `add/m1-db-migrations`.
- Kinds match the commit types: `add`, `fix`, `refactor`, `docs`, `test`, `infra`, `chore`.

### Pull requests

- You open one PR for each task. Keep PRs small.
- CI must pass before you merge.
- You squash-merge into `main`.
- The PR template asks for:
  - the milestone and task number
  - updated module docs
  - a decision record for each new architecture decision

### Commits

[writing-rules.md](writing-rules.md#commit-messages) defines the commit format. In short:

```text
<type>(<regarding>): <brief and clear description>

<One paragraph that explains the changes.>
```

- Example subject: `add(db): migrations for queries and tweets`.
- The body is one paragraph at most.
- No commit carries a `Co-authored-by` line.
- The squash-merge subject is the PR title, so the PR title follows the same format.

### Definition of done for a task

- CI passes.
- Every touched module's `README.md` matches the code.
- The glossary holds every new term.
- Every new architecture decision has a decision record.

### Closing a milestone

You check every gate item in this plan. Milestones get no git tags.

## CI

GitHub Actions runs every workflow.

| Workflow | Trigger | Jobs | Starts in |
|----------|---------|------|-----------|
| `ci.yml` | every push to `main`; every PR | `rust`, `docs`, `commits`, `integration`, `e2e`, `python`, `contract` | M0, grows each milestone |
| `deploy.yml` | `ci.yml` succeeds on `main` | build Docker images, push to ECR, roll out to EC2 | M3 |
| `nightly.yml` | daily schedule; manual | hosted backend contract test; small experiment on the eval set | M2 |
| `gpu.yml` | manual only | parity tests and benchmarks on the GPU instance | M6 |

### `ci.yml` jobs

- **rust**: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` for unit tests. CI starts Postgres and applies the migrations with `psql`, so the `sqlx` macros check every query against a live schema. `Swatinem/rust-cache` caches builds.
- **docs**: `markdownlint`, `lychee` for links. Claude checks the writing rules itself, so no linter runs them.
- **commits** (PRs only): `scripts/check-commits.sh` checks the PR title and every commit against the commit format. It rejects any `Co-authored-by` line.
- **integration** (from M1): after checkout, CI runs `docker compose up -d --wait` to start Postgres (pgvector) and LocalStack. CI applies the migrations with `psql`. CI then runs `cargo test --workspace --features integration`. Integration tests sit behind the `integration` Cargo feature, so the `rust` job never runs them. Each test creates its own queues with unique names and deletes them at the end, so parallel tests never read each other's jobs.
- **e2e** (from M2): Docker Compose runs Postgres and LocalStack. One test process runs the API and both worker roles with the fixture scraper and the fake classifier. The test sends a query and polls until the status is `done`.
- **python** (from M6, only when `inference/` changes): `ruff`, `pyright --strict`, `pytest`. A check confirms the generated Python models match the current JSON Schema.
- **contract** (from M6): the reference server's request and response pass validation against the JSON Schema from `crates/types`.

CI calls no paid services on PRs. Only `nightly.yml` and `gpu.yml` call paid services.

## Test layers

| Layer | What it checks | Where it runs | Network |
|-------|----------------|---------------|---------|
| Unit | One function or one type | `ci.yml` rust | None |
| Integration | One crate against Postgres or SQS | `ci.yml` integration | Docker Compose: LocalStack, Postgres |
| End-to-end | The full request flow with fakes | `ci.yml` e2e | Containers only |
| Contract | The hosted backend still accepts Real Feel's model inputs | `nightly.yml` | Cloudflare (paid) |
| Parity | Two inference servers give the same bot probabilities | `gpu.yml` | GPU instance |
| Eval | Accuracy and calibration | `nightly.yml`, manual | Inference backend |
| Benchmark | Inference server throughput and latency | `gpu.yml` | GPU instance |
| Load | API break points and autoscaling | manual, k6 | AWS |

## Documentation

You write docs as you build. Docs ship in the same PR as the code.

| File | Contents | Owner milestone |
|------|----------|-----------------|
| `README.md` | What Real Feel is. How to start. Links to `docs/`. | M0, updated each milestone |
| `docs/local-dev.md` | Exact steps from a clean machine to a running stack. | M0 |
| `docs/decisions/` | One decision record for each decision. Local only: git ignores the folder. | M0 onward |
| `docs/reports/` | Spike results, eval reports, benchmarks, load test reports. | M2 onward |
| `crates/*/README.md` | Onboarding doc for each crate. Template in [writing-rules.md](writing-rules.md#module-docs). | Same PR as the crate |

---

## Milestones

| Milestone | Name | Outcome |
|-----------|------|---------|
| M0 | Foundation | Clean repo, CI, branch protection, docs skeleton |
| M1 | Contracts, database, API | `POST /queries` and `GET /queries/{id}` work against Postgres and SQS locally |
| M2 | Workers with the hosted backend | A query completes end to end locally |
| M3 | First AWS deployment | The same flow runs on AWS, with Datadog |
| M4 | Eval set and metadata versions | Evidence on which metadata version classifies best |
| M5 | Load tests and autoscaling | Measured break points. ALB, Auto Scaling Groups, SLOs. |
| M6 | Self-hosted backend: reference server | Clef-flash runs on your own GPU. Quantization cost measured. |
| M7 | Self-hosted backend: candle server | Rust inference server checked for parity and benchmarked |
| M8 | Later experiments | Sharding, semantic search, time-series analysis, query decomposition |

M3 comes before M4 on purpose. You find deployment problems while the system is small.

The self-hosted backend (M6 and M7) comes after the load tests. The load tests need only the hosted backend, and AWS needs days to approve the GPU quota.

---

### M0: Foundation

**Goal:** a clean repository where CI guards `main` and docs exist before code.

**Tasks**

- [x] **0.1** You tag the current `main` as `legacy-v1` and push the tag. The tag keeps the old pipeline, model, and SAM code reachable.
- [x] **0.2** You protect `main` in GitHub settings: require a PR, require `ci.yml` to pass, allow squash merge only.
- [x] **0.3** You add `.github/pull_request_template.md` with the checklist from [Pull requests](#pull-requests).
- [x] **0.4** You delete the legacy code: `pipeline/`, `model/`, `api/template.yaml`, `api/LOCAL_DEV.md`, and the root `PLAN.md`.
- [x] **0.5** You trim `docker-compose.yml` to two services: Postgres (pgvector) and LocalStack running SQS. You delete Kafka, ZooKeeper, Spark, and DynamoDB. You update `infra/localstack/init.sh` to create the four queues.
- [x] **0.6** You create the Cargo workspace. You move `api/` to `crates/api/`. The API serves only `GET /health` for now.
- [x] **0.7** You remove `Cargo.lock` from `.gitignore` and commit the lock file. You remove the SAM and Lambda entries from `.gitignore`.
- [x] **0.8** You add `rust-toolchain.toml` to pin the Rust version.
- [x] **0.9** You write `ci.yml` with the `rust` and `docs` jobs.
- [x] **0.10** You add the Vale configuration: passive voice, sentence length, and a substitution list from the glossary's "Do not use" column. M1 removed Vale. Claude now checks the writing rules itself.
- [x] **0.11** You write `docs/local-dev.md` and a decision record for each row in [Decisions](#decisions). The records stay on your machine: git ignores `docs/decisions/`.
- [x] **0.12** You rewrite `README.md` for the new architecture. `.claude/CLAUDE.md` stays local: git ignores `.claude/`.
- [x] **0.13** You start the tasks that wait on other companies:
  - Activate the Datadog student plan.
  - Create a Cloudflare account and a Workers AI API token. Store the token in `.env`, never in git. You host Firecrawl yourself, so it needs no account.
- [x] **0.14** You create an AWS Budgets alert with a monthly limit you choose.
- [x] **0.15** You add a CI check on every PR. It rejects a commit or PR title that breaks the [commit format](writing-rules.md#commit-messages). It rejects any commit with a `Co-authored-by` line.

**Docs:** `docs/local-dev.md`, `docs/decisions/0001`–`0017` (local), `crates/api/README.md`.

**Gate**

- [x] The `legacy-v1` tag exists on GitHub.
- [x] `main` rejects direct pushes.
- [x] CI passes on `main`.
- [x] `docker compose up` starts only Postgres and LocalStack. LocalStack lists four queues.
- [x] `cargo run -p api` serves `GET /health` with status `200`.
- [x] Vale passes on every file in `docs/`. M1 removed Vale.
- [x] Every row in [Decisions](#decisions) has a decision record with status `accepted`.

---

### M1: Contracts, database, API

**Goal:** the API accepts a query, stores it, sends a scraper job, and reports query status. Everything runs locally.

**Tasks**

- [x] **1.1** `crates/types`: You define `Query`, `QueryStatus`, `Tweet`, `Account`, `ScraperJob`, `ClassifierJob`, `Prediction`, `MetadataVersion`, `QuestionVersion`, `ModelVersion`. Every type derives `serde`. No type derives `schemars` yet (D16).
  - `ScraperJob` carries the query ID and the trace context.
  - `ClassifierJob` carries the query ID, the batch number, the tweet IDs, and the trace context. It never carries tweet text, so it stays far below the SQS message size limit.
  - `Prediction` holds what Postgres stores: tweet ID, the three versions, and bot probability. `ModelInput` waits for M2, because its shape depends on Cloudflare's request format.
- [x] **1.2** `crates/db`: You write `sqlx` migrations for these tables:
  - `queries`: `id` (`uuid`), `text`, `status` (Postgres enum `query_status`), `error` (null until `failed`), `batch_count` (null until the scraper worker sets it), `created_at`, `updated_at`
  - `accounts`: `id` (`text`), `handle`, `raw` (`jsonb`), `updated_at`
  - `tweets`: `id` (`text`), `account_id`, `text`, `posted_at` (nullable), `raw` (`jsonb`), `updated_at`
  - `query_tweets`: `query_id`, `tweet_id`. Primary key: both columns.
  - `query_batches`: `query_id`, `batch_number`, `finished_at`. Primary key: the first two columns.
  - `predictions`: `tweet_id`, `model_version`, `metadata_version`, `question_version`, `p_bot`, `raw` (`jsonb`), `created_at`. Primary key: the first four columns.

  `accounts` and `tweets` stay minimal on purpose. `raw` holds every field the scraper returns. Task 2.2 adds columns once spike 2.1 shows what the X data source returns. IDs are `text` because the spike may show that a page exposes only the handle.
- [x] **1.3** `crates/db`: You write the repository functions. They are the only code that runs SQL.
  - Each worker write upserts with `ON CONFLICT … DO UPDATE … WHERE (columns) IS DISTINCT FROM (excluded columns)`. A second call with the same input changes no row. `updated_at` changes only when content changes.
  - `POST /queries` inserts. Each query gets a new ID, so this write is never an upsert.
  - Each status change is a conditional `UPDATE` from the [Status changes](#status-changes) table. The function returns whether Postgres applied the change.
  - Git ignores the `sqlx` offline data in `.sqlx/`. It stays on your machine.
- [x] **1.4** `crates/queue`: You write typed `send`, `receive`, and `delete` functions over SQS.
  - `receive` asks SQS for the `ApproximateReceiveCount` attribute. It returns the job, the receive count, and the receipt handle.
  - `send` writes the current trace context into the job. `receive` restores it, so a worker's spans join the API's trace.
  - Queue URLs come from configuration: `SCRAPER_QUEUE_URL`, `CLASSIFIER_QUEUE_URL`.
- [x] **1.5** `crates/telemetry`: You set up `tracing` with JSON logs and an OpenTelemetry exporter. A missing endpoint turns the exporter off. You write the helpers that read and write the W3C `traceparent` value.
- [x] **1.6** `crates/api`: You build four endpoints with axum:
  - `POST /queries`: accepts `{ "text": … }`. Text must hold 1 to 512 characters after trimming. The API inserts a `queued` query, sends a scraper job, and returns `202` with `{ "query_id": … }`. If the send fails, the API sets the status to `failed`, records the error, and returns `503`.
  - `GET /queries/{id}?limit=&after=`: returns the query status. When the status is `done`, it also returns one page of the result. The API sorts tweets by `tweet_id`. `after` is the cursor: the last `tweet_id` of the previous page. `limit` defaults to 50, with a maximum of 200. The response carries `next_cursor`, which is null on the last page.
  - `GET /health`: the liveness check. It returns `200` while the process runs. It checks no dependency, so a short Postgres outage never marks every API instance unhealthy.
  - `GET /ready`: the readiness check. It returns `200` when Postgres and SQS respond, and `503` when either does not.
- [x] **1.7** `crates/api`: The result shows one prediction for each tweet. The API selects it with `DEFAULT_MODEL_VERSION`, `DEFAULT_METADATA_VERSION`, and `DEFAULT_QUESTION_VERSION` from configuration. The result carries `p_bot` only. `is_bot` arrives in task 4.7, when O4 sets the bot threshold.
- [x] **1.8** `crates/api`: You return every error as JSON: `{ "code": …, "message": … }`. The error codes:

  | Code | HTTP status | When |
  |------|-------------|------|
  | `invalid_query` | 400 | Query text is empty or longer than 512 characters. |
  | `invalid_query_id` | 400 | The ID in the path is not a UUID. |
  | `invalid_cursor` | 400 | `limit` is out of range, or `after` is not a valid tweet ID. |
  | `query_not_found` | 404 | No query has that ID. |
  | `queue_unavailable` | 503 | SQS rejected the scraper job. |
  | `internal` | 500 | Any other failure. The message hides internal detail. |

- [x] **1.9** You add the `integration` job to `ci.yml`. The `rust` job starts Postgres and applies the migrations, so the `sqlx` macros compile with no `.sqlx/`. See [`ci.yml` jobs](#ciyml-jobs).
- [x] **1.10** You add the new terms to the glossary. The terms: upsert, repository function, receive count, trace context, page, cursor, liveness check, readiness check, error code.

#### Status changes

Postgres applies a status change only from the statuses this table allows. Any other change is a no-op, and the repository function returns `false`.

| New status | Allowed current statuses | Who sets it |
|------------|--------------------------|-------------|
| `scraping` | `queued`, `scraping` | scraper worker. `scraping` → `scraping` lets a retried scraper job continue. |
| `classifying` | `scraping` | scraper worker |
| `done` | `scraping` (zero tweets), `classifying` | scraper worker or classifier worker |
| `failed` | `queued`, `scraping`, `classifying` | API or any worker |

`done` and `failed` are final. Nothing moves a query out of them.

**Docs:** a `README.md` for `types`, `db`, `queue`, `telemetry`, `api`. The `db` README holds the status change table. The `api` README documents every endpoint with an example HTTP request and response, and lists every error code.

**Gate**

- [x] An integration test sends `POST /queries` and then `GET /queries/{id}`. It sees status `queued`.
- [x] The same test receives a scraper job from its own queue. The job's query ID matches, and the job carries a trace context.
- [x] A test points the API at a queue that does not exist. `POST /queries` returns `503` with `queue_unavailable`, and the query status is `failed`.
- [x] A test calls each upsert repository function twice with the same input. The second call reports zero changed rows.
- [x] A test tries every status change that the table forbids. Each one returns `false` and leaves the status unchanged.
- [x] A test seeds a `done` query with 120 tweets and predictions. It pages through with `limit=50` and gets 50, 50, and 20 tweets, then a null `next_cursor`.
- [x] A test checks each error code against its HTTP status.
- [x] `GET /health` returns `200` with Postgres stopped. `GET /ready` returns `503`.
- [x] CI runs the integration tests on every PR. The `rust` job passes with no `.sqlx/`.

---

### M2: Workers with the hosted backend

**Goal:** a query travels the full flow locally and ends with status `done`. The hosted backend classifies.

**Task order**

- 2.1 comes before 2.2. 2.2 comes before 2.6.
- 2.3, 2.4, and 2.5 start at once, beside 2.1. They use the M1 `Tweet` and `Account` types.
- 2.6 and 2.7 need 2.2 to 2.5. 2.8 and 2.9 need 2.6 and 2.7.
- 2.10 needs 2.4. 2.12 needs only M1. 2.13 needs 2.6 and 2.7.
- 2.11 needs every other task.

**Configuration**

| Variable | Default | Read by |
|----------|---------|---------|
| `SCRAPER` | `twitterapi` | worker: `twitterapi` or `fixture` |
| `QUERY_REUSE_SECS` | `3600` | API: how long a query stays reusable. `0` turns query reuse off. |
| `TWITTERAPI_KEY` | none | scraper |
| `TWITTERAPI_PAGES` | `5` | scraper: the most search pages for each query, 20 tweets each |
| `TWITTERAPI_INTERVAL_SECS` | `5` | scraper: the time from the start of one request to the start of the next. The free tier allows one request every 5 seconds. |
| `FIXTURE_PATH` | `crates/scraper/fixtures/default.json` | worker, with `SCRAPER=fixture` |
| `CLASSIFIER` | `hosted` | worker: `hosted` or `fake` |
| `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | none | classifier |
| `CLASSIFIER_RPS` | `5` | classifier: the requests-per-second limit |
| `BATCH_SIZE` | `25` | scraper worker |
| `JOB_DEADLINE_SECS` | `240` | worker |
| `SCRAPER_DLQ_URL`, `CLASSIFIER_DLQ_URL` | LocalStack URLs | worker |

The worker reads the M1 variables too. `.env.example` lists every variable. `infra/localstack/init.sh` sets the 300-second visibility timeout.

**Tasks**

- [x] **2.1** **Spike:** You test self-hosted Firecrawl against an X search page, a profile page, and a tweet page. X blocks all three, so you test twitterapi.io with `scripts/spike-twitterapi.sh`. You record which fields come back. You write `docs/reports/spike-x-data.md` and a decision record that settles O1.
- [x] **2.2** `crates/scraper`: You define `trait Scraper`. You write the twitterapi.io implementation and a fixture implementation. The fixture implementation reads JSON files from `crates/scraper/fixtures/`. You add migrations for the `accounts` columns that spike 2.1 confirms. A tweet ID and an account ID hold X's numeric IDs as text.
- [x] **2.3** `crates/metadata`: You define `ModelInput` in `crates/types`. You write metadata version `v1` (tweet text only) and question version `v1` (bot or human). The functions are pure: the same tweet and account always give the same model input.
- [x] **2.4** `crates/classifier`: You define `trait Classifier`. You write the hosted implementation and the fake. The hosted implementation:
  - reads Cloudflare's request format for Clef-flash and records it in the crate README
  - retries on `429` and `5xx` with exponential backoff and jitter
  - stays under a configured requests-per-second limit
- [x] **2.5** ~~`crates/embedder`~~ Deferred to M8. You built a Workers AI embedder, then removed it, because no task reads embeddings. O2 moves to M8.
- [x] **2.6** `crates/worker`, role `scraper`: the worker takes a scraper job and carries out steps 5–9 of the [request flow](#request-flow). `BATCH_SIZE` sets the batch size. A redelivered scraper job follows the [failure flow](#failure-flow).
- [x] **2.7** `crates/worker`, role `classifier`: the worker takes a classifier job and carries out steps 10–13. The worker opens one Postgres transaction and locks the query row first. In that transaction it upserts the predictions and the `query_batches` row, counts the batches, and changes the status. The race test runs this commit step from two tasks at once behind a `tokio::sync::Barrier`.
- [x] **2.8** `crates/worker`: You add the [failure flow](#failure-flow), the job deadline, and the dead-letter queue reader. A fake classifier that fails N times proves the retry and the DLQ. The tests create their own queues with a 1-second visibility timeout and `maxReceiveCount` 5.
- [x] **2.9** You add the `e2e` job to `ci.yml`. The test sits in `crates/worker/tests/e2e.rs` behind an `e2e` feature. Docker Compose runs Postgres and LocalStack. The test starts the API router and both worker roles in one process, with the fixture scraper and the fake classifier. A fixture of 5 tweets and `BATCH_SIZE=2` give 3 batches. The test polls `GET /queries/{id}` until the status is `done`.
- [x] **2.10** You add `nightly.yml` with a contract test against the hosted backend. It reads `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` from GitHub secrets. It sends one fixed model input and asserts one bot probability from 0 to 1. A failure fails the workflow, and GitHub emails you.
- [x] **2.11** You write `scripts/smoke.sh`. It sends one real query through twitterapi.io and the hosted backend on your machine. It passes when the status is `done`, the result holds at least 10 tweets, and every tweet has a prediction.
- [x] **2.12** `crates/api` and `crates/db`: You add [query reuse](#query-reuse) to `POST /queries`. A migration adds an index on `LOWER(text)` and `created_at`. Tests cover a match, a match while the earlier query runs, a `failed` earlier query, an earlier query older than `QUERY_REUSE_SECS`, and two identical requests sent at once.
- [x] **2.13** Local observability. `docker-compose.yml` runs Jaeger. The API and both worker roles log every step of a query at `info` and name each process in its spans. `scripts/smoke.sh` writes the run log and prints the Jaeger trace link. The trace of one query spans the API, the scraper worker, and the classifier worker.

**Docs:** a `README.md` for `scraper`, `metadata`, `classifier`, `worker`. The `metadata` README defines `v1` field by field. You write `docs/reports/smoke-m2.md` with the query text, the tweet count, the batch count, and the run time.

**Gate**

- [ ] The e2e test passes in CI: `POST /queries` leads to status `done` with predictions for every tweet.
- [x] A test shows a failing job reaching the DLQ and the query status becoming `failed`.
- [x] A test shows a duplicated classifier job producing no duplicate predictions and no extra `query_batches` row.
- [x] A test runs two classifier workers on the last two batches at once. The query ends with status `done`.
- [x] A test redelivers a scraper job after the status reaches `classifying`. The query still ends with status `done`, with one `query_batches` row for each batch.
- [x] A test leaves a job in the dead-letter queue without a `failed` status. The dead-letter queue reader sets the status to `failed`.
- [x] A test with zero scraped tweets ends with status `done`.
- [x] A test sends two identical queries at once. Both get one query ID, and SQS holds one scraper job.
- [x] `scripts/smoke.sh` completes one real query. The smoke report records it.
- [x] A decision record settles O1. O2 moves to M8 with the embeddings.

---

### M3: First AWS deployment

**Goal:** the M2 flow runs on AWS. A merge to `main` deploys it. Datadog shows it.

**Tasks**

- [ ] **3.1** `infra/terraform`: You set up the S3 bucket for the Terraform state file. You split the configuration into modules: `network`, `database`, `queues`, `compute`, `registry`, `iam`.
- [ ] **3.2** `network`: You create one VPC, public subnets for the API, and private subnets for RDS. Security groups allow only the traffic each component needs.
- [ ] **3.3** `database`: You create RDS Postgres 16 with the `vector` extension, on the smallest instance class that runs it.
- [ ] **3.4** `queues`: You create `scraper-jobs`, `classifier-jobs`, and their DLQs, with `maxReceiveCount = 5`.
- [ ] **3.5** `registry`: You create ECR repositories. You write multi-stage Dockerfiles for `api` and `worker`.
- [ ] **3.6** `compute`: You create one EC2 instance for the API and one for the workers. Both use instance profiles. SSM Parameter Store holds the secrets.
- [ ] **3.7** `iam`: You create an OIDC role for GitHub Actions. The role can push to ECR and send SSM commands. It cannot do anything else.
- [ ] **3.8** You write `deploy.yml`. It builds the images, pushes them to ECR, and tells each instance to pull and restart through SSM Run Command.
- [ ] **3.9** You install the Datadog agent on both instances. The agent receives OpenTelemetry traces and JSON logs. You build a first dashboard: request rate, error rate, queue depth, query time from `queued` to `done`.
- [ ] **3.10** You tag every AWS resource with `project=real-feel`. You write the teardown steps in `infra/terraform/README.md`.

The API serves plain HTTP on an Elastic IP in M3. HTTPS arrives with the ALB in M5.

**Docs:** `infra/terraform/README.md` (setup, deploy, teardown, costs), plus a decision record for any change to D15 or D17.

**Gate**

- [ ] A merge to `main` deploys new images with no manual steps.
- [ ] One real query completes on AWS.
- [ ] Datadog shows one trace that spans the API, the scraper worker, and the classifier worker.
- [ ] `terraform destroy` followed by `terraform apply` rebuilds the stack. You prove it once.
- [ ] The AWS Budgets alert covers the new resources.

---

### M4: Eval set and metadata versions

**Goal:** evidence on which metadata version and question version give the most accurate and best-calibrated bot probabilities.

**Tasks**

- [ ] **4.1** You settle O3. You hand-label 300 to 500 accounts from scraped tweets. You write a loader that turns the labeled accounts into the eval set format. S3 stores the eval set. The repository stores only a manifest with checksums.
- [ ] **4.2** You pick a fixed sample of the eval set with a fixed seed. Every experiment uses that sample, so experiments compare fairly.
- [ ] **4.3** `crates/metadata`: You write more metadata versions. A suggested start:
  - `v2`: `v1` plus account age, follower count, following count, tweet count, verified flag
  - `v3`: `v2` plus posting times of recent tweets and the share of duplicate text
  - `v4`: `v3` plus profile description and profile URL
  Each version gets a field-by-field definition in the README and its own unit tests.
- [ ] **4.4** `crates/metadata`: You write question versions that word the bot question in different ways.
- [ ] **4.5** `crates/eval`: You build the `eval` binary. `eval experiment --model-version … --metadata-version … --question-version …` runs one experiment. The binary prints the expected cost before it calls a paid backend, and asks you to confirm.
- [ ] **4.6** `crates/eval`: The binary computes accuracy, precision, recall, F1, ROC-AUC, Brier score, and expected calibration error. Postgres stores every experiment. `eval report` writes an eval report that compares experiments.
- [ ] **4.7** You settle O4 from the eval report. You record the chosen bot threshold in a decision record. The API adds `is_bot` to the result.
- [ ] **4.8** You compare each metadata version's fields against the fields twitterapi.io returns. You list the gaps in the eval report.
- [ ] **4.9** CI runs one experiment on a ten-item fixture eval set with the fake classifier.

**Docs:** `crates/eval/README.md`, `docs/reports/eval-m4.md`, decision records for O3 and O4 and the default metadata version.

**Gate**

- [ ] The eval report compares at least three metadata versions on the hosted backend.
- [ ] The eval report shows a calibration curve for each experiment.
- [ ] A decision record names the default metadata version, the default question version, and the bot threshold.
- [ ] Workers use the defaults from the decision record.

---

### M5: Load tests and autoscaling

**Goal:** measured break points, and evidence that each change moved them. The tests use the hosted backend. The self-hosted backend comes in M6 and M7.

**Tasks**

- [ ] **5.1** `loadtest/k6`: You write four scenarios:
  - `smoke`: a low rate that checks the scripts work
  - `ramp`: a rising rate until the targets fail
  - `burst`: a sudden jump in rate
  - `soak`: a steady rate for one hour

  Each scenario sets the share of queries that repeat an earlier query's text. The report gives the query reuse rate next to each break point.
- [ ] **5.2** You run each scenario in two modes:
  - **System only:** fake scraper and fake classifier. The test measures Real Feel's own code.
  - **Full:** real twitterapi.io and the hosted backend. The test measures the whole chain. It costs real money, so run it at a small scale.
- [ ] **5.3** You record the baseline break points on one API instance. You write `docs/reports/load-baseline.md`.
- [ ] **5.4** You add an ALB and an Auto Scaling Group for the API. The ALB checks `GET /health`. ACM provides the HTTPS certificate.
- [ ] **5.5** You give each worker role its own Auto Scaling Group. Each group scales on its queue's backlog per instance.
- [ ] **5.6** You add connection pooling: RDS Proxy or PgBouncer. You add a read replica only if the load tests show reads limit throughput.
- [ ] **5.7** You define SLOs in Datadog: `POST /queries` p99 latency, error rate, and time from `queued` to `done`. You add alerts.
- [ ] **5.8** You run failure drills during a `soak` test:
  - stop an API instance
  - send a broken job to the DLQ
- [ ] **5.9** You re-run every scenario after each change. You add the numbers to the load test report.
- [ ] **5.10** You ask AWS for a G-instance vCPU quota increase, on-demand and Spot. Ask for 4 vCPUs of each. New accounts start at 0. Approval can take days, so M6 needs this request in early.

**Docs:** `loadtest/k6/README.md`, `docs/reports/load-m5.md` with before-and-after break points, decision records for the pooling choice and the autoscaling policies.

**Gate**

- [ ] The load test report shows break points before and after the ALB, autoscaling, and pooling.
- [ ] Autoscaling adds and removes instances during a `ramp` test. Datadog shows it.
- [ ] The failure drills cause no lost queries.
- [ ] The SLO dashboard and alerts are live.
- [ ] AWS has received the GPU quota application.

---

### M6: Self-hosted backend: reference server

**Goal:** Clef-flash runs on your own GPU instance. This milestone needs the quota from task 5.10. Workers can classify through it. You know what quantization costs in accuracy.

**Tasks**

- [ ] **6.1** **Spike:** You read Clef-flash's model card and code. You confirm that Cloudflare publishes the weights, and under which licence. You find how the schema head runs. You settle O5 in a decision record.
- [ ] **6.2** `crates/types`: You define the inference server contract: `POST /v1/classify` takes a batch and returns predictions with a model version. The types generate JSON Schema.
- [ ] **6.3** `inference/reference`: You build the reference server in Python with `uv`. It generates its request and response models from the JSON Schema. It applies dynamic batching with a maximum batch size and a maximum wait.
- [ ] **6.4** `inference/reference`: You load Clef-flash in the `bf16`, `int8`, and `int4` variants. The variant is a setting.
- [ ] **6.5** `infra/terraform`: You add a GPU module. It holds a launch template with an NVIDIA Deep Learning AMI, Spot instances, and an Auto Scaling Group of size 0 or 1. A script on the instance shuts the instance down after a set idle time.
- [ ] **6.6** `crates/classifier`: You write the self-hosted implementation. `CLASSIFIER_BACKEND` selects `hosted`, `self_hosted`, or `fake`.
- [ ] **6.7** You run the same experiments on each variant and on the hosted backend. You settle O6. The eval report shows:
  - agreement rate on bot or human against the hosted backend
  - mean absolute difference in bot probability
  - the change in each metric
- [ ] **6.8** You build the benchmark harness. It sends a fixed set of model inputs at batch sizes 1, 8, 32, and 64. It records throughput, p50, p95, and p99 latency, and GPU memory.
- [ ] **6.9** You enable Datadog GPU monitoring: GPU use, GPU memory, batch size, inference latency.
- [ ] **6.10** You add the `python` and `contract` jobs to `ci.yml`. You add `gpu.yml` for benchmarks.
- [ ] **6.11** You scale the GPU Auto Scaling Group on the `classifier-jobs` backlog, with a minimum of 0.
- [ ] **6.12** You run a failure drill during a `soak` test: let AWS reclaim a Spot GPU instance.

**Docs:** `inference/reference/README.md`, `docs/reports/quantization-m6.md`, `docs/reports/benchmark-reference.md`, decision records for O5 and O6.

**Gate**

- [ ] On AWS, a query completes with `CLASSIFIER_BACKEND=self_hosted`.
- [ ] The quantization report compares every variant that fits on the chosen instance against the hosted backend.
- [ ] The benchmark report records the reference server's baseline.
- [ ] The GPU instance stops when idle. You prove it once.
- [ ] The Spot reclaim drill causes no lost queries.

---

### M7: Self-hosted backend: candle server

**Goal:** a Rust inference server that matches the reference server's output. Evidence on whether it is faster.

**Tasks**

- [ ] **7.1** **Spike:** You check whether `candle` supports Clef-flash's base architecture and quantized weights. You list what you must write yourself. You settle O7 in a decision record. If the work is too large, M7 stops here and the decision record says why.
- [ ] **7.2** `crates/inference`: You load Clef-flash's weights with `candle`. You write the schema head.
- [ ] **7.3** `crates/inference`: You serve the same `POST /v1/classify` contract. You write dynamic batching with tokio.
- [ ] **7.4** You write parity tests. Both servers classify the same model inputs. The tests pass when every bot probability differs by less than a tolerance you set in the spike.
- [ ] **7.5** You run the M6 benchmark harness on the same instance type. You settle O8 in a decision record.

**Docs:** `crates/inference/README.md`, `docs/reports/benchmark-candle.md`, decision records for O7 and O8.

**Gate**

- [ ] Parity tests pass in `gpu.yml`.
- [ ] The benchmark report compares both servers on the same hardware.
- [ ] A decision record names the default self-hosted backend.

---

### M8: Later experiments

You plan each of these when M7 closes. Each becomes its own spike and decision record.

- **Sharding experiment.** Use Citus or Aurora Limitless if M5 shows one Postgres instance limits throughput. Measure the change with the M5 scenarios.
- **Embeddings.** An embedder turns tweet text into embeddings, and Postgres stores them with pgvector. Two uses: near-duplicate tweets across accounts, a sign of coordinated bots, and `GET /search`, which finds tweets by meaning with an HNSW index. Settle O2 first. M2 built a Workers AI embedder on `bge-small-en-v1.5` (384 dimensions) and removed it, because nothing read the embeddings. Git history keeps that code.
- **Time-series analysis.** Bot and human activity for one topic over time.
- **Query decomposition.** An LLM splits one query into several scraper jobs and merges their tweets.

---

## Risks

| Risk | Effect | What you do |
|------|--------|-------------|
| twitterapi.io changes its API, raises prices, or shuts down | The scraper stops | `trait Scraper` lets you swap to another X data API. Spike 2.1 lists the alternatives. |
| Several scraper workers share one `TWITTERAPI_KEY` and together pass its rate limit | The scraper gets `429`s and slows down | Each worker retries a `429`. If the M5 load tests show many `429`s, you add a shared rate limiter: one call budget for the key across all scraper workers. |
| twitterapi.io does not return the metadata that `v2`+ need | M4 compares versions that production cannot fill | Task 4.8 lists the gaps. The eval set has the fields even if live data does not. |
| Cloudflare does not publish the Clef-flash weights | M6 and M7 stop | Task 6.1 checks the licence before you build anything. |
| Clef-flash's schema head needs custom code | M6 takes longer | Task 6.1 answers this before you build anything. |
| `candle` lacks Clef-flash's base architecture | M7 grows large | Task 7.1 can stop M7 with a decision record. |
| AWS approves GPU quota slowly | M6 waits | Task 5.10 asks during M5, so approval lands before M6. |
| GPU costs run high | Budget overrun | Spot instances, idle shutdown, an Auto Scaling Group minimum of 0, a budget alert. |
| Paid service costs run high in tests | Budget overrun | CI uses fakes. Only `nightly.yml` and manual runs call paid services. The `eval` binary prints the cost first. |
| Hand labeling takes too long | M4 has a small eval set | Label 300 accounts first. Cresci-2017 is public. |
| Cloudflare changes the hosted backend | Hosted experiments drift | `nightly.yml` catches contract changes. Every prediction stores its model version. |
