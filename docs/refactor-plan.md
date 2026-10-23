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
                 (M7: ALB + Auto Scaling Group)
User ──HTTP──▶ API (Rust, axum, EC2) ──────────▶ Postgres + pgvector (RDS)
                  │                                   ▲
                  │ scraper job                       │ tweets, accounts,
                  ▼                                   │ predictions, embeddings
          queue: scraper-jobs                         │
                  │                                   │
                  ▼                                   │
        worker, role: scraper ────────────────────────┤
          └─ scraper ──▶ Firecrawl ──▶ X              │
                  │                                   │
                  │ classifier jobs (one per batch)   │
                  ▼                                   │
          queue: classifier-jobs                      │
                  │                                   │
                  ▼                                   │
        worker, role: classifier ─────────────────────┘
          ├─ builds model inputs (metadata version, question version)
          ├─ classifier ──▶ inference backend
          │                  ├─ hosted backend: Clef-flash on Workers AI
          │                  └─ self-hosted backend: inference server on GPU EC2
          │                       ├─ reference server (Python)
          │                       └─ candle server (Rust)
          └─ embedder ──▶ embedding model

Datadog watches every component. OpenTelemetry carries the traces.
```

### Request flow

1. The user sends `POST /queries` with the query text.
2. The API inserts the query into Postgres with query status `queued`.
3. The API sends a scraper job to the `scraper-jobs` queue.
4. The API returns `202` and the query ID.
5. A scraper worker takes the scraper job. It sets the query status to `scraping`.
6. The scraper fetches tweets and accounts through Firecrawl.
7. The scraper worker upserts the tweets and accounts into Postgres.
8. The scraper worker splits the tweets into batches. It sends one classifier job for each batch.
9. The scraper worker records the batch count. It sets the query status to `classifying`.
10. A classifier worker takes a classifier job. It builds one model input for each tweet.
11. The classifier sends the batch to the inference backend. It gets back one bot probability for each model input.
12. The embedder creates one embedding for each tweet.
13. The classifier worker writes predictions and embeddings to Postgres. It adds one to the finished batch count.
14. The classifier worker that finishes the last batch sets the query status to `done`.
15. The user polls `GET /queries/{id}`. The API returns the result once the status is `done`.

### Failure flow

- SQS delivers each job at least once. The worker upserts every row, so a second delivery changes nothing.
- The worker deletes a job only after Postgres commits.
- When a job fails, SQS makes the job visible again after the visibility timeout. A worker retries it.
- After five failed receives, SQS moves the job to the dead-letter queue.
- On the fifth receive, the worker reads `ApproximateReceiveCount`. If that attempt fails, the worker sets the query status to `failed` and records the error.

## Decisions

Each row becomes a decision record in M0. "Agreed" rows come from the design discussion. "Proposed" rows are new in this plan. You accept or replace them in M0.

| # | Decision | Reason | Status |
|---|----------|--------|--------|
| D1 | The scraper uses Firecrawl. You host Firecrawl yourself. The scraper hides Firecrawl behind `trait Scraper`. | Writing an X scraper is a project of its own. The plan puts the effort into the system. Self-hosting needs no paid plan. The trait lets you swap tools. | Agreed |
| D2 | Clef-flash classifies tweets. Clef is not used. | Clef-flash costs less for each input token and fits one cheap GPU. If the eval set shows it is not accurate enough, you revisit this decision. | Agreed |
| D3 | Metadata versions and question versions are first-class. Every prediction stores them. | Metadata decides accuracy as much as the model does. You must compare versions with evidence. | Agreed |
| D4 | SQS replaces Kafka. | Real Feel needs jobs that run later and retry. It does not need replay, many consumers, or high-volume streams. SQS needs no servers. | Agreed |
| D5 | Rust workers replace PySpark. | Each tweet costs one HTTP call and one database write. That is a job queue, not big-data work. | Agreed |
| D6 | Rust for the API, workers, eval tooling, and shared types. Python only for the reference server. | One language for every component you write. Rust types give compile-time checks across components. Python stays where Clef-flash's own code lives. | Agreed |
| D7 | The API and workers run on EC2. | Fixed capacity makes load tests measure Real Feel's code, not Lambda's scaling. | Agreed |
| D8 | Postgres holds query status. DynamoDB goes away. | One database is simpler than two. Postgres handles the load. | Agreed |
| D9 | One Postgres instance, with connection pooling in M7. Sharding waits for M8. | A million tweets with embeddings fits in a few GB. Load tests must prove a need before you shard. | Agreed |
| D10 | No load balancer until M7. Then an ALB with an Auto Scaling Group. | One API instance needs no load balancer. Scaling experiments need one. | Agreed |
| D11 | You run Clef-flash on your own GPU instance (M5). | Learning goal. You own the model operations end to end. | Agreed |
| D12 | Datadog for observability. The code emits OpenTelemetry. | The Datadog student plan is free for 2 years and covers 10 hosts. OpenTelemetry keeps the code vendor-neutral. | Agreed |
| D13 | Two queues: `scraper-jobs` and `classifier-jobs`. | Scraping and classifying have different speeds and costs. Two queues let each worker role autoscale on its own queue. | Proposed |
| D14 | Postgres access through `sqlx`. | Pure Rust. `sqlx` checks SQL against the schema at compile time. | Proposed |
| D15 | Terraform for AWS infrastructure. Terraform stores its state file in S3 with native S3 locking. | Widely used. Plain configuration files. No extra lock table. | Proposed |
| D16 | Shared types generate JSON Schema. The reference server generates its Python models from that JSON Schema. | Rust and Python share one contract. CI catches drift between the two. | Proposed |
| D17 | GitHub Actions authenticates to AWS with OIDC. | No long-lived AWS keys in GitHub secrets. | Proposed |

## Open decisions

You make each decision before its deadline. Each decision ends in a decision record.

| # | Question | Options | Deadline |
|---|----------|---------|----------|
| O1 | Can self-hosted Firecrawl fetch X search pages, profiles, and tweets? | Self-hosted Firecrawl; Apify actors; a third-party X data API | M2, task 2.1 |
| O2 | Which embedding model, and how many dimensions? | A Workers AI embedding model; a small embedding model that runs inside the worker | M2, task 2.5 |
| O3 | Which eval set? | Accounts you hand-label from scraped tweets; Cresci-2017; both | M4, task 4.1 |
| O4 | Which bot threshold? | 0.5 by default; tuned from the eval report | M4, task 4.7 |
| O5 | How does Clef-flash's schema head run outside Cloudflare? | Hugging Face `transformers` with Cloudflare's code; custom code | M5, task 5.1 |
| O6 | Which variant does the self-hosted backend run? | `bf16`, `int8`, or `int4`, all on g6.xlarge | M5, task 5.7 |
| O7 | Does `candle` support Clef-flash's base architecture? | Use `candle` as is; add the architecture to `candle`; stop M6 | M6, task 6.1 |
| O8 | Which server becomes the default self-hosted backend? | reference server; candle server | M6, task 6.5 |

## Repository layout

This is the layout after M6. Each milestone adds its own parts.

```text
real-feel/
├── Cargo.toml                 # workspace root
├── Cargo.lock                 # committed (binaries need a lock file)
├── rust-toolchain.toml
├── crates/
│   ├── types/                 # shared contracts: Query, Tweet, Account, ModelInput, Prediction, jobs
│   ├── db/                    # sqlx pool, migrations, repository functions
│   ├── queue/                 # SQS send, receive, delete for typed jobs
│   ├── scraper/               # trait Scraper; Firecrawl and fixture implementations
│   ├── metadata/              # builds model inputs for each metadata version and question version
│   ├── classifier/            # trait Classifier; hosted, self-hosted, and fake implementations
│   ├── embedder/              # trait Embedder; hosted and fake implementations
│   ├── telemetry/             # tracing and OpenTelemetry setup, shared by all binaries
│   ├── api/                   # binary: axum HTTP server
│   ├── worker/                # binary: --role scraper | classifier
│   ├── eval/                  # binary: experiments and eval reports
│   └── inference/             # binary: candle server (M6)
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
- Branch names: `<kind>/m<N>-<short-name>`. Example: `feat/m1-db-migrations`.
- Kinds: `feat`, `fix`, `refactor`, `docs`, `test`, `infra`, `chore`.

### Pull requests

- You open one PR for each task. Keep PRs small.
- CI must pass before you merge.
- You squash-merge into `main`.
- The PR template asks for:
  - the milestone and task number
  - updated module docs
  - new glossary terms
  - the three writing rules, checked

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

1. You check every gate item.
2. You tag `main` as `m<N>-done`. Example: `m1-done`.

## CI

GitHub Actions runs every workflow.

| Workflow | Trigger | Jobs | Starts in |
|----------|---------|------|-----------|
| `ci.yml` | every push to `main`; every PR | `rust`, `docs`, `commits`, `integration`, `e2e`, `python`, `contract` | M0, grows each milestone |
| `deploy.yml` | `ci.yml` succeeds on `main` | build Docker images, push to ECR, roll out to EC2 | M3 |
| `nightly.yml` | daily schedule; manual | hosted backend contract test; small experiment on the eval set | M2 |
| `gpu.yml` | manual only | parity tests and benchmarks on the GPU instance | M5 |

### `ci.yml` jobs

- **rust**: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` for unit tests. `Swatinem/rust-cache` caches builds.
- **docs**: `markdownlint`, `lychee` for links, Vale for the writing rules.
- **commits** (PRs only): `scripts/check-commits.sh` checks the PR title and every commit against the commit format. It rejects any `Co-authored-by` line.
- **integration** (from M1): GitHub service containers run Postgres (pgvector) and LocalStack. CI runs the tests for `db`, `queue`, and `api` against them. `SQLX_OFFLINE=true` makes compile-time SQL checks work without a database.
- **e2e** (from M2): Docker Compose runs the whole stack with the fixture scraper, the fake classifier, and the fake embedder. The test sends a query and polls until the status is `done`.
- **python** (from M5, only when `inference/` changes): `ruff`, `pyright --strict`, `pytest`. A check confirms the generated Python models match the current JSON Schema.
- **contract** (from M5): the reference server's request and response pass validation against the JSON Schema from `crates/types`.

CI calls no paid services on PRs. Only `nightly.yml` and `gpu.yml` call paid services.

## Test layers

| Layer | What it checks | Where it runs | Network |
|-------|----------------|---------------|---------|
| Unit | One function or one type | `ci.yml` rust | None |
| Integration | One crate against Postgres or SQS | `ci.yml` integration | LocalStack, Postgres containers |
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
| `.claude/CLAUDE.md` | Current milestone and conventions for Claude Code. | Updated each milestone |

---

## Milestones

| Milestone | Name | Outcome |
|-----------|------|---------|
| M0 | Foundation | Clean repo, CI, branch protection, docs skeleton |
| M1 | Contracts, database, API | `POST /queries` and `GET /queries/{id}` work against Postgres and SQS locally |
| M2 | Workers with the hosted backend | A query completes end to end locally |
| M3 | First AWS deployment | The same flow runs on AWS, with Datadog |
| M4 | Eval set and metadata versions | Evidence on which metadata version classifies best |
| M5 | Self-hosted backend: reference server | Clef-flash runs on your own GPU. Quantization cost measured. |
| M6 | Self-hosted backend: candle server | Rust inference server checked for parity and benchmarked |
| M7 | Load tests and autoscaling | Measured break points. ALB, Auto Scaling Groups, SLOs. |
| M8 | Later experiments | Sharding, semantic search, time-series analysis, query decomposition |

M3 comes before M4 on purpose. You find deployment problems while the system is small.

---

### M0: Foundation

**Goal:** a clean repository where CI guards `main` and docs exist before code.

**Tasks**

- [ ] **0.1** You tag the current `main` as `legacy-v1` and push the tag. The tag keeps the old pipeline, model, and SAM code reachable.
- [ ] **0.2** You protect `main` in GitHub settings: require a PR, require `ci.yml` to pass, allow squash merge only.
- [ ] **0.3** You add `.github/pull_request_template.md` with the checklist from [Pull requests](#pull-requests).
- [ ] **0.4** You delete the legacy code: `pipeline/`, `model/`, `api/template.yaml`, `api/LOCAL_DEV.md`, and the root `PLAN.md`.
- [ ] **0.5** You trim `docker-compose.yml` to two services: Postgres (pgvector) and LocalStack running SQS. You delete Kafka, ZooKeeper, Spark, and DynamoDB. You update `infra/localstack/init.sh` to create the four queues.
- [ ] **0.6** You create the Cargo workspace. You move `api/` to `crates/api/`. The API serves only `GET /health` for now.
- [ ] **0.7** You remove `Cargo.lock` from `.gitignore` and commit the lock file. You remove the SAM and Lambda entries from `.gitignore`.
- [ ] **0.8** You add `rust-toolchain.toml` to pin the Rust version.
- [ ] **0.9** You write `ci.yml` with the `rust` and `docs` jobs.
- [ ] **0.10** You add the Vale configuration: passive voice, sentence length, and a substitution list from the glossary's "Do not use" column.
- [ ] **0.11** You write `docs/local-dev.md` and a decision record for each row in [Decisions](#decisions). The records stay on your machine: git ignores `docs/decisions/`.
- [ ] **0.12** You rewrite `README.md` and `.claude/CLAUDE.md` for the new architecture.
- [ ] **0.13** You start the tasks that wait on other companies:
  - Ask AWS for a G-instance vCPU quota increase, on-demand and Spot. New accounts start at 0. Approval can take days.
  - Activate the Datadog student plan.
  - Create a Cloudflare account and a Workers AI API token. Store the token in `.env`, never in git. You host Firecrawl yourself, so it needs no account.
- [ ] **0.14** You create an AWS Budgets alert with a monthly limit you choose.
- [ ] **0.15** You add a CI check on every PR. It rejects a commit or PR title that breaks the [commit format](writing-rules.md#commit-messages). It rejects any commit with a `Co-authored-by` line.

**Docs:** `docs/local-dev.md`, `docs/decisions/0001`–`0017` (local), `crates/api/README.md`.

**Gate**

- [ ] The `legacy-v1` tag exists on GitHub.
- [ ] `main` rejects direct pushes.
- [ ] CI passes on `main`.
- [ ] `docker compose up` starts only Postgres and LocalStack. LocalStack lists four queues.
- [ ] `cargo run -p api` serves `GET /health` with status `200`.
- [ ] Vale passes on every file in `docs/`.
- [ ] Every row in [Decisions](#decisions) has a decision record with status `accepted`.
- [ ] AWS has received the GPU quota application.

---

### M1: Contracts, database, API

**Goal:** the API accepts a query, stores it, sends a scraper job, and reports query status. Everything runs locally.

**Tasks**

- [ ] **1.1** `crates/types`: You define `Query`, `QueryStatus`, `Tweet`, `Account`, `ScraperJob`, `ClassifierJob`, `ModelInput`, `Prediction`, `MetadataVersion`, `QuestionVersion`, `ModelVersion`. Every type derives `serde` and `schemars`.
- [ ] **1.2** `crates/db`: You write `sqlx` migrations for these tables:
  - `queries`: `id`, `text`, `status`, `error`, `batch_count`, `batches_done`, `created_at`, `updated_at`
  - `accounts`: `id`, `handle`, X profile fields, `raw` (`jsonb`), `fetched_at`
  - `tweets`: `id`, `account_id`, `text`, `posted_at`, `raw` (`jsonb`), `embedding` (`vector`)
  - `query_tweets`: `query_id`, `tweet_id`
  - `predictions`: `tweet_id`, `model_version`, `metadata_version`, `question_version`, `p_bot`, `raw` (`jsonb`), `created_at`. The primary key covers the first four columns.
- [ ] **1.3** `crates/db`: You write repository functions. Every write is an upsert. You commit the `sqlx` offline data in `.sqlx/`.
- [ ] **1.4** `crates/queue`: You write typed `send`, `receive`, and `delete` functions over SQS. The functions return the receive count with each job.
- [ ] **1.5** `crates/telemetry`: You set up `tracing` with JSON logs and an OpenTelemetry exporter. A missing endpoint turns the exporter off.
- [ ] **1.6** `crates/api`: You build three endpoints with axum:
  - `POST /queries`: inserts a `queued` query, sends a scraper job, returns `202` with `{ "query_id": … }`.
  - `GET /queries/{id}`: returns the query status. When the status is `done`, it also returns the result, one page at a time.
  - `GET /health`: returns `200` when Postgres and SQS respond.
- [ ] **1.7** `crates/api`: You return every error as JSON with a stable `code` field.
- [ ] **1.8** You add the `integration` job to `ci.yml`.

**Docs:** a `README.md` for `types`, `db`, `queue`, `telemetry`, `api`. The `api` README documents every endpoint with an example HTTP request and response.

**Gate**

- [ ] An integration test sends `POST /queries` and then `GET /queries/{id}`. It sees status `queued`.
- [ ] The same test finds the scraper job in the LocalStack `scraper-jobs` queue.
- [ ] A second identical upsert leaves every table unchanged. A test proves it.
- [ ] CI runs the integration tests on every PR.

---

### M2: Workers with the hosted backend

**Goal:** a query travels the full flow locally and ends with status `done`. The hosted backend classifies.

**Tasks**

- [ ] **2.1** **Spike:** You test self-hosted Firecrawl against an X search page, a profile page, and a tweet page. You record which fields come back. You write `docs/reports/spike-firecrawl-x.md` and a decision record that settles O1.
- [ ] **2.2** `crates/scraper`: You define `trait Scraper`. You write the Firecrawl implementation and a fixture implementation. The fixture implementation reads JSON files from `crates/scraper/fixtures/`.
- [ ] **2.3** `crates/metadata`: You write metadata version `v1` (tweet text only) and question version `v1` (bot or human). The functions are pure: the same tweet and account always give the same model input.
- [ ] **2.4** `crates/classifier`: You define `trait Classifier`. You write the hosted implementation and the fake. The hosted implementation:
  - reads Cloudflare's request format for Clef-flash and records it in the crate README
  - retries on `429` and `5xx` with exponential backoff and jitter
  - stays under a configured requests-per-second limit
- [ ] **2.5** `crates/embedder`: You settle O2 with a decision record. You define `trait Embedder`. You write the hosted implementation and the fake. You add a migration that sets the `embedding` column dimension.
- [ ] **2.6** `crates/worker`, role `scraper`: the worker takes a scraper job and carries out steps 5–9 of the [request flow](#request-flow). `BATCH_SIZE` sets the batch size.
- [ ] **2.7** `crates/worker`, role `classifier`: the worker takes a classifier job and carries out steps 10–14. The batch-count update and the status change happen in one Postgres transaction.
- [ ] **2.8** `crates/worker`: You add the [failure flow](#failure-flow). A fake classifier that fails N times proves the retry and the DLQ.
- [ ] **2.9** You add the `e2e` job to `ci.yml`.
- [ ] **2.10** You add `nightly.yml` with a contract test against the hosted backend.
- [ ] **2.11** You write `scripts/smoke.sh`. It sends one real query through Firecrawl and the hosted backend on your machine.

**Docs:** a `README.md` for `scraper`, `metadata`, `classifier`, `embedder`, `worker`. The `metadata` README defines `v1` field by field. You write `docs/reports/smoke-m2.md` with one real query and its result.

**Gate**

- [ ] The e2e test passes in CI: `POST /queries` leads to status `done` with predictions for every tweet.
- [ ] A test shows a failing job reaching the DLQ and the query status becoming `failed`.
- [ ] A test shows a duplicated classifier job producing no duplicate predictions.
- [ ] `scripts/smoke.sh` completes one real query. The smoke report records it.
- [ ] Decision records settle O1 and O2.

---

### M3: First AWS deployment

**Goal:** the M2 flow runs on AWS. A merge to `main` deploys it. Datadog shows it.

**Tasks**

- [ ] **3.1** `infra/terraform`: You set up the S3 bucket for the Terraform state file. You split the configuration into modules: `network`, `database`, `queues`, `compute`, `registry`, `iam`.
- [ ] **3.2** `network`: You create one VPC, public subnets for the API, and private subnets for RDS. Security groups allow only the traffic each component needs.
- [ ] **3.3** `database`: You create RDS Postgres 16 with the `vector` extension, on the smallest instance class that runs it.
- [ ] **3.4** `queues`: You create `scraper-jobs`, `classifier-jobs`, and their DLQs, with `maxReceiveCount = 5`.
- [ ] **3.5** `registry`: You create ECR repositories. You write multi-stage Dockerfiles for `api` and `worker`.
- [ ] **3.6** `compute`: You create one EC2 instance for the API and one for the workers. Both use instance profiles. SSM Parameter Store holds the secrets. You also deploy self-hosted Firecrawl. The spike in task 2.1 shows how much memory it needs.
- [ ] **3.7** `iam`: You create an OIDC role for GitHub Actions. The role can push to ECR and send SSM commands. It cannot do anything else.
- [ ] **3.8** You write `deploy.yml`. It builds the images, pushes them to ECR, and tells each instance to pull and restart through SSM Run Command.
- [ ] **3.9** You install the Datadog agent on both instances. The agent receives OpenTelemetry traces and JSON logs. You build a first dashboard: request rate, error rate, queue depth, query time from `queued` to `done`.
- [ ] **3.10** You tag every AWS resource with `project=real-feel`. You write the teardown steps in `infra/terraform/README.md`.

The API serves plain HTTP on an Elastic IP in M3. HTTPS arrives with the ALB in M7.

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
- [ ] **4.7** You settle O4 from the eval report. You record the chosen bot threshold in a decision record.
- [ ] **4.8** You compare each metadata version's fields against the fields Firecrawl returns. You list the gaps in the eval report.
- [ ] **4.9** CI runs one experiment on a ten-item fixture eval set with the fake classifier.

**Docs:** `crates/eval/README.md`, `docs/reports/eval-m4.md`, decision records for O3 and O4 and the default metadata version.

**Gate**

- [ ] The eval report compares at least three metadata versions on the hosted backend.
- [ ] The eval report shows a calibration curve for each experiment.
- [ ] A decision record names the default metadata version, the default question version, and the bot threshold.
- [ ] Workers use the defaults from the decision record.

---

### M5: Self-hosted backend: reference server

**Goal:** Clef-flash runs on your own GPU instance. Workers can classify through it. You know what quantization costs in accuracy.

**Tasks**

- [ ] **5.1** **Spike:** You read Clef-flash's model card and code. You confirm that Cloudflare publishes the weights, and under which licence. You find how the schema head runs. You settle O5 in a decision record.
- [ ] **5.2** `crates/types`: You define the inference server contract: `POST /v1/classify` takes a batch and returns predictions with a model version. The types generate JSON Schema.
- [ ] **5.3** `inference/reference`: You build the reference server in Python with `uv`. It generates its request and response models from the JSON Schema. It applies dynamic batching with a maximum batch size and a maximum wait.
- [ ] **5.4** `inference/reference`: You load Clef-flash in the `bf16`, `int8`, and `int4` variants. The variant is a setting.
- [ ] **5.5** `infra/terraform`: You add a GPU module. It holds a launch template with an NVIDIA Deep Learning AMI, Spot instances, and an Auto Scaling Group of size 0 or 1. A script on the instance shuts the instance down after a set idle time.
- [ ] **5.6** `crates/classifier`: You write the self-hosted implementation. `CLASSIFIER_BACKEND` selects `hosted`, `self_hosted`, or `fake`.
- [ ] **5.7** You run the same experiments on each variant and on the hosted backend. You settle O6. The eval report shows:
  - agreement rate on bot or human against the hosted backend
  - mean absolute difference in bot probability
  - the change in each metric
- [ ] **5.8** You build the benchmark harness. It sends a fixed set of model inputs at batch sizes 1, 8, 32, and 64. It records throughput, p50, p95, and p99 latency, and GPU memory.
- [ ] **5.9** You enable Datadog GPU monitoring: GPU use, GPU memory, batch size, inference latency.
- [ ] **5.10** You add the `python` and `contract` jobs to `ci.yml`. You add `gpu.yml` for benchmarks.

**Docs:** `inference/reference/README.md`, `docs/reports/quantization-m5.md`, `docs/reports/benchmark-reference.md`, decision records for O5 and O6.

**Gate**

- [ ] On AWS, a query completes with `CLASSIFIER_BACKEND=self_hosted`.
- [ ] The quantization report compares every variant that fits on the chosen instance against the hosted backend.
- [ ] The benchmark report records the reference server's baseline.
- [ ] The GPU instance stops when idle. You prove it once.

---

### M6: Self-hosted backend: candle server

**Goal:** a Rust inference server that matches the reference server's output. Evidence on whether it is faster.

**Tasks**

- [ ] **6.1** **Spike:** You check whether `candle` supports Clef-flash's base architecture and quantized weights. You list what you must write yourself. You settle O7 in a decision record. If the work is too large, M6 stops here and the decision record says why.
- [ ] **6.2** `crates/inference`: You load Clef-flash's weights with `candle`. You write the schema head.
- [ ] **6.3** `crates/inference`: You serve the same `POST /v1/classify` contract. You write dynamic batching with tokio.
- [ ] **6.4** You write parity tests. Both servers classify the same model inputs. The tests pass when every bot probability differs by less than a tolerance you set in the spike.
- [ ] **6.5** You run the M5 benchmark harness on the same instance type. You settle O8 in a decision record.

**Docs:** `crates/inference/README.md`, `docs/reports/benchmark-candle.md`, decision records for O7 and O8.

**Gate**

- [ ] Parity tests pass in `gpu.yml`.
- [ ] The benchmark report compares both servers on the same hardware.
- [ ] A decision record names the default self-hosted backend.

---

### M7: Load tests and autoscaling

**Goal:** measured break points, and evidence that each change moved them.

**Tasks**

- [ ] **7.1** `loadtest/k6`: You write four scenarios:
  - `smoke`: a low rate that checks the scripts work
  - `ramp`: a rising rate until the targets fail
  - `burst`: a sudden jump in rate
  - `soak`: a steady rate for one hour
- [ ] **7.2** You run each scenario in two modes:
  - **System only:** fake scraper and fake classifier. The test measures Real Feel's own code.
  - **Full:** real Firecrawl and a real inference backend. The test measures the whole chain. It costs real money, so run it at a small scale.
- [ ] **7.3** You record the baseline break points on one API instance. You write `docs/reports/load-baseline.md`.
- [ ] **7.4** You add an ALB and an Auto Scaling Group for the API. The ALB checks `GET /health`. ACM provides the HTTPS certificate.
- [ ] **7.5** You give each worker role its own Auto Scaling Group. Each group scales on its queue's backlog per instance.
- [ ] **7.6** You scale the GPU Auto Scaling Group on the `classifier-jobs` backlog, with a minimum of 0.
- [ ] **7.7** You add connection pooling: RDS Proxy or PgBouncer. You add a read replica only if the load tests show reads limit throughput.
- [ ] **7.8** You define SLOs in Datadog: `POST /queries` p99 latency, error rate, and time from `queued` to `done`. You add alerts.
- [ ] **7.9** You run failure drills during a `soak` test:
  - stop an API instance
  - let AWS reclaim a Spot GPU instance
  - send a broken job to the DLQ
- [ ] **7.10** You re-run every scenario after each change. You add the numbers to the load test report.

**Docs:** `loadtest/k6/README.md`, `docs/reports/load-m7.md` with before-and-after break points, decision records for the pooling choice and the autoscaling policies.

**Gate**

- [ ] The load test report shows break points before and after the ALB, autoscaling, and pooling.
- [ ] Autoscaling adds and removes instances during a `ramp` test. Datadog shows it.
- [ ] The failure drills cause no lost queries.
- [ ] The SLO dashboard and alerts are live.

---

### M8: Later experiments

You plan each of these when M7 closes. Each becomes its own spike and decision record.

- **Sharding experiment.** Use Citus or Aurora Limitless if M7 shows one Postgres instance limits throughput. Measure the change with the M7 scenarios.
- **Semantic search.** `GET /search` finds tweets by meaning with pgvector and an HNSW index.
- **Time-series analysis.** Bot and human activity for one topic over time.
- **Query decomposition.** An LLM splits one query into several scraper jobs and merges their tweets.

---

## Risks

| Risk | Effect | What you do |
|------|--------|-------------|
| Self-hosted Firecrawl cannot fetch X pages | M2 stalls | Self-hosted Firecrawl lacks the managed anti-bot layer. Task 2.1 checks this first. `trait Scraper` lets you swap tools. |
| Firecrawl does not return the metadata that `v2`+ need | M4 compares versions that production cannot fill | Task 4.8 lists the gaps. The eval set has the fields even if live data does not. |
| Cloudflare does not publish the Clef-flash weights | M5 and M6 stop | Task 5.1 checks the licence before you build anything. |
| Clef-flash's schema head needs custom code | M5 takes longer | Task 5.1 answers this before you build anything. |
| `candle` lacks Clef-flash's base architecture | M6 grows large | Task 6.1 can stop M6 with a decision record. |
| AWS approves GPU quota slowly | M5 waits | Task 0.13 asks in M0. |
| GPU costs run high | Budget overrun | Spot instances, idle shutdown, an Auto Scaling Group minimum of 0, a budget alert. |
| Paid service costs run high in tests | Budget overrun | CI uses fakes. Only `nightly.yml` and manual runs call paid services. The `eval` binary prints the cost first. |
| Hand labeling takes too long | M4 has a small eval set | Label 300 accounts first. Cresci-2017 is public. |
| Cloudflare changes the hosted backend | Hosted experiments drift | `nightly.yml` catches contract changes. Every prediction stores its model version. |
