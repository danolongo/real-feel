# Local development

This file takes you from a clean machine to a running stack. Run every command from the repository root.

## Prerequisites

- `rustup`. It installs the toolchain that `rust-toolchain.toml` pins.
- Docker with Compose v2. It runs Postgres and LocalStack.

## Set up

1. Copy the environment file.

   ```bash
   cp .env.example .env
   ```

   Open `.env` and give `POSTGRES_PASSWORD` a value. Never commit `.env`.

   `.env.example` also lists the worker variables: `SCRAPER`, `TWITTERAPI_KEY`, `TWITTERAPI_PAGES`, `TWITTERAPI_INTERVAL_SECS`, `CLASSIFIER`, `EMBEDDER`, `CLASSIFIER_RPS`, `BATCH_SIZE`, `JOB_DEADLINE_SECS`, `SCRAPER_DLQ_URL`, and `CLASSIFIER_DLQ_URL`. Their defaults run the worker locally. Set `SCRAPER=fixture`, `CLASSIFIER=fake`, and `EMBEDDER=fake` to run with no network. Set `TWITTERAPI_KEY` only when `SCRAPER` is `twitterapi`. Set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` only when `CLASSIFIER` or `EMBEDDER` is `hosted`.

2. Start Postgres (port 5433) and LocalStack (port 4566).

   ```bash
   docker compose up -d
   ```

3. Check that LocalStack created the four queues.

   ```bash
   docker compose exec localstack awslocal sqs list-queues
   ```

   The output lists `scraper-jobs`, `scraper-jobs-dlq`, `classifier-jobs`, and `classifier-jobs-dlq`.

4. Run the API. It listens on `API_ADDR`, default `0.0.0.0:8080`.

   ```bash
   cargo run -p api
   ```

5. In a second terminal, request the health route.

   ```bash
   curl -i http://localhost:8080/health
   ```

   The API answers `HTTP/1.1 200 OK`.

`docker compose down` stops the containers. Add `-v` to delete the data too.

## Run the tests

Unit tests need no Docker.

```bash
cargo test --workspace
```

Integration tests need the stack from step 2 and the variables in `.env.example`. Crates with integration tests declare the `integration` feature.

1. Export the variables from `.env`. Set `DATABASE_URL` with your `POSTGRES_PASSWORD`.

   ```bash
   set -a; source .env; set +a
   ```

2. Apply the migrations.

   ```bash
   sqlx migrate run --source crates/db/migrations
   ```

3. Run the integration tests of one crate.

   ```bash
   cargo test -p db --features integration
   ```

CI runs the integration tests of every crate that declares the feature.

## Run the end-to-end test

The end-to-end test starts the API and both worker roles in one process. It uses the fixture scraper, the fake classifier, and the fake embedder, and creates its own queues. It posts a query over HTTP and waits up to 60 seconds for the status `done`. Then it checks for a prediction on every tweet. It needs the stack from step 2 of "Set up", the exported variables, and the applied migrations from "Run the tests".

```bash
cargo test -p worker --features e2e
```

CI runs the same command in the `e2e` job.

## Compile the SQL checks

The `sqlx` macros check every SQL query at compile time against the database in `DATABASE_URL`. With the stack down, set `SQLX_OFFLINE=true`, and the macros read your local `.sqlx/` instead. Git ignores `.sqlx/`, so it never leaves your machine. Install `sqlx-cli` once.

```bash
cargo install sqlx-cli --version 0.9.0 --locked --no-default-features --features rustls,postgres
```

With the stack running and the migrations applied, refresh `.sqlx/` after you change a SQL query.

```bash
cargo sqlx prepare --workspace
```

CI has no `.sqlx/`. CI starts Postgres, applies the migrations, and compiles against it.

## Run the CI checks

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
markdownlint-cli2 "**/*.md"
lychee --config .lychee.toml "**/*.md"
scripts/check-commits.sh origin/main..HEAD
```

The markdown and link checks need `markdownlint-cli2` and `lychee` on your machine.
