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

## Refresh the sqlx offline data

The `sqlx` macros read `.sqlx/` when `SQLX_OFFLINE=true`. You refresh `.sqlx/` after you change a SQL query. Install `sqlx-cli` once.

```bash
cargo install sqlx-cli --version 0.9.0 --locked --no-default-features --features rustls,postgres
```

With the stack running and the migrations applied, refresh the data and commit `.sqlx/`.

```bash
cargo sqlx prepare --workspace
```

CI runs `cargo sqlx prepare --workspace --check` and fails when `.sqlx/` is stale.

## Run the CI checks

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
markdownlint-cli2 "**/*.md"
lychee --config .lychee.toml "**/*.md"
vale README.md docs crates .github
scripts/check-commits.sh origin/main..HEAD
```

The last four commands need `markdownlint-cli2`, `lychee`, and `vale` on your machine.
