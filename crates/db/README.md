# db

`db` holds the migrations and the repository functions. They are the only code that runs SQL.

## Where it sits

The API and the workers call `db`. `db` calls Postgres through `sqlx` and uses the shapes in [`types`](../types/README.md).

## Terms

See [glossary.md](../../docs/glossary.md): query, query status, tweet, account, prediction, model version, metadata version, question version, batch, result, Postgres.

## How it works

1. `MIGRATOR` holds the migrations in `migrations/`. You run it once against a database.
2. Each repository function takes any `sqlx` executor: a pool or a transaction. A worker passes a transaction when several writes must commit together.
3. `sqlx` checks every SQL string at compile time against the database in `DATABASE_URL`. With `SQLX_OFFLINE=true`, `sqlx` reads your local `.sqlx/` instead. Git ignores `.sqlx/`.
4. `insert_query` creates a `queued` query with a new ID. It is never an upsert.
5. `set_status` runs one conditional `UPDATE`. It returns `true` when Postgres applied the change and `false` when the current status forbids it. For `failed`, the caller passes the error message.
6. Each upsert writes with `ON CONFLICT … DO UPDATE … WHERE (columns) IS DISTINCT FROM (excluded columns)`. It returns the number of changed rows. A second call with the same input returns `0`, and `updated_at` stays the same. `upsert_query_tweet` and `upsert_query_batch` use `DO NOTHING`, because no column can change.
7. `result_page` returns the tweets of one query, sorted by tweet ID. Each tweet carries its account and the prediction that matches the requested model version, metadata version, and question version. A tweet without a matching prediction carries none. The caller passes the last tweet ID of the previous page as `after`.
8. `ping` runs `select 1`. The API uses it for the readiness check.

### Status changes

Postgres applies a status change only from the statuses this table allows. Any other change is a no-op, and `set_status` returns `false`.

| New status | Allowed current statuses | Who sets it |
|------------|--------------------------|-------------|
| `scraping` | `queued`, `scraping` | scraper worker. `scraping` → `scraping` lets a retried scraper job continue. |
| `classifying` | `scraping` | scraper worker |
| `done` | `scraping` (zero tweets), `classifying` | scraper worker or classifier worker |
| `failed` | `queued`, `scraping`, `classifying` | API or any worker |

`done` and `failed` are final. Nothing moves a query out of them.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATABASE_URL` | none | Postgres connection string. `sqlx` reads it at compile time unless `SQLX_OFFLINE=true`, and `cargo sqlx` and the integration tests read it at run time. |

`sqlx` uses `rustls` for TLS, so `db` links no C TLS library.

## Run it

`db` is a library. It has no binary. After you change a SQL string or a migration, apply the migrations and refresh your local `.sqlx/` from the repository root:

```bash
export DATABASE_URL=postgres://realfeel:$POSTGRES_PASSWORD@localhost:5433/realfeel
cargo sqlx migrate run --source crates/db/migrations
cargo sqlx prepare --workspace
```

`.sqlx/` stays on your machine. CI compiles against a live Postgres instead.

## Test it

From the repository root:

```bash
cargo test -p db
```

This runs no database test. The integration tests need Postgres from `docker compose up -d --wait` and `DATABASE_URL`:

```bash
cargo test -p db --features integration
```

Each test gets its own database, so tests run in parallel and never collide.

## Failure modes

- Postgres is down: each function returns `sqlx::Error`. The caller decides whether to retry.
- A forbidden status change: `set_status` returns `false` and changes nothing.
- Your local `.sqlx/` is stale: the build with `SQLX_OFFLINE=true` fails. You run `cargo sqlx prepare --workspace`.

## Decisions

Postgres as the single database. Upserts for every worker write.
