# db

`db` holds the migrations and the repository functions. They are the only code that runs SQL.

## Where it sits

The API and the workers call `db`. `db` calls Postgres through `sqlx` and uses the shapes in [`types`](../types/README.md).

## Terms

See [glossary.md](../../docs/glossary.md): query, query status, tweet, account, prediction, model version, metadata version, question version, batch, batch number, batch count, result, experiment, eval set, repeat, label, tune split, test split, Postgres.

## How it works

1. `MIGRATOR` holds the migrations in `migrations/`. You run it once against a database.
2. Each repository function takes any `sqlx` executor: a pool or a transaction. A worker passes a transaction when several writes must commit together.
3. `sqlx` checks every SQL string at compile time against the database in `DATABASE_URL`. With `SQLX_OFFLINE=true`, `sqlx` reads your local `.sqlx/` instead. Git ignores `.sqlx/`.
4. `insert_query` creates a `queued` query with a new ID. It is never an upsert.
   `create_or_reuse_query` carries out query reuse. In one transaction, it takes an advisory lock on the hash of the lower-cased text. It then looks for the newest query younger than the reuse window with the same lower-cased text and a status other than `failed`. It returns `CreatedQuery::Reused` with that query, or inserts a query and returns `CreatedQuery::New`. The lock gives two identical queries sent at once one query. A zero window skips the lookup and always inserts. The index on `LOWER(text)` and `created_at` serves the lookup.
5. `set_status` runs one conditional `UPDATE`. It returns `true` when Postgres applied the change and `false` when the current status forbids it. For `failed`, the caller passes the error message.
6. Each upsert writes with `ON CONFLICT … DO UPDATE … WHERE (columns) IS DISTINCT FROM (excluded columns)`. It returns the number of changed rows. A second call with the same input returns `0`, and `updated_at` stays the same. `upsert_query_tweet` and `upsert_query_batch` use `DO NOTHING`, because no column can change.
7. `accounts` holds the typed columns `joined_at`, `followers`, `following`, `tweet_count`, `is_verified`, and `is_automated`. All six are nullable. `upsert_account` writes them, and its `IS DISTINCT FROM` check compares them. `result_page` and `tweets_with_accounts` read them.
8. `result_page` returns the tweets of one query, sorted by tweet ID. Each tweet carries its account and the prediction that matches the requested model version, metadata version, and question version. A tweet without a matching prediction carries none. The caller passes the last tweet ID of the previous page as `after`.
9. `set_batch_count` records the batch count and sets the query status to `classifying` in one `UPDATE`. Postgres applies it only while the status is `scraping` and the batch count is null or equal to the new count. It returns whether Postgres applied it.
10. `query_tweet_ids` returns the tweet IDs of one query, sorted by tweet ID. `tweets_with_accounts` returns the tweets and accounts for a list of tweet IDs.
11. `query_batch_exists` returns whether a `query_batches` row exists for one query and batch number. The classifier worker uses it to skip a repeated classifier job.
12. `tweet_ids_without_prediction` returns the tweet IDs, in the order given, that have no prediction for the given model version, metadata version, and question version.
13. `finish_batch` runs one transaction:
    1. It locks the query row with `SELECT … FOR UPDATE`.
    2. It upserts the predictions with `upsert_prediction`.
    3. It upserts the `query_batches` row.
    4. It counts the `query_batches` rows.
    5. When the batch count is not null and equals the count, it sets the query status to `done`.

    It returns whether it set `done`. The lock makes two workers that finish the last two batches at once count one after the other. A repeated call changes no row.
14. `ping` runs `SELECT 1`. The API uses it for the readiness check.
15. `insert_experiment` inserts one `experiments` row with a new ID and returns it. The row holds the three versions, the eval set checksum, the seed, and the repeat count. `fetch_experiment` returns one experiment by ID. `list_experiments` returns all experiments, oldest first.
16. `insert_experiment_results` inserts many `experiment_results` rows for one experiment in one `INSERT … SELECT FROM UNNEST` statement. A row holds the label, the split, the bot probability, and the LLM text fields of one eval set account and one repeat. `experiment_results` returns the rows of one experiment, sorted by account ID and repeat.
17. The two experiment tables have no foreign key to `tweets`, `accounts`, or `predictions`. An experiment never changes a production row. Postgres checks that the label is `human`, `bot`, or `agent` and that the split is `tune` or `test`.

### Status changes

Postgres applies a status change only from the statuses this table allows. Any other change is a no-op, and `set_status` returns `false`.

| New status | Allowed current statuses | Who sets it |
|------------|--------------------------|-------------|
| `scraping` | `queued`, `scraping` | scraper worker. `scraping` → `scraping` lets a retried scraper job continue. |
| `classifying` | `scraping` | scraper worker |
| `done` | `scraping` (zero tweets), `classifying` | scraper worker or classifier worker |
| `failed` | `queued`, `scraping`, `classifying` | API or any worker |

`done` and `failed` are final. Nothing moves a query out of them.

## SQL style

- You write SQL keywords, functions, and type names in capitals: `SELECT`, `ON CONFLICT`, `NOW()`, `ANY()`, `TIMESTAMPTZ`.
- You write table names, column names, aliases, and enum values in lower case.

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
