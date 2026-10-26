# api

The API is the Rust HTTP server that accepts queries and returns results. It calls Postgres through [`db`](../db/README.md) and sends scraper jobs through [`queue`](../queue/README.md). It runs no SQL and makes no SQS call itself.

## Where it sits

The user sends queries to the API. The API inserts each query into Postgres and sends a scraper job to the `scraper-jobs` queue. The scraper worker and the classifier worker do the rest. The user polls the API for the result.

## Terms

See [glossary.md](../../docs/glossary.md): API, query, query status, query reuse, tweet, account, prediction, bot probability, bot threshold, result, page, cursor, scraper job, trace context, liveness check, readiness check, error code, API key, query quota.

## How it works

1. `main` calls `telemetry::init()`. It then reads the configuration and connects to Postgres. It runs `db::MIGRATOR`, and then binds `API_ADDR`. `db::MIGRATOR` skips migrations that Postgres already holds, so a restart changes nothing.
2. Before `POST /queries` and `GET /queries/{id}` run, the API checks the API key. See [Authentication and limits](#authentication-and-limits).
3. `POST /queries` trims the text and checks its length. In one transaction, it checks the query quota and then checks for query reuse. On a match, it returns the earlier query's ID and sends no scraper job. Otherwise it inserts a `queued` query with the ID of the API key and sends a scraper job. The job carries the trace context of the request.
4. `GET /queries/{id}` returns the query status. When the status is `done`, it also returns one page of the result.
5. The page holds one prediction for each tweet. The API selects it with `DEFAULT_MODEL_VERSION`, `DEFAULT_METADATA_VERSION`, and `DEFAULT_QUESTION_VERSION`. A tweet without a matching prediction has `"p_bot": null`.
6. Each tweet also carries `is_bot`. It is `true` when `p_bot` is at or above `BOT_THRESHOLD`, and `false` below it. It is `null` when `p_bot` is `null`.
7. Every error is JSON: `{ "code": …, "message": … }`.

### `POST /queries`

The request needs an API key. The text must hold 1 to 512 characters after trimming. The API stores the trimmed text.

A query reuses an earlier query when the texts match after trimming and lower-casing, the earlier query is younger than `QUERY_REUSE_SECS`, and its status is not `failed`. The earlier query can still be in progress. The response is the same `202` with the earlier query's ID.

```http
POST /queries HTTP/1.1
Authorization: Bearer <API key>
Content-Type: application/json

{ "text": "what are people saying about the new phone?" }
```

```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{ "query_id": "7b0e5c3e-52b4-4a58-8f0c-9d5a1f2a6c11" }
```

If the API key has used its query quota, the API returns `429` with `quota_exceeded` and a `Retry-After` header. A reused query does not count against the query quota.

If the API cannot send the scraper job, it sets the query status to `failed`, records the error, and returns `503` with `queue_unavailable`.

### `GET /queries/{id}`

The request needs an API key. Query parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `limit` | `50` | The number of tweets in the page. From 1 to 200. |
| `after` | none | The cursor: the last `tweet_id` of the previous page. It must not be empty. |

The API sorts tweets by `tweet_id`. `next_cursor` is `null` on the last page. Until the status is `done`, `result` is `null`.

```http
GET /queries/7b0e5c3e-52b4-4a58-8f0c-9d5a1f2a6c11 HTTP/1.1
Authorization: Bearer <API key>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "query_id": "7b0e5c3e-52b4-4a58-8f0c-9d5a1f2a6c11",
  "status": "queued",
  "error": null,
  "result": null
}
```

A `done` query with `limit=2`:

```http
GET /queries/7b0e5c3e-52b4-4a58-8f0c-9d5a1f2a6c11?limit=2 HTTP/1.1
Authorization: Bearer <API key>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "query_id": "7b0e5c3e-52b4-4a58-8f0c-9d5a1f2a6c11",
  "status": "done",
  "error": null,
  "result": {
    "tweets": [
      {
        "tweet_id": "1801",
        "text": "The new phone is fine.",
        "posted_at": "2026-10-01T09:30:00Z",
        "account": { "id": "42", "handle": "alice" },
        "p_bot": 0.04,
        "is_bot": false
      },
      {
        "tweet_id": "1802",
        "text": "BUY NOW best phone deal",
        "posted_at": null,
        "account": { "id": "77", "handle": "deals_bot_9" },
        "p_bot": 0.97,
        "is_bot": true
      }
    ],
    "next_cursor": "1802"
  }
}
```

A `failed` query returns its error in `error`:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "query_id": "7b0e5c3e-52b4-4a58-8f0c-9d5a1f2a6c11",
  "status": "failed",
  "error": "SQS request failed: …",
  "result": null
}
```

### `GET /health`

The liveness check. It needs no API key. It returns `200` with an empty body while the process runs. It checks no dependency.

```http
GET /health HTTP/1.1
```

```http
HTTP/1.1 200 OK
```

### `GET /ready`

The readiness check. It needs no API key. It returns `200` when Postgres and SQS respond. It returns `503` with an empty body when either does not.

```http
GET /ready HTTP/1.1
```

```http
HTTP/1.1 503 Service Unavailable
```

### Error codes

| Code | HTTP status | When |
|------|-------------|------|
| `unauthorized` | 401 | The `Authorization: Bearer` header is missing, or the API key is unknown or revoked. |
| `invalid_query` | 400 | The query text is empty, longer than 512 characters after trimming, or missing from the body. |
| `invalid_query_id` | 400 | The ID in the path is not a UUID. |
| `invalid_cursor` | 400 | `limit` is not a number from 1 to 200, or `after` is empty. |
| `query_not_found` | 404 | No query has that ID. |
| `quota_exceeded` | 429 | The API key has created its query quota of queries in the past hour. |
| `queue_unavailable` | 503 | SQS rejected the scraper job. |
| `internal` | 500 | Any other failure. The message hides internal detail. |

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "code": "invalid_query",
  "message": "the query text must hold 1 to 512 characters after trimming"
}
```

## Authentication and limits

### Who can call the API

Every `POST /queries` and `GET /queries/{id}` request carries an API key in the `Authorization: Bearer <API key>` header. The API hashes the key with SHA-256 and looks up the hash in the `api_keys` table. A missing, unknown, or revoked key gets `401` with `unauthorized`. The check runs before the API reads the body, so a bad key costs nothing.

`GET /health` and `GET /ready` need no key, so the load balancer can check them. Postgres stores only the hash of each key. Nobody can read a key back, not even you.

### What costs money

Each new query makes the scraper worker call twitterapi.io. The classifier worker then calls the hosted backend. Both bill by use. A reused query creates no new query, so it costs nothing. `GET /queries/{id}` calls neither service.

### What each limit stops

| Limit | Set by | What it stops |
|-------|--------|---------------|
| API key | `api-key create` | Anyone without a key. You revoke one key and leave the others working. |
| Query quota | `--quota` of `api-key create` | One key that creates too many new queries. The quota counts the key's queries of the past hour, so it caps the spend of that key. |
| WAF rate limit | `infra/terraform` | One IP address that sends too many requests, including requests the API rejects. It covers the routes that need no key. |

The query quota is the only limit that counts queries. The WAF counts requests.

The API counts in Postgres, so every API instance reads the same count. Two requests of one key at the same time wait in a line. The second request sees the first request's query. No key passes its quota with parallel requests.

A key over its query quota gets `429` with `quota_exceeded`. The `Retry-After` header holds the seconds until the oldest counted query leaves the hour.

### Switch the query quota off

Set `QUERY_QUOTA_ENABLED=false` and restart the API. The API still checks every key. It stops counting queries. Load tests use this setting. `true` or no value turns the query quota on. Any other value stops the API at startup with an error.

### Manage keys

The `api-key` binary sits next to `api` in the image. It reads `DATABASE_URL`, the same variable as the API.

```bash
api-key create --name <owner> --quota <new queries per hour>
api-key revoke --id <id>
```

`create` prints the ID and the key. It prints the key once. Copy it then, because no command shows it again. `revoke` sets `revoked_at`, and the key gets `401` at once.

In production, RDS accepts connections only from the API and worker security groups. You run `api-key` inside the API container, through an SSM session on the API instance:

1. Open the session. Replace `<instance-id>` with the ID of the API instance.

   ```bash
   aws ssm start-session --target <instance-id>
   ```

2. Run `api-key` in the API container. The container name `api` is the name the restart script gives it. Check it with `docker ps`.

   ```bash
   sudo docker exec api api-key create --name alice --quota 20
   sudo docker exec api api-key revoke --id <id>
   ```

The container holds `DATABASE_URL`, so you pass no connection string.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `API_ADDR` | `0.0.0.0:8080` | The address the API binds. |
| `DATABASE_URL` | none | The Postgres connection string. |
| `SCRAPER_QUEUE_URL` | none | The URL of the `scraper-jobs` queue. |
| `QUERY_REUSE_SECS` | `3600` | How long a query stays reusable, in seconds. `0` turns query reuse off. |
| `QUERY_QUOTA_ENABLED` | `true` | `true` counts each key's queries against its query quota. `false` stops the count. Any other value stops the API at startup. |
| `DEFAULT_MODEL_VERSION` | none | The model version of the predictions in a result. |
| `DEFAULT_METADATA_VERSION` | none | The metadata version of the predictions in a result. |
| `DEFAULT_QUESTION_VERSION` | none | The question version of the predictions in a result. |
| `BOT_THRESHOLD` | none | The bot threshold, a number from 0 to 1. The API sets `is_bot` to `true` when `p_bot` is at or above it. |
| `RUST_LOG` | Real Feel's crates at `info`, dependencies at `warn` | The log filter. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | The OTLP collector URL. Unset turns the exporter off. |
| `AWS_ENDPOINT_URL`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | none | The AWS client settings. See [`queue`](../queue/README.md). |

## Run it

From the repository root. Postgres and LocalStack must run. The API applies the migrations of `db` at startup.

```bash
export DATABASE_URL=postgres://realfeel:$POSTGRES_PASSWORD@localhost:5433/realfeel
export SCRAPER_QUEUE_URL=http://localhost:4566/000000000000/scraper-jobs
export DEFAULT_MODEL_VERSION=clef-flash-bf16 DEFAULT_METADATA_VERSION=v1 DEFAULT_QUESTION_VERSION=v1
export BOT_THRESHOLD=0.5
cargo run -p api
```

In a second terminal, create an API key. The command prints the key once. Send it in the `Authorization` header.

```bash
cargo run -p api --bin api-key -- create --name dev --quota 100
curl -s localhost:8080/queries -H "Authorization: Bearer <key>" \
  -H 'Content-Type: application/json' -d '{"text": "what do people say about AI?"}'
```

## Test it

From the repository root. The tests need no Docker and no network.

```bash
cargo test -p api
```

The integration tests need Postgres and LocalStack from `docker compose up -d --wait`. Each test gets its own database and its own queue with a unique name.

```bash
AWS_ENDPOINT_URL=http://localhost:4566 AWS_REGION=us-east-1 \
AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test \
cargo test -p api --features integration
```

## Failure modes

- A configuration variable is missing: the API exits and names the variable.
- `QUERY_QUOTA_ENABLED` is neither `true` nor `false`: the API exits and names the variable.
- You lose a key: no command shows it. You revoke the key and create a new one.
- A migration fails, or Postgres is down at startup: the API exits with the error before it binds the port. You read the error, fix the database, and start the API again.
- `API_ADDR` is invalid or already in use: the API exits with the error. You free the port or set another `API_ADDR`.
- Postgres is down: `GET /health` still returns `200`, `GET /ready` returns `503`, and the query endpoints return `500` with `internal`.
- SQS is down: `GET /ready` returns `503`. `POST /queries` returns `503` with `queue_unavailable` and sets the query status to `failed`.
