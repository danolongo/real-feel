# api

The API is the Rust HTTP server that accepts queries and returns results. Today it serves only `GET /health`, which answers `200` with an empty body. M1 adds the query endpoints and the calls to Postgres and SQS.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `API_ADDR` | `0.0.0.0:8080` | The address the API binds. |
| `RUST_LOG` | `info` | The log filter, in `tracing-subscriber` syntax. |

## Run it

From the repository root:

```bash
cargo run -p api
```

## Test it

From the repository root. The tests need no Docker and no network.

```bash
cargo test -p api
```

## Failure modes

The API exits with the error when `API_ADDR` is invalid or already in use. You free the port or set another `API_ADDR`.
