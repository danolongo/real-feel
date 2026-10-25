# types

`types` holds the shapes that the API, the workers, and Postgres share.

## Where it sits

Every other crate depends on `types`. `types` depends on no crate in the workspace.

## Terms

See [glossary.md](../../docs/glossary.md): query, query status, tweet, account, scraper job, classifier job, model input, prediction, model version, metadata version, question version.

## How it works

1. Each type derives `serde`, so the API and SQS jobs use the same JSON.
2. `QueryStatus` maps to the Postgres enum `query_status`.
3. The three version types wrap a `String`. They serialize as a plain string in JSON and in Postgres.
4. A job carries IDs only. Postgres holds the data.
5. `trace_context` holds a W3C `traceparent` value, so one trace follows a query across the API and the workers.
6. `ModelInput` holds one model input. The `metadata` crate builds it. The classifier sends it to the inference backend.

No type derives `schemars` until task 6.2.

## Run it

`types` is a library. It has no binary.

## Test it

From the repository root:

```bash
cargo test -p types
```

The tests need no network access and no Docker.
