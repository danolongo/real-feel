# embedder

`embedder` turns tweet text into embeddings. It has a hosted implementation and a fake.

## Where it sits

The classifier worker calls `Embedder::embed` in step 12 of the request flow. `embedder` calls Cloudflare Workers AI over HTTPS. It depends on no other Real Feel crate.

## Terms

See [glossary.md](../../docs/glossary.md): embedder, embedding, embedding model, fake, worker.

## How it works

1. The worker builds `Hosted` with `Hosted::from_env()`. Tests build it with `Hosted::new(base_url, account_id, api_token)`.
2. `embed(texts)` sends one `POST` to `{base_url}/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-small-en-v1.5`.
3. The request carries the header `Authorization: Bearer {api_token}` and the body `{"text": ["...", "..."]}`.
4. Workers AI answers `{"result": {"shape": [n, 384], "data": [[...], ...], "pooling": "mean"}, "success": true, ...}`. `embed` reads `result.data`.
5. `embed` retries on `429` and on `5xx`. It makes at most 5 attempts. Before attempt `n + 1` it sleeps a random time between 0 and `500 ms * 2^(n-1)` (exponential backoff with full jitter).
6. `embed` checks that it got one embedding of `DIMENSIONS` numbers for each text. Otherwise it returns `Error::Shape`.
7. `Fake` hashes each text into a seed and derives `DIMENSIONS` numbers from the seed. Equal texts give equal embeddings. `Fake` makes no network calls.

The embedding model is `@cf/baai/bge-small-en-v1.5`. It returns 384 dimensions, so `DIMENSIONS` is 384. It uses `mean` pooling, the default. It reads at most 512 tokens for each text.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `CLOUDFLARE_ACCOUNT_ID` | none | The Cloudflare account that owns the Workers AI quota. |
| `CLOUDFLARE_API_TOKEN` | none | The token with Workers AI access. |

## Run it

`embedder` is a library. It has no binary.

## Test it

From the repository root:

```bash
cargo test -p embedder
```

The tests need no Docker and no network. Wiremock plays Cloudflare.

## Failure modes

- A `429` or `5xx` answer triggers a retry. After 5 attempts, `embed` returns `Error::Status`.
- Any other non-success status returns `Error::Status` at once.
- A connection failure or an unreadable body returns `Error::Http`.
- A wrong embedding count or length returns `Error::Shape`.
- A missing variable makes `from_env` return `Error::MissingVariable`.
- The pgvector column must use the same `DIMENSIONS`. A different embedding model needs a new migration.
