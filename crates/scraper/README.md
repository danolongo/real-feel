# scraper

`scraper` defines `trait Scraper` and fetches tweets and accounts for one query.

## Where it sits

The scraper worker calls `Scraper::scrape` for each scraper job. `scraper` calls self-hosted Firecrawl over HTTP, or reads a fixture file. `scraper` depends on `types`.

## Terms

See [glossary.md](../../docs/glossary.md): scraper, query, tweet, account, fixture, scraper job, worker.

## How it works

1. The scraper worker calls `scrape(query)` and receives a `Scraped` value. `Scraped` holds `accounts` and `tweets`. Every tweet's `account_id` matches an account in the same `Scraped`.
2. The Firecrawl implementation posts the query to `{FIRECRAWL_URL}/v2/search` with the prefix `site:x.com`.
3. The request asks Firecrawl for the `json` format. The request carries a prompt and a JSON schema. The schema lists `accounts` (`id`, `handle`) and `tweets` (`id`, `account_id`, `text`, `posted_at`).
4. Firecrawl returns one extracted object for each search hit. The Firecrawl implementation maps each object to `Account` and `Tweet`. It keeps the first object for each ID. It stores the full extracted object in `raw`.
5. The Firecrawl implementation returns `Error::UnknownAccount` when a tweet names an account that the extraction lacks.
6. The fixture implementation (`Fixture::load(path)`) reads a `Scraped` from a JSON file once. It returns that content for any query.

Spike 2.1 will confirm which fields Firecrawl returns and what `id` holds. Until then, the schema in step 3 is a guess.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `FIRECRAWL_URL` | `http://localhost:3002` | The base URL of self-hosted Firecrawl. The caller reads it and passes it to `Firecrawl::new`. |

`crates/scraper/fixtures/default.json` holds 5 tweets from 3 accounts. The IDs and handles are made up.

## Run it

`scraper` is a library. It has no binary.

## Test it

Run from the repository root. No test needs Docker or network access. `wiremock` plays Firecrawl.

```bash
cargo test -p scraper
```

## Failure modes

- Firecrawl is down or answers with an error status. `scrape` returns `Error::Http`. The worker leaves the scraper job in the queue, and SQS delivers the job again.
- The extraction does not match the schema. `scrape` returns `Error::Json`.
- A tweet names an unknown account. `scrape` returns `Error::UnknownAccount`.
- The fixture file is missing or invalid. `Fixture::load` returns `Error::Io` or `Error::Json`.
