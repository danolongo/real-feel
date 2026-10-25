# scraper

`scraper` defines `trait Scraper` and fetches tweets and accounts for one query.

## Where it sits

The scraper worker calls `Scraper::scrape` for each scraper job. `scraper` calls [twitterapi.io](../../docs/glossary.md) over HTTP, or reads a fixture file. `scraper` depends on `types`.

## Terms

See [glossary.md](../../docs/glossary.md): scraper, query, tweet, account, fixture, scraper job, worker.

## How it works

1. The scraper worker calls `scrape(query)` and receives a `Scraped` value. `Scraped` holds `accounts` and `tweets`. Every tweet's `account_id` matches an account in the same `Scraped`.
2. `TwitterApi::new(base_url, key, pages, interval)` builds the twitterapi.io implementation. Production uses `https://api.twitterapi.io` as `base_url`. Tests use a `wiremock` server and a zero `interval`.
3. `TwitterApi` sends `GET {base_url}/twitter/tweet/advanced_search` with the header `X-API-Key` and these query parameters:
   - `query`: `{query} lang:en -filter:retweets`
   - `queryType`: `Latest`
   - `cursor`: empty for the first page, then the `next_cursor` of the previous page
4. `TwitterApi` reads at most `pages` pages. It stops early when `has_next_page` is `false`. It waits `interval` between two requests.
5. Every tweet in a search result carries its author, so `TwitterApi` makes no profile call.
6. `TwitterApi` builds one `Account` for each distinct author ID and one `Tweet` for each distinct tweet ID. It keeps the first one it sees. `raw` holds the full tweet object or the full author object.
7. The fixture implementation (`Fixture::load(path)`) reads a `Scraped` from a JSON file once. It returns that content for any query.

### Fields

| Type | Field | twitterapi.io field |
|------|-------|---------------------|
| `Tweet` | `id` | `id` |
| `Tweet` | `account_id` | `author.id` |
| `Tweet` | `text` | `text` |
| `Tweet` | `posted_at` | `createdAt` |
| `Account` | `id` | `author.id` |
| `Account` | `handle` | `author.userName` |
| `Account` | `joined_at` | `author.createdAt` |
| `Account` | `followers` | `author.followers` |
| `Account` | `following` | `author.following` |
| `Account` | `tweet_count` | `author.statusesCount` |
| `Account` | `is_verified` | `author.isBlueVerified` |
| `Account` | `is_automated` | `author.isAutomated` |

Tweet IDs and account IDs hold X's numeric IDs as text.

### Dates

Search results use the format `Sun Oct 25 09:56:57 +0000 2026`. `TwitterApi` parses it with `%a %b %d %H:%M:%S %z %Y` and stores UTC. A date that does not match makes `scrape` return `Error::Date`.

### Price and rate limit

- Price: $0.15 per 1,000 tweets, and at least $0.00015 per call. A new account gets up to $1 of credit. See [the spike report](../../docs/reports/spike-x-data.md).
- Rate limit: the free tier allows one request every 5 seconds. A second request inside that window returns `429`. The default `interval` of 5 seconds respects it. When a request still gets `429`, `TwitterApi` waits `interval` and sends the same request again. It makes at most 3 attempts for each page.
- One page holds up to 20 tweets. The default of 5 pages costs about $0.015 for each query.

## Configuration

The scraper reads no environment variable. The worker reads these and passes them to `TwitterApi::new`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `TWITTERAPI_KEY` | none | The twitterapi.io key. |
| `TWITTERAPI_PAGES` | `5` | The most search pages for each query, 20 tweets each. |
| `TWITTERAPI_INTERVAL_SECS` | `5` | The pause between two requests. |

`crates/scraper/fixtures/default.json` holds 5 tweets from 3 accounts. The IDs, handles, and account fields are made up.

## Run it

`scraper` is a library. It has no binary.

## Test it

Run from the repository root. No test needs Docker or network access. `wiremock` plays twitterapi.io.

```bash
cargo test -p scraper
```

## Failure modes

- twitterapi.io is down or rejects the key. `scrape` returns `Error::Http`.
- twitterapi.io answers `429` on all 3 attempts for a page. `scrape` returns `Error::Http`. The worker leaves the scraper job in the queue, and SQS delivers the job again.
- A tweet or author lacks a required field. `scrape` returns `Error::Json`.
- A date does not match the format. `scrape` returns `Error::Date`.
- The fixture file is missing or invalid. `Fixture::load` returns `Error::Io` or `Error::Json`.
