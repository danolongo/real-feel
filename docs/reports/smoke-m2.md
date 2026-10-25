# Smoke test M2

Date: 2026-10-25. Result: pass. `scripts/smoke.sh` sent two real queries through twitterapi.io and the hosted backend. Both ended with status `done`.

| Field | Run 1 | Run 2 |
|-------|-------|-------|
| Query text | `what are people saying about AI?` | `Lebron 76ers` |
| Query ID | `f075ea80-4f38-4de3-b96b-c1aee25cae21` | `310ee44b-7f87-409d-b4e5-720fb71edb70` |
| Tweets | 100 | 100 |
| Accounts | not logged | 92 |
| Tweets without a bot probability | 0 | 0 |
| Batch count | 4 (`BATCH_SIZE` 25) | 4 (`BATCH_SIZE` 25) |
| Run time, from `POST /queries` to `done` | 58 seconds | 51 seconds |

## Where the time went

The run log of run 2 times each step.

| Step | Time | Why |
|------|------|-----|
| API to scraper worker | 0.1 s | |
| Scrape 5 pages | 30.4 s | Each request took about 2 seconds. The scraper then paused 5 seconds for the free tier. |
| Save tweets and accounts | 0.5 s | |
| Classify 4 batches | 20.3 s | 100 Clef-flash requests at `CLASSIFIER_RPS` 5. The 4 batches share that limit. |

Every Clef-flash request succeeded on its first attempt. twitterapi.io returned no `429`.

After run 2, the scraper counts `TWITTERAPI_INTERVAL_SECS` from the start of each request, not from the end of the response. That cuts the scrape of 5 pages from about 30 seconds to about 22. With a paid twitterapi.io account and an interval of 0, the scrape takes about 10 seconds. `CLASSIFIER_RPS` is Real Feel's own limit, not Cloudflare's. M5 finds the real ceiling.

## Findings

- The scraper sends the query text to X search as it is. X search matches words, not questions. Run 1 asked a question and returned Tesla earnings call transcripts. Run 2 used keywords and returned tweets on the topic. A later milestone can turn the query text into search terms.
- Every bot probability in run 2 fell between 0.01 and 0.32. Run 1 gave four transcripts from one account about 0.05 each. Metadata version `v1` reads the tweet text only. M4 measures whether the account fields change this.
