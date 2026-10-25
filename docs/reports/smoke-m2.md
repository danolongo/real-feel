# Smoke test M2

Date: 2026-10-25. Result: pass. `scripts/smoke.sh` sent one real query through twitterapi.io and the hosted backend, and the query ended with status `done`.

| Field | Value |
|-------|-------|
| Query text | `what are people saying about AI?` |
| Query ID | `f075ea80-4f38-4de3-b96b-c1aee25cae21` |
| Tweets | 100 |
| Tweets without a bot probability | 0 |
| Batch count | 4 (`BATCH_SIZE` 25) |
| Run time | 58 seconds, from `POST /queries` to `done` |

## Where the time went

- The scraper fetched 5 pages of 20 tweets. The free tier allows one request every 5 seconds, so the scrape took about 25 seconds.
- The classifier sent 100 requests to Clef-flash. `CLASSIFIER_RPS` 5 spaces them 200 ms apart, so the classification took about 20 seconds.

## Findings

- The scraper sends the query text to X search as it is. X search matches words, not questions, so a question in natural language returns loosely related tweets. This run returned Tesla earnings call transcripts. A later milestone can turn the query text into search terms.
- One account posted the same link format for four transcripts. Clef-flash with metadata version `v1` gave each tweet a bot probability near 0.05. `v1` reads the tweet text only. M4 measures whether the account fields change this.
