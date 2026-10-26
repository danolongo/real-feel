# eval

`eval` builds eval sets, runs experiments, and writes eval reports. This part holds the pure core: the eval set format, the manifest, the splits, the metrics, and the report renderer.

## Where it sits

`eval` is a library and a binary. It depends on `types` and `scraper`. It does not call Postgres, the classifier, or `metadata` yet. A later task wires `eval experiment` to them.

## Terms

See [glossary.md](../../docs/glossary.md): eval set, label, agent account, recent tweets, LLM text, label agent, manifest, tune split, test split, reliability bin, calibration curve, experiment, eval report.

## How it works

1. `eval collect` writes one `CollectedItem` for each account.
2. The label agent writes one `LabelRecord` for each account.
3. `join` matches collected items to label records by account ID. It drops `unsure` labels and items with no label record. It reports both counts. It returns an error when two label records share an account ID.
4. `join` copies the `is_llm_text` flag of the tweet that matched the query into the `EvalItem`. The flag stays empty when no flag names that tweet.
5. `manifest` hashes the eval set file with SHA-256 and counts the items for each label.
6. `split::assign` puts each account ID into the tune split (30%) or the test split (70%). It hashes the seed and the account ID with FNV-1a and a 64-bit finalizer. The same seed and ID give the same split on every platform.
7. `metrics::evaluate` takes a slice of bot probabilities with labels and a bot threshold. It returns the metrics and the ten reliability bins.
8. `metrics::best_threshold` returns the threshold that maximises F1 on a slice. You run it on the tune split. A tie goes to the lowest threshold.
9. `report::render` turns experiment summaries into Markdown: one comparison table, and one calibration curve for each experiment as a Mermaid `xychart-beta`.

### Metric definitions

| Metric | Definition |
|--------|------------|
| Predicted bot | `p_bot >= threshold`. |
| Actual bot | Label `bot` or `agent`. |
| Accuracy, precision, recall, F1 | Standard, from the counts at the threshold. A zero denominator gives 0. |
| ROC-AUC | The share of (bot, human) pairs where the bot has the higher `p_bot`. A tie counts half. Empty when the slice holds one class. |
| Brier | The mean of `(p_bot - outcome)²`. Outcome is 1 for a bot and 0 for a human. |
| Reliability bins | Ten bins `[0, 0.1)` to `[0.9, 1.0]`. Each holds the count, the mean `p_bot`, and the share of bots. |
| ECE | The sum over bins of `count / total × |mean p_bot − share of bots|`. |
| Recall (bot), recall (agent) | Recall over the items with that label alone. Empty when the slice holds none. |

## Formats

### Eval set (JSON Lines, one `EvalItem` for each account)

| Field | Type | Meaning |
|-------|------|---------|
| `account` | `types::Account` | The account. |
| `tweet` | `types::Tweet` | The tweet that matched the query. |
| `recent_tweets` | list of `types::Tweet` | The recent tweets of the account. |
| `label` | `human`, `bot`, `agent` | The label. |
| `is_llm_text` | bool or null | The LLM text flag of `tweet`. |
| `confidence` | `high`, `medium`, `low` | The label agent's confidence. |

### Label record (JSON Lines, one for each account)

```json
{"account_id": "1", "handle": "x", "label": "unsure", "confidence": "low",
 "tweets": [{"tweet_id": "10", "is_llm_text": false}], "reason": "one line"}
```

`label` takes `human`, `bot`, `agent`, or `unsure`.

### Manifest

```bash
cargo run -p eval -- manifest crates/eval/fixtures/eval-set.jsonl
```

```json
{
  "sha256": "…",
  "items": 10,
  "labels": { "human": 4, "bot": 4, "agent": 2 }
}
```

The repository stores this JSON at `crates/eval/manifest.json` for the real eval set. The real eval set lives in a private S3 bucket.

## Collect accounts

You run `eval collect` yourself. It needs `TWITTERAPI_KEY`, and it spends money.

```bash
cargo run -p eval -- collect --queries queries.txt --out collected.jsonl \
  --recent 20 --pages 1 --skip eval-set.jsonl
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--queries` | none | One query for each line. Lines that start with `#` are comments. |
| `--out` | none | The output file, in JSON Lines. A run adds to it and skips the accounts it holds. |
| `--recent` | 20 | Recent tweets to fetch for each account. |
| `--pages` | 1 | Search pages for each query. A page holds up to 20 tweets. |
| `--skip` | none | An eval set file or collected file whose accounts to skip. Repeat for more files. |
| `--yes` | off | Skip the `Continue? [y/N]` prompt. |

| Variable | Default | Meaning |
|----------|---------|---------|
| `TWITTERAPI_KEY` | none | The twitterapi.io key. |
| `TWITTERAPI_INTERVAL_SECS` | 5 | The time from the start of one request to the start of the next. |
| `TWITTERAPI_BASE_URL` | `https://api.twitterapi.io` | The twitterapi.io address. |

How it works:

1. `collect` prints the expected cost and asks `Continue? [y/N]`. It makes no call before you answer `y`.
2. For each query, `collect` calls `TwitterApi::scrape`. It keeps the first tweet of each new account.
3. `collect` skips accounts in a `--skip` file, in the `--out` file, or seen earlier in the run.
4. For each new account, `collect` calls `recent_tweets`. It writes one `CollectedItem` line and flushes the file. A stopped run keeps what it wrote.
5. A failure for one query or one account prints a warning. `collect` moves on.
6. `collect` prints the counts and the estimated cost of what it fetched.

Cost: twitterapi.io charges `USD_PER_1000_TWEETS` (0.15 USD per 1,000 tweets) and at least `USD_PER_CALL_MINIMUM` (0.00015 USD) for each call. The upper bound is queries × pages × 20 search tweets, plus that many accounts × `--recent` recent tweets. Example: 10 queries, 1 page, 20 recent tweets give 200 search tweets and at most 4,000 recent tweets, about 0.63 USD.

## Fixture

`fixtures/eval-set.jsonl` holds ten invented items: 4 human, 4 bot, 2 agent. Two tweets carry `is_llm_text: true`. The handles and texts are made up. They are not X data.

## Subcommands

| Subcommand | State |
|------------|-------|
| `manifest <eval set file>` | Works. |
| `collect` | Works. See [Collect accounts](#collect-accounts). |
| `experiment`, `report` | Print `not built yet` and exit 1. |

## Test it

From the repository root:

```bash
cargo test -p eval
```

The tests need no network access and no Docker.

## Failure modes

| Error | Cause | What you do |
|-------|-------|-------------|
| `Io` | The file does not exist or you cannot read it. | Fix the path. Download the eval set with `aws s3 cp`. |
| `Line` | A line is not valid JSON, or misses a field. The error names the line number. | Fix that line. |
| `MissingSetting` | `TWITTERAPI_KEY` is not set. | Set it, then run again. |
| `Scraper` | twitterapi.io failed on a call that `collect` cannot skip. | Read the message. Run again: `collect` skips collected accounts. |
| `NoQueries` | The queries file holds no query. | Add a query. |
| `DuplicateLabelRecord` | Two label records share an account ID. | Remove one record. |
