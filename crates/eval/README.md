# eval

`eval` builds eval sets, runs experiments, and writes eval reports. This part holds the pure core: the eval set format, the manifest, the splits, the metrics, and the report renderer.

## Where it sits

`eval` is a library and a binary. It depends only on `types`. It does not call Postgres, the classifier, `metadata`, or the scraper yet. A later task wires `eval collect` and `eval experiment` to them.

## Terms

See [glossary.md](../../docs/glossary.md): eval set, label, agent account, recent tweets, LLM text, label agent, manifest, tune split, test split, reliability bin, calibration curve, experiment, eval report.

## How it works

1. `eval collect` writes one `CollectedItem` for each account. (Not built yet.)
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

## Fixture

`fixtures/eval-set.jsonl` holds ten invented items: 4 human, 4 bot, 2 agent. Two tweets carry `is_llm_text: true`. The handles and texts are made up. They are not X data.

## Subcommands

| Subcommand | State |
|------------|-------|
| `manifest <eval set file>` | Works. |
| `collect`, `experiment`, `report` | Print `not built yet` and exit 1. |

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
| `DuplicateLabelRecord` | Two label records share an account ID. | Remove one record. |
