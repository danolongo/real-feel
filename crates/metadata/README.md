# metadata

`metadata` builds the model input for one tweet from a metadata version and a question version.

## Where it sits

The classifier calls `metadata::build` for each tweet in a classifier job. `metadata` depends only on `types`.

## Terms

See [glossary.md](../../docs/glossary.md): tweet, account, metadata, metadata version, question version, model input, classifier.

## How it works

1. The classifier calls `build` with a tweet, its account, a metadata version, and a question version.
2. `build` lays out `text` as the metadata version defines.
3. `build` sets `question` to the wording that the question version defines.
4. `build` returns a model input. It copies the tweet ID and both versions into the model input.
5. `build` is pure. The same tweet, account, and versions always give the same model input.
6. An unknown metadata version or question version returns an `Error`.

### Metadata version `v1`

| Field | Value |
|-------|-------|
| `text` | The tweet text. Nothing else. |
| `tweet_id` | `tweet.id`. |
| `metadata_version` | `v1`. |

`v1` reads no account field and no other tweet field.

### Question version `v1`

| Field | Value |
|-------|-------|
| `question` | `Is the account behind this tweet a bot or a human?` |
| `question_version` | `v1`. |

## Test it

From the repository root:

```bash
cargo test -p metadata
```

The tests need no network access and no Docker.

## Failure modes

| Error | Cause | What you do |
|-------|-------|-------------|
| `UnknownMetadataVersion` | The metadata version is not `v1`. | Add the version to `build`, or fix the caller. |
| `UnknownQuestionVersion` | The question version is not `v1`. | Add the version to `build`, or fix the caller. |
