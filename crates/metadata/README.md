# metadata

`metadata` builds the model input for one tweet from a metadata version and a question version.

## Where it sits

The classifier calls `metadata::build` for each tweet in a classifier job. `metadata` depends on `types`, `serde_json`, `chrono`, and `thiserror`.

## Terms

See [glossary.md](../../docs/glossary.md): tweet, account, metadata, metadata version, question version, model input, classifier, agent account.

## How it works

1. The classifier calls `build` with a tweet, its account, the recent tweets of the account, a metadata version, and a question version. The worker passes an empty slice of recent tweets, because production does not fetch them yet.
2. `build` lays out `state` as the metadata version defines. `state` is a JSON string (`v1`) or a JSON object (`v2` and later).
3. `build` sets `questions` to the questions that the question version defines. Each question has an `id` and its `instructions`. `questions` always holds the question `bot`. It can also hold the question `llm_text`.
4. `build` returns a model input. It copies the tweet ID and both versions into the model input.
5. `build` is pure. The same tweet, account, and versions always give the same model input.
6. An unknown metadata version or question version returns an `Error`.

Every metadata version works with every question version.

### Rules for `v2` and later

- `state` is one JSON object. Its keys are plain English snake_case words, because the keys are part of the prompt Clef-flash reads.
- `build` never reads the clock. The account age is the time between `account.joined_at` and `tweet.posted_at`, in days.
- A missing value becomes JSON `null`. It never becomes `0`.
- Derived numbers carry 2 decimals.
- An empty string counts as a missing value.

### Metadata version `v1`

| Field | Value |
|-------|-------|
| `state` | The tweet text as a JSON string. Nothing else. |

`v1` reads no account field, no recent tweet, and no other tweet field.

### Metadata version `v2`

`v2` adds the account fields.

| Key | Source | Meaning |
|-----|--------|---------|
| `tweet_text` | `tweet.text` | The tweet text. |
| `account.age_days` | `tweet.posted_at` minus `account.joined_at` | Account age at the time of the tweet, in days. |
| `account.followers` | `account.followers` | Follower count. |
| `account.following` | `account.following` | Following count. |
| `account.tweet_count` | `account.tweet_count` | Tweet count. |
| `account.favourites_count` | `account.raw.favouritesCount` | The number of tweets the account liked. |
| `account.media_count` | `account.raw.mediaCount` | The number of media items the account posted. |
| `account.blue_check` | `account.is_verified` | The blue check. Anyone can buy it, so it flags a bot only slightly. |
| `account.x_automated_label` | `account.is_automated` | X's automated-account label. |

### Metadata version `v3`

`v3` is `v2` plus `account.rates`. A rate is `null` when its divisor is missing or not above 0.

| Key | Meaning |
|-----|---------|
| `account.rates.tweets_per_day` | `tweet_count` divided by `age_days`. |
| `account.rates.favourites_per_day` | `favourites_count` divided by `age_days`. |
| `account.rates.followers_per_following` | `followers` divided by `following`. |

### Metadata version `v4`

`v4` is `v3` plus `recent_tweets`. It summarizes the recent tweets of the account. With no recent tweets, `count` is `0` and the other keys are `null`.

| Key | Meaning |
|-----|---------|
| `recent_tweets.count` | The number of recent tweets. |
| `recent_tweets.posting_hours_utc` | The hour of day (0 to 23, UTC) of each recent tweet that has a posting time. |
| `recent_tweets.median_minutes_between` | The median gap in minutes between the posting times, sorted by time. `null` with fewer than 2 posting times. |
| `recent_tweets.duplicate_text_share` | The share of recent tweets whose text equals the text of another recent tweet. |

Two texts are equal when they match after `build` lowercases them, collapses whitespace, and drops every word that starts with `http://` or `https://`.

### Metadata version `v5`

`v5` is `v4` plus four profile keys.

| Key | Source | Meaning |
|-----|--------|---------|
| `account.name` | `account.raw.name` | The display name. |
| `account.bio` | `account.raw.profile_bio.description` | The bio. twitterapi.io leaves the top-level `description` empty. |
| `account.website` | `account.raw.profile_bio.entities.url.urls[0].expanded_url` | The website in the profile. |
| `account.location` | `account.raw.location` | The location in the profile. |

### Metadata version `v6`

`v6` is `v3` plus the four profile keys of `v5`. It holds no `recent_tweets`. The search result carries every key of `v6`, so `v6` needs no call to twitterapi.io beyond the search.

### Question version `v1`

| Question ID | Instructions |
|-------------|--------------|
| `bot` | `Is the account behind this tweet a bot?` |

### Question version `v2`

| Question ID | Instructions |
|-------------|--------------|
| `bot` | `Is this account run by software rather than by a person? Software includes scripts, schedulers, and LLM agents.` |

### Question version `v3`

| Question ID | Instructions |
|-------------|--------------|
| `bot` | `Is this account a bot? Count both scripted bots and AI agents: accounts that an LLM runs and that write their own fluent, varied tweets. A person who sometimes posts AI-written text is not a bot.` |

### Question version `v4`

`v4` holds the `bot` question of `v3` and a second question.

| Question ID | Instructions |
|-------------|--------------|
| `bot` | The wording of `v3`. |
| `llm_text` | `Did an LLM write the text of this tweet, whoever posted it?` |

## Test it

From the repository root:

```bash
cargo test -p metadata
```

The tests need no network access and no Docker.

## Failure modes

| Error | Cause | What you do |
|-------|-------|-------------|
| `UnknownMetadataVersion` | The metadata version is not `v1` to `v6`. | Add the version to `build`, or fix the caller. |
| `UnknownQuestionVersion` | The question version is not `v1` to `v4`. | Add the version to `build`, or fix the caller. |
