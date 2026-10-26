# classifier

`classifier` turns model inputs into predictions. It defines `trait Classifier` and two implementations: `Hosted` and `Fake`.

## Where it sits

The classifier worker builds model inputs and calls `classify`. `classifier` calls the hosted backend over HTTPS. It depends on `types`.

## Terms

See [glossary.md](../../docs/glossary.md): classifier, model input, prediction, bot probability, hosted backend, fake, model version, Clef-flash, contract test, nightly workflow.

## How it works

1. The worker calls `classify(&inputs)`.
2. `Hosted` takes the next free request slot. Slots sit `1 / CLASSIFIER_RPS` seconds apart across all concurrent calls.
3. `Hosted` sends one request for each model input. Cloudflare takes one state for each request. The request carries every question of the model input.
4. On `429` or `5xx`, `Hosted` waits and sends the request again. The wait is random between 0 and `500 ms * 2^(n-1)` after failed attempt `n`, capped at 8 seconds. After 5 attempts, `Hosted` returns `Error::Status`. The waits add up to at most 7.5 seconds, far below the 240-second job deadline.
5. Any other `4xx` returns `Error::Status` at once.
6. `Hosted` returns one prediction for each model input, in the same order. Each prediction copies the tweet ID and versions from its model input. `raw` holds Cloudflare's `result` object.

`Fake::new(p_bot, failures)` returns `p_bot` for every model input. `Fake::by_tweet_id(failures)` returns a bot probability from 0 up to, but not including, 1 for each tweet ID. It hashes the tweet ID with FNV-1a, so one tweet ID always gives the same bot probability on every platform and Rust version. Different tweet IDs give different bot probabilities, so metrics on a fixture produce no ties. Both fakes fail their first `failures` calls with `Error::Fake`. Both leave `p_llm_text` empty.

### Hosted backend format

Model ID: `@cf/cloudflare/clef-flash`. Model version: `clef-flash-hosted`.

The request is `POST {base_url}/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/cloudflare/clef-flash` with the header `Authorization: Bearer {CLOUDFLARE_API_TOKEN}`. The base URL is `https://api.cloudflare.com`.

```json
{
  "model": "clef-flash",
  "state": "<model input state>",
  "questions": {
    "bot": { "type": "noul", "instructions": "<instructions of the question bot>" },
    "llm_text": { "type": "noul", "instructions": "<instructions of the question llm_text>" }
  }
}
```

`state` is the model input `state`, sent as given: a JSON string or a JSON object. `questions` holds one entry for each question of the model input, keyed by the question ID. The question `bot` is always there. The question `llm_text` is there only when the question version asks it.

A `noul` question is a yes/no question. The question text must make "yes" mean "bot". Cloudflare's reference: <https://developers.cloudflare.com/workers-ai/models/clef-flash/>.

The response wraps the model output in `result`:

```json
{
  "result": {
    "model": "clef-flash",
    "answers": {
      "bot": { "type": "noul", "noul": 0.87 },
      "llm_text": { "type": "noul", "noul": 0.12 }
    },
    "usage": { "input_tokens": 120, "output_tokens": 0 }
  }
}
```

`answers.bot.noul` is the bot probability, from 0 to 1. `answers.llm_text.noul` is the LLM-text probability. `Hosted` reads it only when the model input asked the question `llm_text`. Otherwise `p_llm_text` stays empty. The `result` wrapper comes from Cloudflare's REST API. A manual run of `scripts/try-clef-flash.sh` on 2026-10-25 confirmed this format, and the contract test checks it every night.

Clef-flash rejects the Workers AI batch API with error 8007: "This model does not support request queuing". One request can carry several tweets in one `state`, with one question for each tweet. In the manual run, that layout moved each bot probability by less than 0.05. M4 can measure whether that layout is accurate enough to cut the request count.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `CLOUDFLARE_ACCOUNT_ID` | none | The Cloudflare account that owns the Workers AI usage. |
| `CLOUDFLARE_API_TOKEN` | none | A token with Workers AI access. |
| `CLASSIFIER_RPS` | `5` | The most requests each second, across all concurrent calls. |

`Hosted::new(base_url, account_id, api_token, rps)` takes the values directly. Tests pass a wiremock URL.

## Run it

`classifier` is a library. It has no binary.

## Test it

From the repository root:

```bash
cargo test -p classifier
```

This command needs no Docker and no network access. wiremock stands in for Cloudflare.

The contract test calls the hosted backend and spends money. The nightly workflow runs it:

```bash
CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... \
cargo test -p classifier --features contract
```

## Failure modes

- Cloudflare answers `429` or `5xx` five times in a row. `classify` returns `Error::Status`. The worker fails the job, and SQS delivers it again.
- Cloudflare answers another `4xx`, such as `401` for a bad token. `classify` returns `Error::Status` without a retry. You fix the variables.
- The response has no `answers.<id>.noul` for a question of the model input, such as `bot`. `classify` returns `Error::Response`.
- One failed model input fails the whole call. The worker retries the job.
- The nightly workflow fails. Cloudflare changed the format, or a secret expired. Read the failed run, then update the README and `Hosted`.
