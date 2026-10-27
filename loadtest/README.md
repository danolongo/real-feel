# loadtest

`loadtest` holds the k6 scenarios and the tools that run a load test of the API and the workers.

## Where it sits

k6 sends queries to the API through the ALB. The API and the workers write to Postgres. `preload.sql` and `check-queries.sh` talk to Postgres through an SSM port forward, because RDS stays private.

## Terms

See [glossary.md](../docs/glossary.md): query, query reuse, query status, API key, query quota, tweet, account, prediction, fake.

- **break point:** the first moment a run crosses a threshold: `POST /queries` p99 above 500 ms, error rate above 1%, or one unfinished query after 300 s.
- **system-only mode:** the synthetic scraper and the fake run, so the test measures Real Feel's own code.
- **full mode:** twitterapi.io and the hosted backend run.
- **query ID file:** the file that holds one line for each query k6 sent, with its final query status.

## How it works

1. Each k6 iteration sends `POST /queries` with the `Authorization: Bearer` header and the body `{"text": "..."}`. The API answers `202` with `{"query_id": "..."}`. It answers `202` for a reused query too, so k6 cannot tell a reused query from a new one.
2. The iteration polls `GET /queries/{id}` every 2 s until the query status is `done` or `failed`, or 300 s pass. The polls count as load.
3. The iteration writes `qid=<id> status=<done|failed|unfinished>` with `console.log`. `--console-output` sends those lines to the query ID file.
4. A new text is `loadtest <run id> <number>`. A repeat takes the text of an earlier new query. `REUSE_SHARE` sets the share of repeats. The report computes the query reuse rate from the texts k6 sent. The API's own `QUERY_REUSE_SECS` (default 3600) decides whether Postgres reuses the query.
5. At the end, k6 prints a JSON report and writes it to `$OUT.summary.json`. The report holds the break point, the elapsed time and the arrival rate when the run stopped, both request rates, both p99 latencies, and the query reuse rate.
6. `ramp` stops at its first break point (`abortOnFail`). The thresholds use whole-run numbers, so the run stops a little after the true break point. Read the arrival rate at the stop as an upper bound.

## Setup

1. Install k6:

   ```bash
   brew install k6
   ```

2. Apply the load test sizes from `infra/terraform/main`. [`loadtest.tfvars`](../infra/terraform/main/loadtest.tfvars) sets the instance types, the group sizes, and the RDS class. It also turns on system-only mode, turns the query quota off, and switches the WAF rate rule to count. Pass the same `-var` flags as a normal apply:

   ```bash
   terraform apply -var-file=loadtest.tfvars \
     -var domain_name=api.danielemartinez.com \
     -var datadog_external_id=<external ID>
   ```

   For full mode, edit `loadtest.tfvars` as its header says, then apply again.

3. Run `deploy.yml`. The restart script reads the new SSM settings on each restart:

   ```bash
   gh workflow run deploy.yml --ref <branch>
   ```

4. Create the API key. Open a session on an API instance, then run `api-key` in the container. [The API README](../crates/api/README.md) shows the same steps.

   ```bash
   aws ssm start-session --target <api instance id>
   sudo docker exec api api-key create --name loadtest --quota 1000000
   ```

   The instance ID comes from `aws ec2 describe-instances --filters Name=tag:role,Values=api Name=instance-state-name,Values=running --query 'Reservations[].Instances[].InstanceId' --output text`. Copy the key when `api-key` prints it. `api-key` prints the key once.

## Run it

Run every command from the repository root. Never point a scenario at production until you set the stack up as above.

```bash
export API_KEY=<key from api-key create>
export OUT=query-ids.txt
k6 run --console-output "$OUT" loadtest/k6/smoke.js
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `BASE_URL` | `https://api.danielemartinez.com` | The API address. |
| `API_KEY` | none | The API key. Required. |
| `REUSE_SHARE` | `0.2` | The share of queries that repeat an earlier text, from 0 to 1. |
| `OUT` | `query-ids.txt` | The name of the query ID file. Also names `$OUT.summary.json`. Pass the same path to `--console-output`. |
| `RUN_ID` | the start time | Makes the query texts unique. Change it for each run. |
| `MAX_VUS` | rate × 310 | The most virtual users k6 starts. Each in-flight query holds one for up to 300 s. |
| `PRE_ALLOCATED_VUS` | rate × 30 | The virtual users k6 starts first. |

Rates are whole queries per second. Each scenario sets these variables too:

| Scenario | Variables (defaults) | Shape |
|----------|----------------------|-------|
| `smoke` | `RATE=1`, `DURATION_SECS=60` | One flat rate. |
| `ramp` | `START_RATE=1`, `MAX_RATE=100`, `STEPS=10`, `STEP_SECS=60` | The rate climbs in `STEPS` equal steps until a break point. |
| `burst` | `BASE_RATE=2`, `BURST_RATE=50`, `BASE_SECS=60`, `BURST_SECS=60`, `RECOVERY_SECS=120` | A jump to `BURST_RATE` in 1 s, then back. |
| `soak` | `RATE=5`, `DURATION_SECS=3600` | One flat rate for one hour. |

Check the scripts without a stack: `API_KEY=dummy k6 inspect loadtest/k6/ramp.js`.

### Modes

- **System only:** run all four scenarios.
- **Full:** run `smoke` and one low fixed rate (`RATE=1 DURATION_SECS=600`), with one scraper instance. The scraper paces twitterapi.io calls in each process, so a second scraper instance doubles the call rate on one key. A full-mode `ramp` finds the free-tier rate limit, not a break point.

### Where k6 runs

k6 runs on your laptop first. Move it to a `c7g` instance in `us-east-1` when k6 reports `dropped_iterations` above 0 or k6's CPU passes 80%. The laptop then limits the test. The report names the host.

## Reach RDS with psql

1. Read the RDS endpoint and the password:

   ```bash
   DB_HOST=$(aws ssm get-parameter --name /real-feel/DB_HOST --query Parameter.Value --output text)
   aws secretsmanager get-secret-value --secret-id <db_secret_arn> --query SecretString --output text
   ```

   `terraform output` does not print `db_secret_arn`. Read it with `aws rds describe-db-instances --query 'DBInstances[].MasterUserSecret.SecretArn'`. The secret holds `username` and `password`.

2. Forward local port 5434 to RDS through an API instance. Port 5433 belongs to the local Postgres of Docker Compose. Keep this terminal open:

   ```bash
   aws ssm start-session --target <api instance id> \
     --document-name AWS-StartPortForwardingSessionToRemoteHost \
     --parameters "{\"host\":[\"$DB_HOST\"],\"portNumber\":[\"5432\"],\"localPortNumber\":[\"5434\"]}"
   ```

3. In a second terminal, connect:

   ```bash
   export DATABASE_URL="postgres://<username>:<password>@localhost:5434/realfeel?sslmode=require"
   psql "$DATABASE_URL"
   ```

## Fill Postgres with the preload

Run `preload.sql` before the first `ramp` of each session. The stack starts empty after each destroy. Every preload row has a fixed ID, so a second run changes nothing.

```bash
psql "$DATABASE_URL" -f loadtest/preload.sql
```

| psql variable | Default | Meaning |
|---------------|---------|---------|
| `preload_accounts` | `20000` | Accounts. IDs start with `preload-account-`. |
| `preload_tweets` | `100000` | Tweets and predictions. IDs start with `preload-tweet-`. |
| `preload_queries` | `5000` | Queries with the status `done`. Texts start with `preload query`. |
| `preload_tweets_per_query` | `50` | Rows in `query_tweets` for each query. |
| `preload_batches_per_query` | `2` | Rows in `query_batches` for each query. |
| `preload_model_version` | `clef-flash-hosted` | The model version of each preload prediction. |
| `preload_metadata_version` | `v1` | The metadata version of each preload prediction. |
| `preload_question_version` | `v1` | The question version of each preload prediction. |

Set a variable with `-v preload_tweets=500000`. The synthetic scraper's IDs never start with `preload-`, so the preload rows never collide with them.

## Check for lost queries

After a run, `check-queries.sh` counts the query status of every ID in the query ID file. A query that is not `done` or `failed` counts as lost. The script prints those IDs and exits with 1. It exits with 2 when the file holds no ID.

```bash
loadtest/check-queries.sh "$OUT"
```

Run it after the stack drains. A query that k6 marks `unfinished` can still finish later, and `check-queries.sh` then reports it as `done`.

## Record a flame graph

1. Deploy the `profiling` images from your branch:

   ```bash
   gh workflow run deploy.yml --ref <branch> -f profile=profiling
   ```

2. Open a session on the instance that runs the process you profile (`api`, `scraper`, or `classifier`) and install `perf`:

   ```bash
   aws ssm start-session --target <instance id>
   sudo dnf install -y perf
   ```

3. Find the host process ID of the container's process. `docker top` prints it. Start a `ramp` first, so the process has work:

   ```bash
   sudo docker top api
   sudo perf record -F 99 -g -p <pid> -- sleep 60
   sudo perf script > out.perf
   ```

4. Copy `out.perf` to your laptop. Install `inferno` on the laptop, not on the instance, and fold the output:

   ```bash
   cargo install inferno
   inferno-collapse-perf < out.perf | inferno-flamegraph > flame.svg
   ```

Delete `out.perf` from the instance when you finish.

## Failure modes

- **k6 prints `dropped_iterations`.** k6 ran out of virtual users or CPU. Raise `MAX_VUS`, or move k6 to a `c7g` instance.
- **Every `POST /queries` returns `401`.** `API_KEY` is wrong or revoked.
- **Every `POST /queries` returns `429`.** The query quota is on. Set `QUERY_QUOTA_ENABLED=false`, or create the key with a larger `--quota`.
- **The WAF returns `403` or `429` above a rate.** Set `waf_rate_limit_enabled=false`.
- **`check-queries.sh` cannot connect.** The port forward closed. Start it again.
