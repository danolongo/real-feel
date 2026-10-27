# Datadog dashboard

This folder holds the Datadog dashboard, the monitors, and the SLOs for Real Feel. The dashboard shows the request rate, the error rate, the queue depth, and the time from `queued` to `done`.

## Where it sits

The Datadog agent runs on both EC2 instances. The agent receives traces from the API and the workers, and it collects the JSON logs of every container. The Datadog AWS integration reads the SQS, RDS, and EC2 metrics from CloudWatch. The EC2 metrics include `CPUCreditBalance`, which shows when a `t4g` instance runs out of CPU credits. See [the Terraform README](../terraform/README.md) for the setup.

## Terms

See [glossary.md](../../docs/glossary.md) for `queue`, `query status`, `dead-letter queue`, and `trace`.

## How it works

| Widget | Source | Query |
|--------|--------|-------|
| Request rate | Trace metrics of the `api` service | `sum:trace.http.server.request.hits{service:api}` |
| Error rate | Trace metrics of the `api` service | `trace.http.server.request.errors` divided by `trace.http.server.request.hits` |
| Queue depth | Datadog AWS integration | `aws.sqs.approximate_number_of_messages_visible` for the four queues |
| Time from queued to done | Log-based metric `real_feel.queued_to_done_ms` | `p50` and `p95` of the metric |

The classifier worker that sets a query to `done` logs the field `queued_to_done_ms`. The log-based metric turns that field into a distribution.

## Run it

Import the dashboard:

1. Open Datadog and go to **Dashboards** → **New Dashboard**.
2. Choose any layout and create the dashboard.
3. Open the dashboard settings (the gear icon) and choose **Import dashboard JSON**.
4. Paste the contents of `infra/datadog/dashboard.json` and confirm.

Create the log-based metric:

1. Go to **Logs** → **Generate Metrics** → **New Metric**.
2. Choose **Count of logs with a measure** and set the measure to `@queued_to_done_ms`. Choose **Distribution** with percentiles enabled.
3. Set the filter query to `@queued_to_done_ms:*`.
4. Name the metric `real_feel.queued_to_done_ms`.
5. Create the metric.

Before step 2, make `queued_to_done_ms` a measure. Open any worker log line that carries the field, click the field, and choose **Create measure for @queued_to_done_ms**.

## Monitors and SLOs

`monitors.json` holds four monitors. `slos.json` holds three SLOs. `apply.sh` creates them in Datadog, or updates them when they exist.

| Monitor | Alerts when |
|---------|-------------|
| POST /queries p99 latency | p99 passes 500 ms for 5 minutes |
| API error rate | more than 1% of API requests fail in 5 minutes |
| Time from queued to done p95 | p95 passes 2 minutes for 15 minutes |
| Job in a dead-letter queue | a worker logs a line with `dead-letter` |

The DLQ reader deletes each DLQ message within seconds. The DLQ depth therefore stays near 0, so the DLQ monitor reads the worker logs instead.

| SLO | Kind | Target |
|-----|------|--------|
| API availability | metric: requests without an error over all requests | 99% over 7 days |
| POST /queries latency | the p99 latency monitor | 99% over 7 days |
| Time from queued to done | the p95 monitor | 99% over 7 days |

The thresholds are starting values. The M6 load tests set the final values.

Apply them:

1. In Datadog, go to **Organization Settings** → **Application Keys** and create a key.
2. Run the script with the Datadog API key and the application key:

   ```bash
   DD_API_KEY=... DD_APP_KEY=... infra/datadog/apply.sh
   ```

3. Add your notification handle, such as `@your-email`, to each monitor's message in Datadog. The JSON holds no handle.

`DD_SITE` defaults to `us5.datadoghq.com`. The time-from-queued-to-done monitor needs the log-based metric from "Run it".

## Test it

1. Send one query to the API. See the "Send one query" section of [the Terraform README](../terraform/README.md).
2. Wait five minutes. Datadog needs that time to show new log-based metric points and CloudWatch metrics.
3. Check that every widget shows data.

## Failure modes

- **The request widgets stay empty.** The widgets assume that the API's HTTP server spans have the operation name `http.server.request`. Open **Metrics** → **Summary**, search for `trace.`, and check the real name. Then edit the two widget queries to match.
- **The queue widget stays empty.** The Datadog AWS integration is not set up, or CloudWatch has no data for the queue yet. Check the integration with the role ARN from `terraform output datadog_role_arn`.
- **The p99 latency monitor shows no data.** The query assumes the resource tag `resource_name:post_/queries`. Open a `POST /queries` trace and check the real tag, then edit the monitor query.
- **The queued-to-done widget stays empty.** The log-based metric is missing, or the field is not a measure. Redo the steps under "Run it".
