# telemetry

`telemetry` sets up logs and traces for every binary, and moves a trace between processes.

## Where it sits

The API and the workers call `telemetry::init` at startup. `queue` calls `current_traceparent` and `continue_trace`. `telemetry` depends on no crate in the workspace.

## Terms

See [glossary.md](../../docs/glossary.md): API, worker, job, queue, trace, span, trace ID, trace context, Jaeger.

## How it works

1. A binary calls `telemetry::init(service)` once. `service` names the process in Jaeger: `api`, `worker-scraper`, or `worker-classifier`. The binary keeps the returned `Guard` until it exits.
2. `init` installs `tracing` with JSON logs. Each log line carries the fields of its spans, so a worker line names its query ID and trace ID. The `RUST_LOG` variable sets the filter. The default logs Real Feel's crates at `info` and their dependencies at `warn`.
3. `init` checks `OTEL_EXPORTER_OTLP_ENDPOINT`. If you set the variable, `init` adds an OpenTelemetry exporter that sends spans over OTLP/HTTP. If you leave it unset, `init` adds no exporter.
4. `current_traceparent()` returns the current span's context as a W3C `traceparent` string. It returns `None` when no exporter is on, because spans then carry no trace.
5. `continue_trace(&span, traceparent)` makes `span` a child of the span that `traceparent` names. Both spans then share one trace ID.
6. `trace_id(&span)` returns the trace ID of `span`, or `None` without an exporter. The API and the worker record it in the `trace_id` field.
7. `shutdown_signal()` waits for SIGTERM or Ctrl-C. The API and the worker return from `main` after it, so the `Guard` drops.
8. The `Guard` flushes the pending spans when it drops. A process killed with SIGKILL loses its last spans.

The OTLP/HTTP transport uses `reqwest` with rustls, so it needs no OpenSSL. rustls itself uses `aws-lc-rs` as its crypto provider, which builds a C library. No pure Rust provider is ready for production.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `RUST_LOG` | Real Feel's crates at `info`, dependencies at `warn` | The log filter. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | The OTLP collector URL. Unset turns the exporter off. The exporter reads the other `OTEL_EXPORTER_OTLP_*` variables too. |

## Run it

`telemetry` is a library. It has no binary.

## Test it

From the repository root:

```bash
cargo test -p telemetry
```

The tests need no network access and no Docker.

## Failure modes

- `init` returns an error when it cannot build the exporter. The binary should stop.
- `init` panics when a binary calls it twice.
- The exporter drops spans when the collector is down. The binary keeps running.
- `continue_trace` changes nothing when `traceparent` is not valid, or when `init` set up no exporter.
