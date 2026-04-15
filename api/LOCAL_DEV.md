# Local Lambda Development

The API is a Rust Lambda function tested locally using AWS SAM CLI.

## Prerequisites

- AWS SAM CLI: `brew install aws-sam-cli`
- Zig (for cross-compilation): `brew install zig`
- cargo-zigbuild: `cargo install cargo-zigbuild`
- Java 17 (required by PySpark, not Lambda): `brew install openjdk@17`

## Why cross-compile?

Lambda runs on Linux (ARM64). Your Mac compiles Mach-O binaries. SAM runs the
function inside a Linux Docker container, so the binary must be a Linux ELF.

`cargo-zigbuild` uses Zig as a cross-linker to produce Linux ARM64 binaries from Mac.
It also avoids the C dependency issues that `rdkafka` has with standard cross-compilation toolchains.

## Build & run

```bash
# 1. Cross-compile for Linux ARM64
cd api
cargo zigbuild --release --target aarch64-unknown-linux-gnu

# 2. Copy binary to where SAM expects it (repo root of api/)
cp target/aarch64-unknown-linux-gnu/release/bootstrap bootstrap

# 3. Start local API (runs Lambda inside Docker)
sam local start-api --warm-containers EAGER
```

## Test

```bash
# Submit a query
curl -X POST http://localhost:3000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "show me AI bot tweets"}'

# Get results (replace with real query_id)
curl http://localhost:3000/query/{query_id}/results
```

## Notes

- SAM adds a `/Prod` stage prefix to all paths — the handler strips it via `trim_start_matches("/Prod")`
- LocalStack (DynamoDB): runs at `http://localhost:4566` via Docker Compose
- Logs are JSON-structured via `tracing` — in production these go to CloudWatch
- The `bootstrap` binary in `api/` is gitignored — always rebuild before testing
