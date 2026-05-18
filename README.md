# real-feel

Agentic Twitter/X analysis platform for bot detection. Users submit natural language queries; a crawler scrapes tweets (no X API), a PySpark pipeline enriches them with bot probability and semantic embeddings, and results are served via a Rust Lambda polling API.

## Stack

| Layer | Tech |
|-------|------|
| API | Rust + AWS Lambda |
| Message broker | Kafka (self-hosted) |
| ML pipeline | PySpark + ONNX + MiniLM |
| Storage | PostgreSQL + pgvector |
| Local infra | Docker Compose + LocalStack |

## Local Dev

Copy `.env.example` to `.env` and fill in values.

```bash
# Terminal 1 — all services
docker compose up

# Terminal 2 — Lambda (from api/)
sam local start-api --docker-network real-feel_default

# Terminal 3 — PySpark pipeline (from api/)
uv run pipeline.py
```

Requires Java 17 for PySpark: `export JAVA_HOME=/opt/homebrew/opt/openjdk@17`

## Status

Early development. POST `/query` → Kafka → PySpark enrichment → PostgreSQL working locally. Crawler and AWS deploy not yet started.
