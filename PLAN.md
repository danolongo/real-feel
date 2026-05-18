# Real Feel — Project Plan

## Context
Build an agentic Twitter/X analysis platform focused on bot detection. Users submit natural language queries, a crawler collects tweets (no X API — scraping only), a trained bot detection model classifies them, and results are served with semantic search and temporal divergence analysis. Think exa.ai but for bot intelligence on X.

The bot detection model (rf.v1.0.0) is already trained — RoBERTa-based, binary classification, saved as `.pt`.

⚠️ **Model quality issue (rf.v1.0.0):** Model is heavily biased toward predicting HUMAN (~99% confidence) on all test inputs. Root cause: training data imbalance (82% human / 18% bot) and likely insufficient generalization. ONNX export and inference pipeline are confirmed correct — PyTorch and ONNX outputs match exactly. Issue is the model itself.
- TODO: Retrain with better-balanced data and/or different dataset sources
- TODO: Consider architecture changes (e.g. use pretrained RoBERTa backbone via HuggingFace instead of custom transformer from scratch)

## Architecture Overview

```
User → AWS API Gateway → Rust Lambda
         ↓                    ↓ (poll)     ↓ (write)
    [immediate response]  DynamoDB    Kafka: "queries"
                            ↑              ↓
                            └─ EC2 t4g.micro (self-hosted Kafka)
                                 ↓
                    Rust Crawler Service (Docker)
                    [GraphQL-first, headless fallback]
                                 ↓
                           Kafka: "raw-tweets"
                                 ↓
                    EMR Spot Cluster (on-demand)
                    PySpark job (launch when needed)
                    ├─ Bot detection (ONNX)
                    ├─ Embedding (MiniLM)
                    └─ Temporal tagging
                                 ↓
                    Kafka: "enriched-tweets"
                           ↓        ↓
                     Supabase    S3 (optional archive)
                   PostgreSQL
                    + pgvector
```

**Flow:**
1. User submits query → Lambda writes to DynamoDB (`status: processing`) + publishes to Kafka
2. PySpark job (on-demand) consumes, enriches, writes to Supabase
3. User polls → Lambda checks DynamoDB, fetches results from Supabase when ready

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| API Gateway | AWS API Gateway | Public entry point; logs in CloudWatch |
| API Backend | Rust + Lambda | Submits queries to Kafka, polls DynamoDB/Supabase for results |
| Query State | DynamoDB | Tracks query status (processing/complete); enables polling pattern |
| Message Broker | Kafka (self-hosted on EC2) | t4g.micro instance; $5-7/month instead of MSK |
| Crawler | Rust (async reqwest + serde) | GraphQL scraping primary, Playwright sidecar fallback |
| ML Pipeline | PySpark (EMR Spot on-demand) | Launch clusters only when running queries; $5-15/month |
| Bot Model | ONNX (exported from .pt) | Run in PySpark UDFs via onnxruntime |
| Embedding Model | all-MiniLM-L6-v2 | Encodes tweet text → vector (384 dims) for semantic similarity search |
| Search & Storage | Supabase PostgreSQL + pgvector | Managed PostgreSQL; pgvector for k-NN semantic search; free tier (~500MB) |
| Optional Archive | S3 (Parquet via Spark) | Long-term storage; optional for learning |
| Local Dev | Docker Compose | Kafka + Spark + PostgreSQL (with pgvector) + LocalStack (DynamoDB) |

## Component Breakdown

### 1. Rust API (Lambda)
**Query submission & polling:**
- POST `/query` — accepts natural language query
  1. Generate `query_id`
  2. Publish to Kafka "queries" topic
  3. Create DynamoDB entry: `{ query_id, status: "processing", created_at, ... }`
  4. Return `{ query_id, status: "processing" }` immediately
- GET `/query/{id}/results` — polls for results
  1. Check DynamoDB: query complete?
  2. If yes, fetch results from Supabase, return with `status: "complete"`
  3. If no, return `{ status: "processing", eta: "X minutes" }`

**Results & analysis:**
- GET `/query/{id}/timeline` — temporal divergence data (bot vs human sentiment over time window)
- GET `/search` — semantic search across stored tweets (k-NN on embeddings in Supabase via pgvector)

### 2. Crawler Service (Rust, Docker) — SEPARATE REPO
**How it works:**
- Kafka consumer polls "queries" topic for new search requests
- Hits X's GraphQL API endpoint (`TweetSearchTimeline`) — reverse-engineered, no official API
- Maintains pool of rotating guest tokens (X issues tokens via `/guest/activate.json`)
- If GraphQL fails → auto-failover to headless Playwright browser sidecar
- Collects N tweets per query, rate-limits to avoid detection (~1-5 sec between requests)
- Produces raw tweets to Kafka "raw-tweets" topic: `{ tweet_id, text, author, timestamp, reply_count, ... }`

**Example flow:**
```
Kafka "queries": { query_id: "q123", search: "AI bots", count: 500 }
  → Crawler fetches 500 tweets from X
  → Kafka "raw-tweets": { tweet_id: "1234", text: "...", author: "@bob", timestamp: ... } × 500
```

### 3. PySpark Pipeline (EMR Spot, On-Demand)
**Launched only when needed:**
- Triggered by Kafka "raw-tweets" messages
- Per each tweet received, runs 3 enrichment stages in parallel:
  1. **Bot detection** — tokenize text with HuggingFace, run ONNX model, output `is_bot`, `bot_probability`
  2. **Embedding** — encode tweet text with MiniLM model → 384-dimensional vector for semantic search
  3. **Temporal tagging** — add time bucket, track for rolling averages per (query, is_bot) pair
- Writes enriched tweet to Kafka "enriched-tweets" topic: `{ tweet_id, text, is_bot, bot_probability, embedding_vector, ... }`
- Sinks to Supabase PostgreSQL (for search via pgvector) and optionally S3 Parquet (archive)

**Example enrichment:**
```
Input:  { tweet_id: "1234", text: "AI is great", author: "@bob" }
Output: { tweet_id: "1234", text: "AI is great", author: "@bob", 
          is_bot: false, bot_probability: 0.15,
          embedding_vector: [0.42, -0.15, 0.88, ...] }
```

### 4. Supabase PostgreSQL + pgvector
**Semantic Search & Storage:**
- Table: `tweets` — stores enriched tweets with embedding vector (384 dims)
  - pgvector IVFFLAT index for fast k-NN search
  - User query "what do people think about AI?" → embed query → pgvector k-NN returns closest vectors
  - Benefit: catches synonyms, related concepts, not just keyword matches
- Table: `temporal` — pre-aggregated time-series data
  - Stores: `{ query, timestamp, human_count, bot_count }`
  - Enables: temporal divergence graphs (bot vs human activity over time)
- **Free tier:** 500MB database (enough for ~250K enriched tweets for learning)

### 5. LLM Query Layer (Phase 2)
- Start with direct keyword pass-through (Phase 1)
- Later: LLM decomposes complex queries into multiple crawler searches
- Cross-references results across sub-queries
- Identifies patterns (coordinated timing, shared URLs, sentiment manipulation)

## Project Structure

```
real-feel/                  # This repo — main platform
├── api/                    # Rust Lambda API
│   ├── Cargo.toml
│   └── src/
├── pipeline/               # PySpark streaming pipeline
│   ├── pyproject.toml
│   ├── bot_detection/
│   ├── sentiment/
│   └── embeddings/
├── model/                  # Model artifacts
│   ├── rf.v1.0.0.pt
│   └── rf.v1.0.0.onnx
├── infra/                  # IaC (CDK or Terraform) — EC2 Kafka, EMR Spark, Supabase
├── docker-compose.yml      # Local dev: Kafka + Spark + PostgreSQL (with pgvector) + LocalStack (DynamoDB)
└── README.md

real-feel-crawler/          # Separate repo — crawler microservice
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── graphql/            # X GraphQL API reverse-engineering
│   ├── headless/           # Playwright fallback
│   ├── tokens/             # Guest token rotation
│   └── kafka/              # Producer logic
├── playwright-sidecar/     # Python Playwright fallback container
├── Dockerfile
└── README.md
```

## Implementation Order

1. ✅ **Local Docker Compose** — Kafka + Spark + PostgreSQL (with pgvector) + LocalStack (DynamoDB)
2. **Supabase setup** — Create project, enable pgvector extension, create tables for tweets & temporal
3. ⏳ **Crawler** (separate repo) — Rust service that scrapes X and produces to Kafka "raw-tweets" — NOT YET STARTED
4. ✅ **PySpark pipeline** — Locally test: consume raw tweets, run all 4 models, write to local PostgreSQL
5. ⏳ **Lambda API** — Locally test with SAM: submit queries to Kafka, poll DynamoDB/PostgreSQL
   - ✅ POST /query — parses body, generates query_id, writes to DynamoDB (status: processing), returns 202
   - ✅ POST /query — publish to Kafka "queries" topic
   - ⏳ GET /query/{id}/results — poll DynamoDB, fetch results from PostgreSQL
6. **End-to-end testing** — Full pipeline working locally: query → crawler → Spark → PostgreSQL → results
7. **Deploy to AWS** — EC2 Kafka broker + EMR Spot Spark cluster + Lambda + Supabase
8. **Semantic search** — k-NN queries via pgvector working end-to-end
9. **Temporal analysis** — Time-series sentiment divergence
10. **LLM query decomposition** — Agentic layer (Phase 2)

## Verification Checklist

**Local Development:**
- Docker Compose `up` brings Kafka, Spark, PostgreSQL + pgvector, and LocalStack (DynamoDB) online
- Crawler fetches tweets and produces to Kafka "raw-tweets" topic
- PySpark consumes raw tweets, runs all 4 models (bot detection, sentiment, embedding, temporal tagging)
- PySpark writes enriched tweets to local PostgreSQL `tweets` table
- Lambda (local SAM): POST `/query` creates DynamoDB entry + publishes to Kafka "queries"
- Lambda: GET `/query/{id}/results` polls DynamoDB, returns `status: processing` or `status: complete` with results from PostgreSQL

**AWS Deployment:**
- EC2 t4g.micro Kafka broker running and accepting messages
- EMR Spot cluster launches on-demand when Spark jobs submitted
- Supabase PostgreSQL + pgvector accessible via Lambda
- Full end-to-end: User query → Lambda → Kafka → Crawler → Spark → Supabase → results returned
- GET `/search` returns semantically similar tweets via pgvector k-NN
- GET `/query/{id}/timeline` shows bot vs human sentiment divergence over time
