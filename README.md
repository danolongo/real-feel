# Real Feel

Real Feel finds bots on X. A user sends a query. Real Feel finds matching tweets, and Clef classifies the account behind each tweet as bot or human. The API returns the result.

Real Feel is mid-refactor into an all-Rust system on AWS. [docs/refactor-plan.md](docs/refactor-plan.md) holds the plan and the milestones. M0 finished. The API serves `GET /health`, and CI guards `main`.

## Start

You need `rustup` and Docker. From the repository root:

```bash
cp .env.example .env        # then set POSTGRES_PASSWORD
docker compose up -d
cargo run -p api
```

[docs/local-dev.md](docs/local-dev.md) lists every step and the checks.

## Contribute

- Follow [docs/writing-rules.md](docs/writing-rules.md) in code comments, commits, and PRs.
- Use the terms in [docs/glossary.md](docs/glossary.md).
- Open one PR for each task.
