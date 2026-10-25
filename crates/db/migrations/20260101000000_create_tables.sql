CREATE TYPE query_status AS ENUM ('queued', 'scraping', 'classifying', 'done', 'failed');

CREATE TABLE queries (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    status query_status NOT NULL DEFAULT 'queued',
    error TEXT,
    batch_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE accounts (
    id TEXT PRIMARY KEY,
    handle TEXT NOT NULL,
    raw JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE tweets (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES accounts (id),
    text TEXT NOT NULL,
    posted_at TIMESTAMPTZ,
    raw JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE query_tweets (
    query_id UUID NOT NULL REFERENCES queries (id),
    tweet_id TEXT NOT NULL REFERENCES tweets (id),
    PRIMARY KEY (query_id, tweet_id)
);

CREATE TABLE query_batches (
    query_id UUID NOT NULL REFERENCES queries (id),
    batch_number INTEGER NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (query_id, batch_number)
);

CREATE TABLE predictions (
    tweet_id TEXT NOT NULL REFERENCES tweets (id),
    model_version TEXT NOT NULL,
    metadata_version TEXT NOT NULL,
    question_version TEXT NOT NULL,
    p_bot DOUBLE PRECISION NOT NULL,
    raw JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tweet_id, model_version, metadata_version, question_version)
);
