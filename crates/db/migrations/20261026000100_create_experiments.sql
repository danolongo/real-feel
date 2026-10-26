CREATE TABLE experiments (
    id UUID PRIMARY KEY,
    model_version TEXT NOT NULL,
    metadata_version TEXT NOT NULL,
    question_version TEXT NOT NULL,
    eval_set_sha256 TEXT NOT NULL,
    seed BIGINT NOT NULL,
    repeats INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE experiment_results (
    experiment_id UUID NOT NULL REFERENCES experiments (id),
    account_id TEXT NOT NULL,
    tweet_id TEXT NOT NULL,
    repeat INTEGER NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('human', 'bot', 'agent')),
    split TEXT NOT NULL CHECK (split IN ('tune', 'test')),
    is_llm_text BOOLEAN,
    p_bot DOUBLE PRECISION NOT NULL,
    p_llm_text DOUBLE PRECISION,
    raw JSONB NOT NULL,
    PRIMARY KEY (experiment_id, account_id, repeat)
);
