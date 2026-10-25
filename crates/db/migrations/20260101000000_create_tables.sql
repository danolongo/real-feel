create type query_status as enum ('queued', 'scraping', 'classifying', 'done', 'failed');

create table queries (
    id uuid primary key,
    text text not null,
    status query_status not null default 'queued',
    error text,
    batch_count integer,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table accounts (
    id text primary key,
    handle text not null,
    raw jsonb not null,
    updated_at timestamptz not null default now()
);

create table tweets (
    id text primary key,
    account_id text not null references accounts (id),
    text text not null,
    posted_at timestamptz,
    raw jsonb not null,
    updated_at timestamptz not null default now()
);

create table query_tweets (
    query_id uuid not null references queries (id),
    tweet_id text not null references tweets (id),
    primary key (query_id, tweet_id)
);

create table query_batches (
    query_id uuid not null references queries (id),
    batch_number integer not null,
    finished_at timestamptz not null default now(),
    primary key (query_id, batch_number)
);

create table predictions (
    tweet_id text not null references tweets (id),
    model_version text not null,
    metadata_version text not null,
    question_version text not null,
    p_bot double precision not null,
    raw jsonb not null,
    created_at timestamptz not null default now(),
    primary key (tweet_id, model_version, metadata_version, question_version)
);
