CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    key_sha256 BYTEA NOT NULL UNIQUE,
    query_quota_per_hour INTEGER NOT NULL CHECK (query_quota_per_hour >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ
);

ALTER TABLE queries ADD COLUMN api_key_id UUID REFERENCES api_keys (id);
CREATE INDEX queries_api_key_id_created_at ON queries (api_key_id, created_at);
