ALTER TABLE accounts
    ADD COLUMN joined_at TIMESTAMPTZ,
    ADD COLUMN followers INTEGER,
    ADD COLUMN following INTEGER,
    ADD COLUMN tweet_count INTEGER,
    ADD COLUMN is_verified BOOLEAN,
    ADD COLUMN is_automated BOOLEAN;
