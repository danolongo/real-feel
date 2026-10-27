-- Fills Postgres with fixed rows before a load test.
-- Run:  psql "$DATABASE_URL" -f loadtest/seed.sql
-- Override a count:  psql "$DATABASE_URL" -v seed_tweets=500000 -f loadtest/seed.sql
-- Every row has a fixed ID that starts with `seed-` (or derives from `seed-query-N`), and every
-- INSERT skips existing keys. A second run with the same counts changes nothing. A larger count
-- adds the missing rows. A smaller count removes nothing.

\set ON_ERROR_STOP on

\if :{?seed_accounts}
\else
\set seed_accounts 20000
\endif
\if :{?seed_tweets}
\else
\set seed_tweets 100000
\endif
\if :{?seed_queries}
\else
\set seed_queries 5000
\endif
\if :{?seed_tweets_per_query}
\else
\set seed_tweets_per_query 50
\endif
\if :{?seed_batches_per_query}
\else
\set seed_batches_per_query 2
\endif
\if :{?seed_model_version}
\else
\set seed_model_version clef-flash-hosted
\endif
\if :{?seed_metadata_version}
\else
\set seed_metadata_version v1
\endif
\if :{?seed_question_version}
\else
\set seed_question_version v1
\endif

BEGIN;

INSERT INTO accounts (id, handle, raw, joined_at, followers, following, tweet_count, is_verified, is_automated)
SELECT
    'seed-account-' || n,
    'seed_user_' || n,
    '{}'::JSONB,
    NOW() - (n % 3650 || ' days')::INTERVAL,
    (n * 37) % 100000,
    (n * 11) % 5000,
    (n * 53) % 200000,
    n % 20 = 0,
    n % 5 = 0
FROM GENERATE_SERIES(1, :seed_accounts) AS n
ON CONFLICT (id) DO NOTHING;

INSERT INTO tweets (id, account_id, text, posted_at, raw)
SELECT
    'seed-tweet-' || n,
    'seed-account-' || ((n - 1) % :seed_accounts + 1),
    'seed tweet ' || n || ' about nothing in particular',
    NOW() - (n % 86400 || ' minutes')::INTERVAL,
    '{}'::JSONB
FROM GENERATE_SERIES(1, :seed_tweets) AS n
ON CONFLICT (id) DO NOTHING;

INSERT INTO queries (id, text, status, batch_count, created_at, updated_at)
SELECT
    MD5('seed-query-' || n)::UUID,
    'seed query ' || n,
    'done',
    :seed_batches_per_query,
    NOW() - (n || ' seconds')::INTERVAL,
    NOW() - (n || ' seconds')::INTERVAL
FROM GENERATE_SERIES(1, :seed_queries) AS n
ON CONFLICT (id) DO NOTHING;

INSERT INTO query_tweets (query_id, tweet_id)
SELECT
    MD5('seed-query-' || q)::UUID,
    'seed-tweet-' || (((q - 1) * :seed_tweets_per_query + k) % :seed_tweets + 1)
FROM GENERATE_SERIES(1, :seed_queries) AS q
CROSS JOIN GENERATE_SERIES(0, :seed_tweets_per_query - 1) AS k
ON CONFLICT (query_id, tweet_id) DO NOTHING;

INSERT INTO query_batches (query_id, batch_number)
SELECT
    MD5('seed-query-' || q)::UUID,
    b
FROM GENERATE_SERIES(1, :seed_queries) AS q
CROSS JOIN GENERATE_SERIES(0, :seed_batches_per_query - 1) AS b
ON CONFLICT (query_id, batch_number) DO NOTHING;

INSERT INTO predictions (tweet_id, model_version, metadata_version, question_version, p_bot, raw, p_llm_text)
SELECT
    'seed-tweet-' || n,
    :'seed_model_version',
    :'seed_metadata_version',
    :'seed_question_version',
    (n % 1000) / 1000.0,
    '{}'::JSONB,
    (n % 997) / 997.0
FROM GENERATE_SERIES(1, :seed_tweets) AS n
ON CONFLICT (tweet_id, model_version, metadata_version, question_version) DO NOTHING;

COMMIT;

ANALYZE accounts, tweets, queries, query_tweets, query_batches, predictions;

SELECT 'accounts' AS table_name, COUNT(*) AS seed_rows FROM accounts WHERE id LIKE 'seed-%'
UNION ALL SELECT 'tweets', COUNT(*) FROM tweets WHERE id LIKE 'seed-%'
UNION ALL SELECT 'queries', COUNT(*) FROM queries WHERE text LIKE 'seed query %'
UNION ALL SELECT 'predictions', COUNT(*) FROM predictions WHERE tweet_id LIKE 'seed-%';
