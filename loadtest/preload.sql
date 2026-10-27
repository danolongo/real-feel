-- Fills Postgres with fixed rows before a load test.
-- Run:  psql "$DATABASE_URL" -f loadtest/preload.sql
-- Override a count:  psql "$DATABASE_URL" -v preload_tweets=500000 -f loadtest/preload.sql
-- Every row has a fixed ID that starts with `preload-` (or derives from `preload-query-N`), and every
-- INSERT skips existing keys. A second run with the same counts changes nothing. A larger count
-- adds the missing rows. A smaller count removes nothing.

\set ON_ERROR_STOP on

\if :{?preload_accounts}
\else
\set preload_accounts 20000
\endif
\if :{?preload_tweets}
\else
\set preload_tweets 100000
\endif
\if :{?preload_queries}
\else
\set preload_queries 5000
\endif
\if :{?preload_tweets_per_query}
\else
\set preload_tweets_per_query 50
\endif
\if :{?preload_batches_per_query}
\else
\set preload_batches_per_query 2
\endif
\if :{?preload_model_version}
\else
\set preload_model_version clef-flash-hosted
\endif
\if :{?preload_metadata_version}
\else
\set preload_metadata_version v1
\endif
\if :{?preload_question_version}
\else
\set preload_question_version v1
\endif

BEGIN;

INSERT INTO accounts (id, handle, raw, joined_at, followers, following, tweet_count, is_verified, is_automated)
SELECT
    'preload-account-' || n,
    'preload_user_' || n,
    '{}'::JSONB,
    NOW() - (n % 3650 || ' days')::INTERVAL,
    (n * 37) % 100000,
    (n * 11) % 5000,
    (n * 53) % 200000,
    n % 20 = 0,
    n % 5 = 0
FROM GENERATE_SERIES(1, :preload_accounts) AS n
ON CONFLICT (id) DO NOTHING;

INSERT INTO tweets (id, account_id, text, posted_at, raw)
SELECT
    'preload-tweet-' || n,
    'preload-account-' || ((n - 1) % :preload_accounts + 1),
    'preload tweet ' || n || ' about nothing in particular',
    NOW() - (n % 86400 || ' minutes')::INTERVAL,
    '{}'::JSONB
FROM GENERATE_SERIES(1, :preload_tweets) AS n
ON CONFLICT (id) DO NOTHING;

INSERT INTO queries (id, text, status, batch_count, created_at, updated_at)
SELECT
    MD5('preload-query-' || n)::UUID,
    'preload query ' || n,
    'done',
    :preload_batches_per_query,
    NOW() - (n || ' seconds')::INTERVAL,
    NOW() - (n || ' seconds')::INTERVAL
FROM GENERATE_SERIES(1, :preload_queries) AS n
ON CONFLICT (id) DO NOTHING;

INSERT INTO query_tweets (query_id, tweet_id)
SELECT
    MD5('preload-query-' || q)::UUID,
    'preload-tweet-' || (((q - 1) * :preload_tweets_per_query + k) % :preload_tweets + 1)
FROM GENERATE_SERIES(1, :preload_queries) AS q
CROSS JOIN GENERATE_SERIES(0, :preload_tweets_per_query - 1) AS k
ON CONFLICT (query_id, tweet_id) DO NOTHING;

INSERT INTO query_batches (query_id, batch_number)
SELECT
    MD5('preload-query-' || q)::UUID,
    b
FROM GENERATE_SERIES(1, :preload_queries) AS q
CROSS JOIN GENERATE_SERIES(0, :preload_batches_per_query - 1) AS b
ON CONFLICT (query_id, batch_number) DO NOTHING;

INSERT INTO predictions (tweet_id, model_version, metadata_version, question_version, p_bot, raw, p_llm_text)
SELECT
    'preload-tweet-' || n,
    :'preload_model_version',
    :'preload_metadata_version',
    :'preload_question_version',
    (n % 1000) / 1000.0,
    '{}'::JSONB,
    (n % 997) / 997.0
FROM GENERATE_SERIES(1, :preload_tweets) AS n
ON CONFLICT (tweet_id, model_version, metadata_version, question_version) DO NOTHING;

COMMIT;

ANALYZE accounts, tweets, queries, query_tweets, query_batches, predictions;

SELECT 'accounts' AS table_name, COUNT(*) AS preload_rows FROM accounts WHERE id LIKE 'preload-%'
UNION ALL SELECT 'tweets', COUNT(*) FROM tweets WHERE id LIKE 'preload-%'
UNION ALL SELECT 'queries', COUNT(*) FROM queries WHERE text LIKE 'preload query %'
UNION ALL SELECT 'predictions', COUNT(*) FROM predictions WHERE tweet_id LIKE 'preload-%';
