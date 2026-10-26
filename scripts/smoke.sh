#!/usr/bin/env bash
# Task 2.11: sends one real query through twitterapi.io and the hosted backend.
# Starts the API and both worker roles, waits for the status `done`, and checks the result.
# Passes when the result holds at least 10 tweets and every tweet has a bot probability.
# Writes every log line of the query, in order, to target/smoke/run.log and prints its Jaeger trace link.
# Needs `docker compose up -d`, the applied migrations, and TWITTERAPI_KEY,
# CLOUDFLARE_ACCOUNT_ID, and CLOUDFLARE_API_TOKEN in .env.
# One run with TWITTERAPI_PAGES=5 costs about $0.02 at twitterapi.io.
# Usage: scripts/smoke.sh "<query text>"
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

if [ $# -ne 1 ]; then
  echo "usage: $0 <query text>" >&2
  exit 1
fi
QUERY=$1
SQS=http://localhost:4566/000000000000
OUT=target/smoke
mkdir -p "$OUT"

export DATABASE_URL=${DATABASE_URL:-postgres://realfeel:${POSTGRES_PASSWORD}@localhost:5433/realfeel}
export AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL:-http://localhost:4566}
export AWS_REGION=${AWS_REGION:-us-east-1}
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-test}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-test}
export SCRAPER_QUEUE_URL=${SCRAPER_QUEUE_URL:-$SQS/scraper-jobs}
export SCRAPER_DLQ_URL=${SCRAPER_DLQ_URL:-$SQS/scraper-jobs-dlq}
export CLASSIFIER_QUEUE_URL=${CLASSIFIER_QUEUE_URL:-$SQS/classifier-jobs}
export CLASSIFIER_DLQ_URL=${CLASSIFIER_DLQ_URL:-$SQS/classifier-jobs-dlq}
export BATCH_SIZE=${BATCH_SIZE:-25}
export JOB_DEADLINE_SECS=${JOB_DEADLINE_SECS:-240}
export DEFAULT_METADATA_VERSION=${DEFAULT_METADATA_VERSION:-v6}
export DEFAULT_QUESTION_VERSION=${DEFAULT_QUESTION_VERSION:-v4}
export BOT_THRESHOLD=${BOT_THRESHOLD:-0.123}
# The smoke test always uses the real services and always scrapes.
export SCRAPER=twitterapi CLASSIFIER=hosted QUERY_REUSE_SECS=0
export DEFAULT_MODEL_VERSION=clef-flash-hosted
export OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4318}
export API_ADDR=127.0.0.1:${SMOKE_PORT:-8090}
API=http://$API_ADDR

cargo build -q -p api -p worker
pids=()
stop() {
  kill "${pids[@]}" 2>/dev/null || true
  # Each process flushes its spans to Jaeger before it exits.
  wait "${pids[@]}" 2>/dev/null || true
}
trap stop EXIT
target/debug/api >"$OUT/api.log" 2>&1 & pids+=($!)
target/debug/worker --role scraper >"$OUT/scraper.log" 2>&1 & pids+=($!)
target/debug/worker --role classifier >"$OUT/classifier.log" 2>&1 & pids+=($!)

for _ in $(seq 30); do
  curl -sf "$API/health" >/dev/null && break
  sleep 1
done

start=$(date +%s)
deadline=$((start + ${SMOKE_TIMEOUT_SECS:-600}))
id=$(jq -n --arg text "$QUERY" '{text: $text}' |
  curl -sSf "$API/queries" -H 'Content-Type: application/json' --data @- | jq -r .query_id)
echo "query $id: $QUERY"

while :; do
  status=$(curl -sSf "$API/queries/$id" | jq -r .status)
  case $status in done | failed) break ;; esac
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "timed out in status $status" >&2
    break
  fi
  sleep 3
done
seconds=$(($(date +%s) - start))

tweets=0 missing=0 after=""
if [ "$status" = done ]; then
  while :; do
    page=$(curl -sSf "$API/queries/$id?limit=200$after")
    tweets=$((tweets + $(jq '.result.tweets | length' <<<"$page")))
    missing=$((missing + $(jq '[.result.tweets[] | select(.p_bot == null)] | length' <<<"$page")))
    cursor=$(jq -r '.result.next_cursor // empty' <<<"$page")
    [ -n "$cursor" ] || break
    after="&after=$cursor"
  done
  jq '.result.tweets[:5][] | {handle: .account.handle, p_bot, text: .text[:80]}' <<<"$page"
fi

batch_count=$(docker compose exec -T postgres \
  psql -U realfeel -d realfeel -tAc "SELECT batch_count FROM queries WHERE id = '$id'")

jq -n --arg query "$QUERY" --arg id "$id" --arg status "$status" \
  --argjson tweets "$tweets" --argjson missing "$missing" \
  --argjson batch_count "${batch_count:-null}" --argjson seconds "$seconds" \
  '{query: $query, query_id: $id, status: $status, tweets: $tweets,
    tweets_without_p_bot: $missing, batch_count: $batch_count, seconds: $seconds}' |
  tee "$OUT/summary.json"

stop
trap - EXIT
for process in api scraper classifier; do
  jq -c --arg process "$process" --arg id "$id" \
    'select(any(.spans[]?; .query_id == $id)) | . + {process: $process}' "$OUT/$process.log"
done | sort >"$OUT/run.log"
trace_id=$(jq -r 'select(.fields.message == "query created") | .span.trace_id // empty' "$OUT/api.log")
echo "Log of the run: $OUT/run.log ($(wc -l <"$OUT/run.log" | tr -d ' ') lines)"
[ -n "$trace_id" ] && echo "Trace: http://localhost:16686/trace/$trace_id"

if [ "$status" = done ] && [ "$tweets" -ge 10 ] && [ "$missing" -eq 0 ]; then
  echo "PASS"
else
  echo "FAIL. Logs: $OUT/" >&2
  curl -sS "$API/queries/$id" | jq '{status, error}' >&2
  exit 1
fi
