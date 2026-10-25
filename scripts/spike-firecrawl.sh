#!/usr/bin/env bash
# Spike 2.1: asks self-hosted Firecrawl for an X search page, a profile page, and a tweet page,
# then runs the scraper's exact search request. Each response lands in target/spike-firecrawl/.
# Usage: scripts/spike-firecrawl.sh <query> <handle> <tweet-url>
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 3 ]; then
  echo "usage: $0 <query> <handle> <tweet-url>" >&2
  exit 1
fi
QUERY=$1 HANDLE=$2 TWEET_URL=$3
FIRECRAWL_URL=${FIRECRAWL_URL:-http://localhost:3002}
OUT=target/spike-firecrawl
mkdir -p "$OUT"

SCHEMA='{"type":"object","properties":{"accounts":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"handle":{"type":"string"}},"required":["id","handle"]}},"tweets":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"account_id":{"type":"string"},"text":{"type":"string"},"posted_at":{"type":"string"}},"required":["id","account_id","text"]}}},"required":["accounts","tweets"]}'
PROMPT="Extract every tweet on the page and the account that posted each tweet."

post() {
  local name=$1 path=$2 body=$3
  echo "== $name"
  curl -sS --max-time 120 -X POST "$FIRECRAWL_URL$path" \
    -H 'Content-Type: application/json' --data "$body" | jq . > "$OUT/$name.json"
  jq '{success, error, markdown_chars: (.data.markdown // "" | length), json: .data.json, status: .data.metadata.statusCode}' "$OUT/$name.json"
}

scrape() {
  local name=$1 url=$2
  post "$name" /v2/scrape "$(jq -n --arg url "$url" --arg p "$PROMPT" --argjson s "$SCHEMA" \
    '{url: $url, timeout: 60000, formats: ["markdown", {type: "json", prompt: $p, schema: $s}]}')"
}

scrape search "https://x.com/search?q=$(jq -rn --arg q "$QUERY" '$q|@uri')&f=live"
scrape profile "https://x.com/$HANDLE"
scrape tweet "$TWEET_URL"
post scraper-search /v2/search "$(jq -n --arg q "site:x.com $QUERY" --arg p "$PROMPT" --argjson s "$SCHEMA" \
  '{query: $q, scrapeOptions: {formats: [{type: "json", prompt: $p, schema: $s}]}}')"

echo "Full responses: $OUT/"
