#!/usr/bin/env bash
# Spike 2.1: asks twitterapi.io for an X search, a profile, and one tweet.
# Prints a summary of each response and saves the full response in target/spike-twitterapi/.
# Usage: scripts/spike-twitterapi.sh <query> <handle> <tweet-id>
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

if [ $# -ne 3 ]; then
  echo "usage: $0 <query> <handle> <tweet-id>" >&2
  exit 1
fi
QUERY=$1 HANDLE=$2 TWEET_ID=$3
URL=https://api.twitterapi.io/twitter
OUT=target/spike-twitterapi
mkdir -p "$OUT"

get() {
  local name=$1 path=$2
  shift 2
  echo "== $name"
  curl -sS --get "$URL$path" -H "X-API-Key: ${TWITTERAPI_KEY}" "$@" | jq . > "$OUT/$name.json"
  jq -e '.error' "$OUT/$name.json" >/dev/null && jq . "$OUT/$name.json"
  # The free tier allows one request every 5 seconds.
  sleep 6
}

get search /tweet/advanced_search --data-urlencode "query=$QUERY lang:en -filter:retweets" --data-urlencode queryType=Latest
jq '{
  status, msg, has_next_page, next_cursor,
  tweets: (.tweets | length),
  first_tweet_fields: (.tweets[0] | keys?),
  first_author_fields: (.tweets[0].author | keys?),
  sample: [.tweets[:3][] | {id, createdAt, text: .text[:80], author: (.author | {id, userName, followers, following, createdAt, statusesCount, isBlueVerified})}]
}' "$OUT/search.json"

get profile /user/info --data-urlencode "userName=$HANDLE"
jq '{status, msg, fields: (.data | keys?), data: (.data | {id, userName, followers, following, createdAt, statusesCount, isAutomated})}' "$OUT/profile.json"

get tweet /tweets --data-urlencode "tweet_ids=$TWEET_ID"
jq '{status, msg, tweets: (.tweets | length), tweet: (.tweets[0] | {id, createdAt, text, author: .author.userName})}' "$OUT/tweet.json"

echo "Full responses: $OUT/"
