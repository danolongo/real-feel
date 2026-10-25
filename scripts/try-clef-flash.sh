#!/usr/bin/env bash
# Sends two model inputs to Clef-flash on Workers AI in three ways and prints each raw response.
# The first input reads like a bot, the second like a human, so "yes = bot" shows in the numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

QUESTION="Is the account behind this tweet a bot?"
URL="https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/cloudflare/clef-flash"
STATES='[
  "Follow me for free crypto giveaways!!! DM now to claim 1000 USDT #airdrop #giveaway #crypto",
  "Finally finished repotting the basil. Half of it died over the winter but the rest looks happy."
]'

run() {
  curl -sS "$URL$1" \
    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
    -H 'Content-Type: application/json' \
    --data @- | jq .
}

echo "== 1. One request for each input"
jq -c '.[]' <<<"$STATES" | while read -r state; do
  jq -n --argjson state "$state" --arg q "$QUESTION" \
    '{model: "clef-flash", state: $state, questions: {bot: {type: "noul", instructions: $q}}}' | run ""
done

echo "== 2. Both inputs in one batch request (Workers AI batch API)"
queued=$(jq --arg q "$QUESTION" \
  '{requests: map({model: "clef-flash", state: ., questions: {bot: {type: "noul", instructions: $q}}})}' \
  <<<"$STATES" | run "?queueRequest=true")
echo "$queued"
request_id=$(jq -r '.result.request_id // empty' <<<"$queued")
if [ -n "$request_id" ]; then
  for _ in $(seq 1 12); do
    sleep 5
    polled=$(jq -n --arg id "$request_id" '{request_id: $id}' | run "")
    status=$(jq -r '.result.status // "done"' <<<"$polled")
    case $status in queued | running) echo "still $status" ;; *) break ;; esac
  done
  echo "$polled"
fi

echo "== 3. Both inputs in one state, one question for each tweet"
jq '{
  model: "clef-flash",
  state: (to_entries | map({tweet: .key, text: .value})),
  questions: (to_entries | map({
    key: "bot_\(.key)",
    value: {type: "noul", instructions: "Is the account behind tweet \(.key) a bot?"}
  }) | from_entries)
}' <<<"$STATES" | run ""
