#!/usr/bin/env bash
# Counts the status of every query ID in a k6 query ID file and exits 1 when any query is lost.
# Usage: DATABASE_URL=postgres://... loadtest/check-queries.sh [query-ids.txt]
set -euo pipefail

file="${1:-${OUT:-query-ids.txt}}"
: "${DATABASE_URL:?set DATABASE_URL}"

ids="$(grep -Eo 'qid=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "$file" | cut -d= -f2 | sort -u || true)"
if [ -z "$ids" ]; then
  echo "no query IDs in $file" >&2
  exit 2
fi

report="$(
  {
    echo 'CREATE TEMP TABLE ids (id UUID PRIMARY KEY);'
    echo '\copy ids FROM STDIN'
    echo "$ids"
    echo '\.'
    echo "SELECT 'count', COALESCE(q.status::TEXT, 'missing'), COUNT(*) FROM ids LEFT JOIN queries q ON q.id = ids.id GROUP BY 2 ORDER BY 2;"
    echo "SELECT 'lost', ids.id, COALESCE(q.status::TEXT, 'missing') FROM ids LEFT JOIN queries q ON q.id = ids.id WHERE q.status IS NULL OR q.status NOT IN ('done', 'failed') ORDER BY 2;"
  } | psql "$DATABASE_URL" -X -q -t -A -F ' ' -v ON_ERROR_STOP=1
)"

echo "query IDs in $file: $(echo "$ids" | wc -l | tr -d ' ')"
echo "status counts:"
echo "$report" | awk '$1 == "count" { printf "  %-12s %s\n", $2, $3 }'

lost="$(echo "$report" | awk '$1 == "lost" { print "  " $2 " " $3 }')"
if [ -n "$lost" ]; then
  echo "lost queries (not done or failed):"
  echo "$lost"
  exit 1
fi
echo "no lost queries"
