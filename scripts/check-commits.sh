#!/bin/bash
# Usage: scripts/check-commits.sh <git-range>
# Set PR_TITLE to check a pull request title too.
# Enforces the commit format in docs/writing-rules.md.
set -euo pipefail

range=${1:?usage: check-commits.sh <git-range>}
subject_pattern='^(add|fix|refactor|docs|test|infra|chore)\([a-z0-9_-]+\): [a-z0-9`].*[^.]$'
max_description=71
failed=0

fail() {
  echo "error: $1" >&2
  failed=1
}

check_subject() {
  local label=$1 subject=$2
  if [[ ! $subject =~ $subject_pattern ]]; then
    fail "$label: '$subject' must match '<type>(<regarding>): <description>', in lower case, with no full stop"
    return
  fi
  local description=${subject#*): }
  if ((${#description} > max_description)); then
    fail "$label: the description is ${#description} characters. The limit is $max_description"
  fi
}

if [[ -n ${PR_TITLE:-} ]]; then
  check_subject "PR title" "$PR_TITLE"
fi

for sha in $(git rev-list --no-merges "$range"); do
  label=$(git log -1 --format='commit %h' "$sha")
  check_subject "$label" "$(git log -1 --format=%s "$sha")"

  body=$(git log -1 --format=%b "$sha")
  if grep -qiE '^co-authored-by:' <<<"$body"; then
    fail "$label: remove the Co-authored-by line"
  fi
  paragraphs=$(awk 'BEGIN{n=0; in_p=0} /^[[:space:]]*$/ {in_p=0; next} !in_p {n++; in_p=1} END{print n}' <<<"$body")
  if ((paragraphs > 1)); then
    fail "$label: the body has $paragraphs paragraphs. Write one"
  fi
done

exit $failed
