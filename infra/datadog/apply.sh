#!/usr/bin/env bash
#
# Creates or updates the monitors in monitors.json and the SLOs in slos.json.
# The script matches each monitor and each SLO by name, so a second run updates
# them instead of adding copies.
#
#   DD_API_KEY=... DD_APP_KEY=... infra/datadog/apply.sh
set -euo pipefail

readonly DIR=$(cd "$(dirname "$0")" && pwd)
readonly API="https://api.${DD_SITE:-us5.datadoghq.com}/api/v1"
: "${DD_API_KEY:?DD_API_KEY is not set}"
: "${DD_APP_KEY:?DD_APP_KEY is not set}"

dd() {
  curl -sSf -X "$1" "$API$2" \
    -H "DD-API-KEY: $DD_API_KEY" \
    -H "DD-APPLICATION-KEY: $DD_APP_KEY" \
    -H 'Content-Type: application/json' \
    ${3:+--data "$3"}
}

uri() {
  jq -rn --arg value "$1" '$value | @uri'
}

monitor_id() {
  dd GET "/monitor?name=$(uri "$1")" | jq -r --arg name "$1" '[.[] | select(.name == $name)][0].id // empty'
}

slo_id() {
  dd GET "/slo?query=$(uri "$1")" | jq -r --arg name "$1" '[.data[] | select(.name == $name)][0].id // empty'
}

apply_monitors() {
  local monitor name id
  jq -c '.[]' "$DIR/monitors.json" | while read -r monitor; do
    name=$(jq -r .name <<<"$monitor")
    id=$(monitor_id "$name")
    if [[ -n "$id" ]]; then
      dd PUT "/monitor/$id" "$monitor" >/dev/null
      printf 'updated monitor %s\n' "$name"
    else
      dd POST /monitor "$monitor" >/dev/null
      printf 'created monitor %s\n' "$name"
    fi
  done
}

# A monitor SLO names its monitors in monitor_names. The script swaps the names
# for the monitor IDs, because the Datadog API takes IDs only.
apply_slos() {
  local slo name id ids monitor_name
  jq -c '.[]' "$DIR/slos.json" | while read -r slo; do
    name=$(jq -r .name <<<"$slo")
    ids=()
    while read -r monitor_name; do
      [[ -n "$monitor_name" ]] || continue
      id=$(monitor_id "$monitor_name")
      [[ -n "$id" ]] || { printf 'monitor %s does not exist\n' "$monitor_name" >&2; exit 1; }
      ids+=("$id")
    done < <(jq -r '.monitor_names[]?' <<<"$slo")
    slo=$(jq -c --argjson ids "$(printf '%s\n' ${ids[@]+"${ids[@]}"} | jq -Rsc 'split("\n")' | jq -c 'map(select(. != "") | tonumber)')" \
      'del(.monitor_names) | if .type == "monitor" then .monitor_ids = $ids else . end' <<<"$slo")
    id=$(slo_id "$name")
    if [[ -n "$id" ]]; then
      dd PUT "/slo/$id" "$slo" >/dev/null
      printf 'updated SLO %s\n' "$name"
    else
      dd POST /slo "$slo" >/dev/null
      printf 'created SLO %s\n' "$name"
    fi
  done
}

apply_monitors
apply_slos
