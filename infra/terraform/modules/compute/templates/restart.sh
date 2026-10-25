#!/usr/bin/env bash
#
# Replaces the containers on this instance with the images for one image tag.
# Terraform installs this file at /opt/real-feel/restart.sh. The deploy
# workflow runs it through SSM Run Command: restart.sh <image tag>.
set -euo pipefail

readonly APP_DIR=/opt/real-feel
readonly ENV_FILE=$APP_DIR/app.env
readonly AGENT_ENV_FILE=$APP_DIR/datadog.env
readonly NETWORK=real-feel
readonly AGENT_NAME=datadog-agent
readonly AGENT_IMAGE=public.ecr.aws/datadog/agent:7
readonly IMDS=http://169.254.169.254/latest

log() {
  printf 'restart: %s\n' "$*"
}

die() {
  printf 'restart: %s\n' "$*" >&2
  exit 1
}

# Prints the value of KEY from an env file.
read_value() {
  sed -n "s/^$1=//p" "$2" | tail -n 1
}

get_param() {
  aws ssm get-parameter --name "$1" --query Parameter.Value --output text
}

# The instance reads its region from the metadata service. The AWS CLI does
# not do this on its own.
imds_region() {
  local token
  token=$(curl -sf -X PUT "$IMDS/api/token" -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
  curl -sf -H "X-aws-ec2-metadata-token: $token" "$IMDS/meta-data/placement/region"
}

# Writes the app env file from /real-feel/ and the database secret. The
# non-secret settings and the secrets come from the same read. mktemp creates
# the file with mode 600, so the secrets never sit in a world-readable file.
write_app_env() {
  local tmp name value db_host db_port db_name db_secret database_url

  tmp=$(mktemp "$APP_DIR/app.env.XXXXXX")
  aws ssm get-parameters-by-path --path /real-feel/ --with-decryption \
    --query 'Parameters[].[Name,Value]' --output text |
    while IFS=$'\t' read -r name value; do
      printf '%s=%s\n' "${name##*/}" "$value"
    done >"$tmp"

  db_host=$(read_value DB_HOST "$tmp")
  db_port=$(read_value DB_PORT "$tmp")
  db_name=$(read_value DB_NAME "$tmp")
  db_secret=$(aws secretsmanager get-secret-value \
    --secret-id "$(read_value DB_SECRET_ARN "$tmp")" \
    --query SecretString --output text)

  # The password can hold characters that break a URL, so it is percent-encoded.
  # The secret goes through stdin, so it never appears in the process list.
  database_url=$(printf '%s' "$db_secret" | python3 -c '
import json
import sys
import urllib.parse

secret = json.load(sys.stdin)
host, port, name = sys.argv[1:4]
user = urllib.parse.quote(secret["username"], safe="")
password = urllib.parse.quote(secret["password"], safe="")
print(f"postgres://{user}:{password}@{host}:{port}/{name}?sslmode=require")
' "$db_host" "$db_port" "$db_name")

  {
    printf 'DATABASE_URL=%s\n' "$database_url"
    printf '%s\n' \
      'OTEL_EXPORTER_OTLP_ENDPOINT=http://datadog-agent:4318' \
      'SCRAPER=twitterapi' \
      'CLASSIFIER=hosted'
  } >>"$tmp"

  mv -f "$tmp" "$ENV_FILE"
}

# Writes the env file for the Datadog agent. It holds the API key, so it has
# mode 600 as well.
write_agent_env() {
  local tmp

  tmp=$(mktemp "$APP_DIR/datadog.env.XXXXXX")
  {
    printf 'DD_API_KEY=%s\n' "$(read_value DD_API_KEY "$ENV_FILE")"
    printf 'DD_SITE=%s\n' "$(read_value DD_SITE "$ENV_FILE")"
    printf '%s\n' \
      'DD_OTLP_CONFIG_RECEIVER_PROTOCOLS_HTTP_ENDPOINT=0.0.0.0:4318' \
      'DD_LOGS_ENABLED=true' \
      'DD_LOGS_CONFIG_CONTAINER_COLLECT_ALL=true' \
      'DD_PROC_ROOT=/host/proc'
  } >"$tmp"

  mv -f "$tmp" "$AGENT_ENV_FILE"
}

# Stops and removes a container when it exists. docker stop uses the
# container's --stop-timeout, so the process gets time to flush.
remove_container() {
  if docker container inspect "$1" >/dev/null 2>&1; then
    docker stop "$1" >/dev/null
    docker rm "$1" >/dev/null
  fi
}

# Starts the Datadog agent, or keeps it when its config has not changed. The
# config hash goes into a label, so a changed env file or image restarts it.
ensure_agent() {
  local config_hash current_hash running

  config_hash=$({ cat "$AGENT_ENV_FILE"; printf '%s\n' "$AGENT_IMAGE"; } | sha256sum | cut -d ' ' -f 1)
  current_hash=$(docker inspect --format '{{ index .Config.Labels "real-feel.config-hash" }}' "$AGENT_NAME" 2>/dev/null || true)
  running=$(docker inspect --format '{{ .State.Running }}' "$AGENT_NAME" 2>/dev/null || true)

  if [[ "$current_hash" == "$config_hash" && "$running" == true ]]; then
    log "$AGENT_NAME already runs with the current config"
    return 0
  fi

  docker pull "$AGENT_IMAGE" >/dev/null
  remove_container "$AGENT_NAME"
  docker run -d --name "$AGENT_NAME" \
    --network "$NETWORK" \
    --restart unless-stopped \
    --label "real-feel.config-hash=$config_hash" \
    --env-file "$AGENT_ENV_FILE" \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    -v /proc/:/host/proc/:ro \
    -v /sys/fs/cgroup/:/host/sys/fs/cgroup:ro \
    -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
    "$AGENT_IMAGE" >/dev/null
  log "$AGENT_NAME started"
}

start_api() {
  local image=$1

  remove_container api
  docker run -d --name api \
    --env-file "$ENV_FILE" \
    --network "$NETWORK" \
    -p 80:8080 \
    -e API_ADDR=0.0.0.0:8080 \
    --stop-timeout 15 \
    --restart unless-stopped \
    "$image" >/dev/null
  log "api started from $image"
}

start_worker() {
  local image=$1 kind=$2

  remove_container "worker-$kind"
  docker run -d --name "worker-$kind" \
    --env-file "$ENV_FILE" \
    --network "$NETWORK" \
    --stop-timeout 15 \
    --restart unless-stopped \
    "$image" --role "$kind" >/dev/null
  log "worker-$kind started from $image"
}

# Polls the API until /ready answers 200. The API can take a while to run its
# migrations, so this waits up to 90 seconds.
wait_for_ready() {
  local attempt status running

  for ((attempt = 1; attempt <= 90; attempt++)); do
    status=$(curl -s --max-time 2 -o /dev/null -w '%{http_code}' http://localhost/ready || true)
    if [[ "$status" == 200 ]]; then
      log "api answers 200 on /ready after $attempt s"
      return 0
    fi

    running=$(docker inspect --format '{{ .State.Running }}' api)
    if [[ "$running" != true ]]; then
      docker logs --tail 50 api >&2 || true
      die "api stopped before it answered /ready"
    fi
    sleep 1
  done

  die "api did not answer 200 on /ready within 90 s"
}

main() {
  local tag role repo

  [[ $# -eq 1 ]] || die "usage: restart.sh <image tag>"
  [[ $1 =~ ^[A-Za-z0-9_.-]+$ ]] || die "invalid image tag: $1"
  tag=$1

  [[ -f $APP_DIR/role ]] || die "missing $APP_DIR/role"
  role=$(cat "$APP_DIR/role")

  AWS_REGION=$(imds_region)
  export AWS_REGION

  write_app_env
  write_agent_env

  case "$role" in
    api) repo=$(get_param /real-feel/deploy/API_REPOSITORY) ;;
    worker) repo=$(get_param /real-feel/deploy/WORKER_REPOSITORY) ;;
    *) die "unknown role in $APP_DIR/role: $role" ;;
  esac

  aws ecr get-login-password | docker login --username AWS --password-stdin "${repo%%/*}" >/dev/null
  docker pull "$repo:$tag" >/dev/null
  log "pulled $repo:$tag"

  docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null

  ensure_agent

  if [[ "$role" == api ]]; then
    start_api "$repo:$tag"
    wait_for_ready
  else
    start_worker "$repo:$tag" scraper
    start_worker "$repo:$tag" classifier
  fi

  log "done"
}

main "$@"
