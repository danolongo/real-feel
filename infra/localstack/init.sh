#!/bin/bash
set -euo pipefail

REGION=us-east-1
ACCOUNT_ID=000000000000
MAX_RECEIVE_COUNT=5

create_queue_pair() {
  local name=$1
  local dlq_arn="arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${name}-dlq"

  awslocal sqs create-queue --queue-name "${name}-dlq" --region "$REGION"
  awslocal sqs create-queue --queue-name "$name" --region "$REGION" \
    --attributes "{\"RedrivePolicy\":\"{\\\"deadLetterTargetArn\\\":\\\"${dlq_arn}\\\",\\\"maxReceiveCount\\\":\\\"${MAX_RECEIVE_COUNT}\\\"}\"}"
}

create_queue_pair scraper-jobs
create_queue_pair classifier-jobs
