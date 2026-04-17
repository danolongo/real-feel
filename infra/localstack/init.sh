#!/bin/bash
awslocal dynamodb create-table \
  --table-name queries \
  --attribute-definitions AttributeName=query_id,AttributeType=S \
  --key-schema AttributeName=query_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
