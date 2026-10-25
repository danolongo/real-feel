locals {
  visibility_timeout_seconds = 300
  max_receive_count          = 5
  dlq_retention_seconds      = 1209600 # 14 days, the SQS maximum
}

# Each job queue sends a job to its DLQ after max_receive_count receives.
resource "aws_sqs_queue" "scraper_dlq" {
  name                      = "scraper-jobs-dlq"
  message_retention_seconds = local.dlq_retention_seconds
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "scraper" {
  name                       = "scraper-jobs"
  visibility_timeout_seconds = local.visibility_timeout_seconds
  sqs_managed_sse_enabled    = true

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.scraper_dlq.arn
    maxReceiveCount     = local.max_receive_count
  })
}

resource "aws_sqs_queue" "classifier_dlq" {
  name                      = "classifier-jobs-dlq"
  message_retention_seconds = local.dlq_retention_seconds
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "classifier" {
  name                       = "classifier-jobs"
  visibility_timeout_seconds = local.visibility_timeout_seconds
  sqs_managed_sse_enabled    = true

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.classifier_dlq.arn
    maxReceiveCount     = local.max_receive_count
  })
}
