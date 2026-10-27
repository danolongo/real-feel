output "scraper_queue_url" {
  description = "URL of the scraper-jobs queue."
  value       = aws_sqs_queue.scraper.url
}

output "scraper_dlq_url" {
  description = "URL of the scraper-jobs-dlq queue."
  value       = aws_sqs_queue.scraper_dlq.url
}

output "classifier_queue_url" {
  description = "URL of the classifier-jobs queue."
  value       = aws_sqs_queue.classifier.url
}

output "classifier_dlq_url" {
  description = "URL of the classifier-jobs-dlq queue."
  value       = aws_sqs_queue.classifier_dlq.url
}

output "queue_arns" {
  description = "ARNs of all four queues."
  value = [
    aws_sqs_queue.scraper.arn,
    aws_sqs_queue.scraper_dlq.arn,
    aws_sqs_queue.classifier.arn,
    aws_sqs_queue.classifier_dlq.arn,
  ]
}

output "scraper_queue_name" {
  description = "Name of the scraper-jobs queue. The CloudWatch metrics use it."
  value       = aws_sqs_queue.scraper.name
}

output "classifier_queue_name" {
  description = "Name of the classifier-jobs queue. The CloudWatch metrics use it."
  value       = aws_sqs_queue.classifier.name
}
