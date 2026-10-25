output "db_host" {
  description = "Hostname of the Postgres endpoint."
  value       = aws_db_instance.main.address
}

output "db_port" {
  description = "Port of the Postgres endpoint."
  value       = aws_db_instance.main.port
}

output "db_name" {
  description = "Name of the database."
  value       = aws_db_instance.main.db_name
}

output "db_username" {
  description = "Name of the master user."
  value       = aws_db_instance.main.username
}

output "db_secret_arn" {
  description = "ARN of the Secrets Manager secret that holds the master username and password."
  value       = aws_db_instance.main.master_user_secret[0].secret_arn
}
