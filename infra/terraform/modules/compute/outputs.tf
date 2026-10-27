output "autoscaling_group_names" {
  description = "Names of the Auto Scaling groups, keyed by role. The deploy workflow targets instances with tag role=<role>."
  value       = { for role, group in aws_autoscaling_group.group : role => group.name }
}
