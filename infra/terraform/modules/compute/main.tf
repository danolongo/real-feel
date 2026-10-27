data "aws_region" "current" {}

data "aws_caller_identity" "current" {}

data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
}

locals {
  region     = data.aws_region.current.region
  account_id = data.aws_caller_identity.current.account_id

  groups = var.groups

  worker_groups = {
    scraper = {
      queue_name           = var.scraper_queue_name
      backlog_per_instance = var.scraper_backlog_per_instance
    }
    classifier = {
      queue_name           = var.classifier_queue_name
      backlog_per_instance = var.classifier_backlog_per_instance
    }
  }

  instance_profiles = {
    api        = aws_iam_instance_profile.api.name
    scraper    = aws_iam_instance_profile.worker.name
    classifier = aws_iam_instance_profile.worker.name
  }

  security_groups = {
    api        = var.api_sg_id
    scraper    = var.worker_sg_id
    classifier = var.worker_sg_id
  }

  ssm_parameter_arns = [
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter/real-feel",
    "arn:aws:ssm:${local.region}:${local.account_id}:parameter/real-feel/*",
  ]
}

# Parameters

resource "aws_ssm_parameter" "config" {
  for_each = var.config

  name  = "/real-feel/${each.key}"
  type  = "String"
  value = each.value
}

resource "aws_ssm_parameter" "deploy" {
  for_each = {
    API_REPOSITORY    = var.api_repository_url
    WORKER_REPOSITORY = var.worker_repository_url
  }

  name  = "/real-feel/deploy/${each.key}"
  type  = "String"
  value = each.value
}

# deploy.yml writes the image tag here before it restarts the instances. A new
# instance reads it on first boot. The restart script reads /real-feel/ without
# recursion, so this parameter stays out of the app env.
resource "aws_ssm_parameter" "image_tag" {
  name  = "/real-feel/deploy/IMAGE_TAG"
  type  = "String"
  value = "none"

  lifecycle {
    ignore_changes = [value]
  }
}

# Terraform creates each secret with a placeholder. You set the real value
# once with aws ssm put-parameter --overwrite, and ignore_changes keeps that
# value out of the state file.
resource "aws_ssm_parameter" "secret" {
  for_each = toset([
    "TWITTERAPI_KEY",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_API_TOKEN",
    "DD_API_KEY",
  ])

  name  = "/real-feel/${each.key}"
  type  = "SecureString"
  value = "placeholder"

  lifecycle {
    ignore_changes = [value]
  }
}

# Instance roles

data "aws_iam_policy_document" "ec2_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "instance_common" {
  statement {
    sid = "EcrPull"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = var.repository_arns
  }

  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "ReadConfigParameters"
    actions = [
      "ssm:GetParametersByPath",
      "ssm:GetParameter",
    ]
    resources = local.ssm_parameter_arns
  }

  statement {
    sid       = "DecryptConfigParameters"
    actions   = ["kms:Decrypt"]
    resources = ["*"]

    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${local.region}.amazonaws.com"]
    }
  }

  statement {
    sid       = "ReadDatabaseSecret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.db_secret_arn]
  }
}

# The API only enqueues jobs, and its /ready endpoint reads the queue
# attributes. It never receives or deletes a message.
data "aws_iam_policy_document" "api_queue" {
  statement {
    sid = "EnqueueJobs"
    actions = [
      "sqs:SendMessage",
      "sqs:GetQueueAttributes",
    ]
    resources = var.queue_arns
  }
}

data "aws_iam_policy_document" "worker_queue" {
  statement {
    sid = "ProcessJobs"
    actions = [
      "sqs:SendMessage",
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
      "sqs:ChangeMessageVisibility",
    ]
    resources = var.queue_arns
  }
}

resource "aws_iam_role" "api" {
  name               = "real-feel-api"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_role_policy_attachment" "api_ssm" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "api_common" {
  name   = "common"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.instance_common.json
}

resource "aws_iam_role_policy" "api_queue" {
  name   = "queue"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api_queue.json
}

resource "aws_iam_instance_profile" "api" {
  name = "real-feel-api"
  role = aws_iam_role.api.name
}

resource "aws_iam_role" "worker" {
  name               = "real-feel-worker"
  assume_role_policy = data.aws_iam_policy_document.ec2_trust.json
}

resource "aws_iam_role_policy_attachment" "worker_ssm" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "worker_common" {
  name   = "common"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.instance_common.json
}

resource "aws_iam_role_policy" "worker_queue" {
  name   = "queue"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.worker_queue.json
}

resource "aws_iam_instance_profile" "worker" {
  name = "real-feel-worker"
  role = aws_iam_role.worker.name
}

# Instances

resource "aws_launch_template" "group" {
  for_each = local.groups

  name          = "real-feel-${each.key}"
  image_id      = data.aws_ssm_parameter.al2023_ami.value
  instance_type = each.value.instance_type

  # The restart script ships inside user data. A change to user data adds a
  # launch template version, and the next instance that starts runs it.
  user_data = base64encode(templatefile("${path.module}/templates/user-data.sh.tftpl", {
    role           = each.key
    region         = local.region
    restart_script = file("${path.module}/templates/restart.sh")
  }))

  iam_instance_profile {
    name = local.instance_profiles[each.key]
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [local.security_groups[each.key]]
    delete_on_termination       = true
  }

  # Containers on the Docker bridge reach the metadata service through one
  # extra hop, so the hop limit is 2.
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  # Standard credits throttle a t-class instance at its baseline instead of
  # billing extra CPU. Other classes have no credits.
  dynamic "credit_specification" {
    for_each = can(regex("^t[0-9]", each.value.instance_type)) ? [1] : []

    content {
      cpu_credits = "standard"
    }
  }

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_type           = "gp3"
      volume_size           = 20
      encrypted             = true
      delete_on_termination = true
    }
  }

  # A new AMI would add a launch template version on every plan. Change the AMI
  # on purpose, with terraform apply -replace.
  lifecycle {
    ignore_changes = [image_id]
  }
}

resource "aws_autoscaling_group" "group" {
  for_each = local.groups

  name                = "real-feel-${each.key}"
  min_size            = each.value.min_size
  max_size            = each.value.max_size
  desired_capacity    = each.value.min_size
  vpc_zone_identifier = var.public_subnet_ids

  target_group_arns         = each.key == "api" ? [var.target_group_arn] : []
  health_check_type         = each.key == "api" ? "ELB" : "EC2"
  health_check_grace_period = 300

  metrics_granularity = "1Minute"
  enabled_metrics = [
    "GroupMinSize",
    "GroupMaxSize",
    "GroupDesiredCapacity",
    "GroupInServiceInstances",
    "GroupPendingInstances",
    "GroupStandbyInstances",
    "GroupTerminatingInstances",
    "GroupTotalInstances",
  ]

  launch_template {
    id      = aws_launch_template.group[each.key].id
    version = aws_launch_template.group[each.key].latest_version
  }

  tag {
    key                 = "Name"
    value               = "real-feel-${each.key}"
    propagate_at_launch = true
  }

  tag {
    key                 = "role"
    value               = each.key
    propagate_at_launch = true
  }

  # The scaling policies own the desired capacity after the first apply.
  lifecycle {
    ignore_changes = [desired_capacity]
  }

  depends_on = [
    aws_ssm_parameter.config,
    aws_ssm_parameter.deploy,
    aws_ssm_parameter.image_tag,
    aws_ssm_parameter.secret,
    aws_iam_role_policy.api_common,
    aws_iam_role_policy.api_queue,
    aws_iam_role_policy.worker_common,
    aws_iam_role_policy.worker_queue,
  ]
}

# Scaling

resource "aws_autoscaling_policy" "api_requests" {
  name                      = "real-feel-api-requests"
  autoscaling_group_name    = aws_autoscaling_group.group["api"].name
  policy_type               = "TargetTrackingScaling"
  estimated_instance_warmup = 300

  target_tracking_configuration {
    target_value = var.api_requests_per_target

    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${var.alb_arn_suffix}/${var.target_group_arn_suffix}"
    }
  }
}

# Target tracking only scales out here. disable_scale_in stops it from removing
# several instances at once. The scale-in policy below removes one instance
# and then waits for its cooldown.
resource "aws_autoscaling_policy" "worker_backlog" {
  for_each = local.worker_groups

  name                      = "real-feel-${each.key}-backlog"
  autoscaling_group_name    = aws_autoscaling_group.group[each.key].name
  policy_type               = "TargetTrackingScaling"
  estimated_instance_warmup = 300

  target_tracking_configuration {
    target_value     = each.value.backlog_per_instance
    disable_scale_in = true

    customized_metric_specification {
      metrics {
        id          = "backlog"
        label       = "Backlog per instance"
        expression  = "visible / instances"
        return_data = true
      }

      metrics {
        id          = "visible"
        return_data = false

        metric_stat {
          stat = "Average"

          metric {
            namespace   = "AWS/SQS"
            metric_name = "ApproximateNumberOfMessagesVisible"

            dimensions {
              name  = "QueueName"
              value = each.value.queue_name
            }
          }
        }
      }

      metrics {
        id          = "instances"
        return_data = false

        metric_stat {
          stat = "Average"

          metric {
            namespace   = "AWS/AutoScaling"
            metric_name = "GroupInServiceInstances"

            dimensions {
              name  = "AutoScalingGroupName"
              value = aws_autoscaling_group.group[each.key].name
            }
          }
        }
      }
    }
  }
}

resource "aws_autoscaling_policy" "worker_scale_in" {
  for_each = local.worker_groups

  name                   = "real-feel-${each.key}-scale-in"
  autoscaling_group_name = aws_autoscaling_group.group[each.key].name
  policy_type            = "SimpleScaling"
  adjustment_type        = "ChangeInCapacity"
  scaling_adjustment     = -1
  cooldown               = 300
}

# The alarm fires after 15 minutes of backlog below 90 % of the target. An idle
# queue sends no data, so missing data counts as a breach.
resource "aws_cloudwatch_metric_alarm" "worker_scale_in" {
  for_each = local.worker_groups

  alarm_name          = "real-feel-${each.key}-backlog-low"
  comparison_operator = "LessThanThreshold"
  threshold           = each.value.backlog_per_instance * 0.9
  evaluation_periods  = 15
  datapoints_to_alarm = 15
  treat_missing_data  = "breaching"
  alarm_actions       = [aws_autoscaling_policy.worker_scale_in[each.key].arn]

  metric_query {
    id          = "backlog"
    expression  = "visible / instances"
    label       = "Backlog per instance"
    return_data = true
  }

  metric_query {
    id = "visible"

    metric {
      namespace   = "AWS/SQS"
      metric_name = "ApproximateNumberOfMessagesVisible"
      period      = 60
      stat        = "Average"

      dimensions = {
        QueueName = each.value.queue_name
      }
    }
  }

  metric_query {
    id = "instances"

    metric {
      namespace   = "AWS/AutoScaling"
      metric_name = "GroupInServiceInstances"
      period      = 60
      stat        = "Average"

      dimensions = {
        AutoScalingGroupName = aws_autoscaling_group.group[each.key].name
      }
    }
  }
}
