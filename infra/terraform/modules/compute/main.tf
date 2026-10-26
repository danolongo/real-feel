data "aws_region" "current" {}

data "aws_caller_identity" "current" {}

data "aws_ssm_parameter" "al2023_ami" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
}

locals {
  region     = data.aws_region.current.region
  account_id = data.aws_caller_identity.current.account_id

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

resource "aws_instance" "api" {
  ami                         = data.aws_ssm_parameter.al2023_ami.value
  instance_type               = var.instance_type
  subnet_id                   = element(var.public_subnet_ids, 0)
  vpc_security_group_ids      = [var.api_sg_id]
  iam_instance_profile        = aws_iam_instance_profile.api.name
  associate_public_ip_address = true

  # The restart script ships inside user data. Replacing the instance on a
  # change to user data installs the new script.
  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/templates/user-data.sh.tftpl", {
    role           = "api"
    restart_script = file("${path.module}/templates/restart.sh")
  })

  # Containers on the Docker bridge reach the metadata service through one
  # extra hop, so the hop limit is 2.
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  # Standard credits throttle the instance at its baseline instead of billing extra CPU.
  credit_specification {
    cpu_credits = "standard"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }

  tags = {
    Name = "real-feel-api"
    role = "api"
  }

  # A new AMI would replace the instance. Change the AMI on purpose, not on
  # every plan.
  lifecycle {
    ignore_changes = [ami]
  }
}

resource "aws_instance" "worker" {
  ami                         = data.aws_ssm_parameter.al2023_ami.value
  instance_type               = var.instance_type
  subnet_id                   = element(var.public_subnet_ids, 1)
  vpc_security_group_ids      = [var.worker_sg_id]
  iam_instance_profile        = aws_iam_instance_profile.worker.name
  associate_public_ip_address = true

  user_data_replace_on_change = true
  user_data = templatefile("${path.module}/templates/user-data.sh.tftpl", {
    role           = "worker"
    restart_script = file("${path.module}/templates/restart.sh")
  })

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  # Standard credits throttle the instance at its baseline instead of billing extra CPU.
  credit_specification {
    cpu_credits = "standard"
  }

  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }

  tags = {
    Name = "real-feel-worker"
    role = "worker"
  }

  lifecycle {
    ignore_changes = [ami]
  }
}
