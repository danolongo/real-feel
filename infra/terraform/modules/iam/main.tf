data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.region

  # The repositories live in the registry module. These ARNs use the same
  # names, so this module does not need the registry outputs as inputs.
  ecr_repository_arns = [
    "arn:aws:ecr:${local.region}:${local.account_id}:repository/real-feel/api",
    "arn:aws:ecr:${local.region}:${local.account_id}:repository/real-feel/worker",
  ]
}

# GitHub Actions OIDC

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
  ]
}

data "aws_iam_policy_document" "github_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = [for branch in var.github_deploy_branches : "repo:${var.github_repo}:ref:refs/heads/${branch}"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "real-feel-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_trust.json
}

data "aws_iam_policy_document" "github_deploy" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = local.ecr_repository_arns
  }

  statement {
    sid       = "RunCommandDocument"
    actions   = ["ssm:SendCommand"]
    resources = ["arn:aws:ssm:${local.region}::document/AWS-RunShellScript"]
  }

  statement {
    sid       = "RunCommandProjectInstances"
    actions   = ["ssm:SendCommand"]
    resources = ["arn:aws:ec2:${local.region}:${local.account_id}:instance/*"]

    condition {
      test     = "StringEquals"
      variable = "ssm:resourceTag/project"
      values   = ["real-feel"]
    }
  }

  statement {
    sid       = "WriteImageTag"
    actions   = ["ssm:GetParameter", "ssm:PutParameter"]
    resources = ["arn:aws:ssm:${local.region}:${local.account_id}:parameter/real-feel/deploy/IMAGE_TAG"]
  }

  statement {
    sid = "ReadCommandResults"
    actions = [
      "ssm:GetCommandInvocation",
      "ssm:ListCommandInvocations",
      "ssm:ListCommands",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_deploy" {
  name   = "deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy.json
}

# Datadog AWS integration

data "aws_iam_policy_document" "datadog_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${var.datadog_aws_account_id}:root"]
    }

    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [var.datadog_external_id]
    }
  }
}

resource "aws_iam_role" "datadog" {
  name               = "real-feel-datadog-integration"
  assume_role_policy = data.aws_iam_policy_document.datadog_trust.json
}

# The CloudWatch reads fetch the SQS, RDS, and EC2 metrics. The describe and
# list reads let Datadog name and tag the queues, the database, and the instances.
data "aws_iam_policy_document" "datadog_metrics" {
  statement {
    sid = "ReadMetrics"
    actions = [
      "cloudwatch:GetMetricData",
      "cloudwatch:ListMetrics",
      "ec2:DescribeInstanceStatus",
      "ec2:DescribeInstances",
      "rds:DescribeDBInstances",
      "rds:ListTagsForResource",
      "sqs:ListQueues",
      "tag:GetResources",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "datadog_metrics" {
  name   = "metrics"
  role   = aws_iam_role.datadog.id
  policy = data.aws_iam_policy_document.datadog_metrics.json
}
