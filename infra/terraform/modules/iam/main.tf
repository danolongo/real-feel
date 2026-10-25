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
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/main"]
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
    sid = "ReadCommandResults"
    actions = [
      "ssm:GetCommandInvocation",
      "ssm:ListCommandInvocations",
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

# The minimum set that Datadog's SQS integration lists, plus the CloudWatch
# reads that Datadog uses to fetch the SQS metrics.
data "aws_iam_policy_document" "datadog_sqs" {
  statement {
    sid = "ReadSqsMetrics"
    actions = [
      "cloudwatch:GetMetricData",
      "cloudwatch:ListMetrics",
      "sqs:ListQueues",
      "tag:GetResources",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "datadog_sqs" {
  name   = "sqs-metrics"
  role   = aws_iam_role.datadog.id
  policy = data.aws_iam_policy_document.datadog_sqs.json
}
