resource "aws_lb" "api" {
  name               = "real-feel-api"
  load_balancer_type = "application"
  subnets            = var.public_subnet_ids
  security_groups    = [var.alb_sg_id]
}

resource "aws_lb_target_group" "api" {
  name     = "real-feel-api"
  vpc_id   = var.vpc_id
  port     = 80
  protocol = "HTTP"

  deregistration_delay = 30

  health_check {
    path     = "/health"
    protocol = "HTTP"
    matcher  = "200"
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"

    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_route53_record" "api" {
  zone_id = var.zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_lb.api.dns_name
    zone_id                = aws_lb.api.zone_id
    evaluate_target_health = true
  }
}

resource "aws_wafv2_web_acl" "api" {
  name  = "real-feel-api"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "rate-limit"
    priority = 1

    # The 5-minute window is the default evaluation window of a rate-based rule.
    statement {
      rate_based_statement {
        limit                 = var.waf_rate_limit
        aggregate_key_type    = "IP"
        evaluation_window_sec = 300
      }
    }

    action {
      dynamic "block" {
        for_each = var.waf_rate_limit_enabled ? [1] : []

        content {
          custom_response {
            response_code = 429
          }
        }
      }

      dynamic "count" {
        for_each = var.waf_rate_limit_enabled ? [] : [1]

        content {}
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "real-feel-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "core-rule-set"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "real-feel-core-rule-set"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "real-feel-api"
    sampled_requests_enabled   = true
  }
}

resource "aws_wafv2_web_acl_association" "api" {
  resource_arn = aws_lb.api.arn
  web_acl_arn  = aws_wafv2_web_acl.api.arn
}
