# Load test sizes. Run:
#   terraform -chdir=infra/terraform/main apply -var-file=loadtest.tfvars \
#     -var datadog_external_id=<external ID>
#
# This file runs system-only mode: the synthetic scraper and the fake
# classifier replace twitterapi.io and Cloudflare, and the query quota is off.
# For full mode, which calls the real services:
#   - Delete the three config_overrides entries below, or set SCRAPER = "twitterapi",
#     CLASSIFIER = "hosted", and QUERY_QUOTA_ENABLED = "true".
#   - Keep the real secrets in SSM. See the Terraform README, step 5.
# Run terraform destroy when the load test session ends.

api_instance_type        = "c7g.large"
scraper_instance_type    = "c7g.large"
classifier_instance_type = "c7g.large"

api_min_size = 1
api_max_size = 3

scraper_min_size = 1
scraper_max_size = 2

classifier_min_size = 1
classifier_max_size = 2

db_instance_class = "db.m7g.large"

waf_rate_limit_enabled = false

config_overrides = {
  SCRAPER             = "synthetic"
  CLASSIFIER          = "fake"
  QUERY_QUOTA_ENABLED = "false"
}
