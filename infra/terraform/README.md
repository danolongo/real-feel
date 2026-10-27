# Terraform

This folder builds Real Feel on AWS: a VPC, two EC2 instances, an ALB with AWS WAF, RDS Postgres, four SQS queues, two ECR repositories, and two IAM roles.

## Where it sits

```text
infra/terraform/
  bootstrap/   the S3 buckets, the Route 53 hosted zone, and the ACM certificate
  main/        the root configuration that wires the modules
  modules/     network, database, queues, registry, iam, compute, edge
```

`deploy.yml` builds the images and pushes them to ECR. It then runs the restart script on both instances through SSM Run Command. See [the Datadog README](../datadog/README.md) for the dashboard.

## Terms

See [glossary.md](../../docs/glossary.md) for `bootstrap configuration`, `eval bucket`, `eval set`, `manifest`, `restart script`, `ALB`, `WAF`, `hosted zone`, `queue`, `dead-letter queue`, and `worker`.

## How it works

1. You apply the bootstrap configuration once. It creates the S3 bucket for the state file, the eval bucket, the hosted zone, and the certificate.
2. You apply the main configuration. Terraform creates the stack and writes the non-secret settings to SSM Parameter Store as `String` parameters under `/real-feel/`.
3. Terraform creates four `SecureString` parameters with the value `placeholder`. You overwrite them with the real secrets.
4. `deploy.yml` pushes the images and runs the restart script. The restart script reads `/real-feel/` and starts the containers.

## Prerequisites

- Terraform 1.10 or newer.
- The AWS CLI v2.
- AWS credentials in your shell with permission to create the resources. Check them with `aws sts get-caller-identity`.
- The GitHub CLI, logged in with admin rights on the repository.
- A domain name. See [Domain and certificate](#domain-and-certificate).
- A Datadog account and a Datadog API key.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `region` | `us-east-1` | The AWS region. |
| `datadog_external_id` | none | The external ID that Datadog shows when you add the AWS integration. |
| `datadog_site` | `datadoghq.com` | The Datadog site. |
| `domain_name` | `api.danielemartinez.com` | The domain name of the API. Both configurations take it. |
| `waf_rate_limit` | `1000` | The requests that one IP address may send in 5 minutes before WAF answers `429`. |
| `waf_rate_limit_enabled` | `true` | `false` makes the rate rule count requests and block none. |

The bootstrap configuration takes `region` and `domain_name`. The main configuration takes the rest.

## Domain and certificate

The API needs a domain name, because ACM issues a certificate only for a domain.

`domain_name` is `api.danielemartinez.com`. Cloudflare registers `danielemartinez.com` and runs its DNS. Route 53 holds a hosted zone for the `api` subdomain alone, and Cloudflare delegates the subdomain to it.

ACM validates the certificate only after the delegation works. The first bootstrap apply therefore runs in two steps:

1. Create the zone alone:

   ```bash
   terraform -chdir=infra/terraform/bootstrap apply \
     -var domain_name=api.danielemartinez.com -target=aws_route53_zone.main
   terraform -chdir=infra/terraform/bootstrap output zone_name_servers
   ```

2. In the Cloudflare dashboard, open the `danielemartinez.com` zone and then **DNS > Records**. Add one record for each of the 4 name servers: type `NS`, name `api`, content the name server. NS records take no proxy.
3. Check the delegation. `dig NS api.danielemartinez.com +short` prints the 4 Route 53 name servers.
4. Run the full bootstrap apply in [Run it](#run-it), step 1. It creates the certificate and waits until ACM validates it, usually a few minutes.

The bootstrap configuration owns the zone, and `prevent_destroy` keeps it, so the 4 NS records in Cloudflare stay valid.

The bootstrap configuration owns the zone and the certificate. The main configuration reads both by domain name. A `destroy` of the main configuration keeps the name servers and the validated certificate.

## WAF

The main configuration attaches a WAF web ACL to the ALB. The web ACL has two rules:

- **Rate rule.** WAF counts the requests of each IP address over 5 minutes. Above `waf_rate_limit`, WAF answers `429`. The rule covers every route. WAF takes time to start blocking, so a short burst may pass.
- **Core rule set.** The AWS managed core rule set blocks common attack patterns and answers `403`.

## Switches

- **`waf_rate_limit_enabled`.** `false` sets the rate rule action to `count`. WAF records the requests and blocks none. Use it for load tests.
- **`QUERY_QUOTA_ENABLED`.** The `config` map in `main/main.tf` writes it to SSM. `true` turns on the per-key query quota of the API. Change the value and run `apply` and `deploy.yml`. The restart script reads the new value.

## Run it

Run every command from the repository root.

1. Apply the bootstrap configuration once.

   ```bash
   terraform -chdir=infra/terraform/bootstrap init
   terraform -chdir=infra/terraform/bootstrap apply -var domain_name=api.danielemartinez.com
   ```

   On the first apply, create the zone and delegate it first. See [Domain and certificate](#domain-and-certificate).

2. Initialize the main configuration with the bucket from the bootstrap output.

   ```bash
   terraform -chdir=infra/terraform/main init \
     -backend-config="bucket=$(terraform -chdir=infra/terraform/bootstrap output -raw state_bucket_name)"
   ```

3. Get the external ID from Datadog. Open **Integrations** → **AWS** → **Add AWS Account** → **Role Delegation**. Copy the external ID. Stop before you enter a role ARN. You enter it in step 7.

4. Apply the main configuration.

   ```bash
   terraform -chdir=infra/terraform/main apply \
     -var domain_name=api.danielemartinez.com \
     -var datadog_external_id=<external ID> \
     -var datadog_site=us5.datadoghq.com
   ```

   `datadog_site` is the host of your Datadog URL. Replace `us5.datadoghq.com` when your URL differs.

5. Set the four secrets. Terraform ignores later changes to their values, so no secret enters the state file.

   ```bash
   for name in TWITTERAPI_KEY CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_API_TOKEN DD_API_KEY; do
     printf '%s: ' "$name"; read -rs value; echo
     aws ssm put-parameter --name "/real-feel/$name" --type SecureString --value "$value" --overwrite
   done
   ```

6. Set the GitHub repository variable `AWS_DEPLOY_ROLE_ARN` to the role ARN from the output.

   ```bash
   gh variable set AWS_DEPLOY_ROLE_ARN --body "$(terraform -chdir=infra/terraform/main output -raw github_role_arn)"
   ```

7. Finish the Datadog AWS integration. Paste the output of `terraform -chdir=infra/terraform/main output -raw datadog_role_arn` into the role ARN field. Datadog adds the account only when you finish the form. Open the account's **Metric Collection** tab, turn on SQS, RDS, and EC2, turn off every other service, and keep the region `us-east-1`. Then save.

8. Run `deploy.yml` by hand. The new ECR repositories are empty, so no merge to `main` has pushed images yet. GitHub runs `deploy.yml` only from `main`, and the deploy role trusts only `main`. First wait until both instances appear in `aws ssm describe-instance-information`. The instances need a few minutes after `apply` to install Docker and register with SSM.

   ```bash
   gh workflow run deploy.yml
   gh run watch
   ```

9. Create the dashboard. Follow [the Datadog README](../datadog/README.md).

## Store the eval set

The eval set holds X content, so it lives in the private eval bucket and never in the repository. Only your SSO profile reads it. No instance role and no CI role has access.

1. Apply the bootstrap configuration again. It creates the eval bucket.

   ```bash
   terraform -chdir=infra/terraform/bootstrap apply -var domain_name=api.danielemartinez.com
   ```

2. Copy the eval set and its manifest to the bucket.

   ```bash
   BUCKET=$(terraform -chdir=infra/terraform/bootstrap output -raw eval_bucket_name)
   aws s3 cp <eval set file> "s3://$BUCKET/"
   aws s3 cp <manifest file> "s3://$BUCKET/"
   ```

`terraform destroy` on the main configuration never deletes the eval bucket.

## Send one query

Send the request to the domain name over HTTPS. HTTP redirects to HTTPS. `alb_dns_name` shows the ALB that the domain name points at.

```bash
API=api.danielemartinez.com
curl -s -X POST "https://$API/queries" -H 'content-type: application/json' -d '{"text": "ai"}'
```

The API answers `202` with a `query_id`. Poll the query until `status` is `done`.

```bash
curl -s "https://$API/queries/<query_id>"
```

## Teardown

Destroy the main configuration. The bootstrap configuration stays, because the state bucket, the eval bucket, and the hosted zone have `prevent_destroy`.

```bash
terraform -chdir=infra/terraform/main destroy \
  -var domain_name=api.danielemartinez.com \
  -var datadog_external_id=<external ID> \
  -var datadog_site=us5.datadoghq.com
```

After a new `apply`, repeat steps 5, 6, 7, and 8. A new `apply` creates new placeholder parameters, a new role ARN, and empty ECR repositories.

## Costs

These are estimates, not quotes. They use us-east-1 on-demand prices on 2026-10-25 and 730 hours per month. The EC2 price comes from a [third-party price table](https://calculator.holori.com/aws/ec2/t4g.small/us-east-1), and the RDS price comes from a [third-party RDS price table](https://www.bytebase.com/dbcost/rds/instance/db.t4g.micro/). The gp3 rate, the public IPv4 rate, and the ECR, SQS, Secrets Manager, and S3 rates come from AWS's published list prices. Nobody re-checked them on 2026-10-25. Check the [AWS Pricing Calculator](https://calculator.aws/) before you rely on the totals.

| Item | Rate | Monthly (USD) |
|------|------|---------------|
| 2× EC2 `t4g.small` (Graviton, arm64) | 0.0168 USD/h each | 24.53 |
| RDS `db.t4g.micro`, single-AZ | about 0.016 USD/h | 11.68 |
| RDS storage, 20 GB gp3 | 0.115 USD/GB-month | 2.30 |
| 2 to 3 public IPv4 addresses | 0.005 USD/h each | 7.30 to 10.95 |
| ALB | about 16 USD, plus LCU charges | about 16 |
| WAF web ACL, two rules | 5 USD for the web ACL, 1 USD for each rule, 0.60 USD per million requests | about 7 |
| Route 53 hosted zone | 0.50 USD/month | 0.50 |
| ECR storage, about 1 GB | 0.10 USD/GB-month | 0.10 |
| SQS, four queues | first 1 million requests free, then 0.40 USD per million | 0 to 0.50 |
| Secrets Manager, one secret | 0.40 USD/month | 0.40 |
| S3 state bucket and eval bucket | a few MB | under 0.01 |
| **Total** | | **about 70 to 74** |

- The two instances hold two public IPv4 addresses. A third address appears while you replace an instance. The ALB holds its own addresses, which the ALB price includes.
- The ALB and WAF bill by the hour, so a `destroy` of the main configuration stops both. The hosted zone keeps billing.
- Data transfer, CloudWatch Logs, and the Datadog subscription cost extra.
- RDS keeps one day of backups, which the free backup allowance covers.
- The twitterapi.io and Cloudflare charges for real queries cost extra.
- You run `terraform destroy` at the end of every work session. A destroyed stack costs only the two S3 buckets and the hosted zone. A running stack costs about 17 USD a week.
- You run `apply` and `deploy.yml` at the start of the next session, then repeat steps 5 to 8.

## Failure modes

- **`apply` fails because the GitHub OIDC provider already exists.** Another project in the account created it. Import it with `terraform -chdir=infra/terraform/main import module.iam.aws_iam_openid_connect_provider.github <provider ARN>`, then run `apply` again.
- **`apply` fails with an access error.** Your AWS credentials lack a permission. Fix the credentials and run `apply` again.
- **`deploy.yml` cannot assume the role.** `AWS_DEPLOY_ROLE_ARN` is wrong or missing. Set it again from step 6.
- **The restart script fails on an instance.** Run `aws ssm start-session --target <instance id>`, then read the script output. A secret that still holds `placeholder` is the usual cause. Redo step 5.
- **The bootstrap `apply` waits on the certificate.** ACM cannot see the DNS validation record. Add the `zone_name_servers` output to Cloudflare as NS records named `api`.
- **The main `apply` finds no certificate.** The bootstrap configuration has not issued one for `domain_name`. Apply the bootstrap configuration with the same `domain_name`.
- **The ALB answers `502`.** The health check fails. Check that the API container runs, and that `GET /health` answers `200` on port 80 of the API instance.
- **WAF answers `429` to a legitimate client.** The client passed `waf_rate_limit`. Raise the limit and run `apply`.
- **Datadog shows no SQS, RDS, or EC2 metrics.** The integration does not have the role ARN yet, or the service is off in **Metric Collection**. Redo step 7.

## Decisions

The M3 plan in `docs/refactor-plan.md` shaped this module.
