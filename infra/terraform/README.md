# Terraform

This folder builds Real Feel on AWS: a VPC, two EC2 instances, RDS Postgres, four SQS queues, two ECR repositories, and two IAM roles.

## Where it sits

```text
infra/terraform/
  bootstrap/   the S3 bucket for the state file
  main/        the root configuration that wires the modules
  modules/     network, database, queues, registry, iam, compute
```

`deploy.yml` builds the images and pushes them to ECR. It then runs the restart script on both instances through SSM Run Command. See [the Datadog README](../datadog/README.md) for the dashboard.

## Terms

See [glossary.md](../../docs/glossary.md) for `bootstrap configuration`, `restart script`, `queue`, `dead-letter queue`, and `worker`.

## How it works

1. You apply the bootstrap configuration once. It creates the S3 bucket for the state file.
2. You apply the main configuration. Terraform creates the stack and writes the non-secret settings to SSM Parameter Store as `String` parameters under `/real-feel/`.
3. Terraform creates four `SecureString` parameters with the value `placeholder`. You overwrite them with the real secrets.
4. `deploy.yml` pushes the images and runs the restart script. The restart script reads `/real-feel/` and starts the containers.

## Prerequisites

- Terraform 1.10 or newer.
- The AWS CLI v2.
- AWS credentials in your shell with permission to create the resources. Check them with `aws sts get-caller-identity`.
- The GitHub CLI, logged in with admin rights on the repository.
- Your public IPv4 address as a `/32` CIDR, for example `203.0.113.7/32`. `curl -s https://checkip.amazonaws.com` prints the address.
- A Datadog account and a Datadog API key.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `region` | `us-east-1` | The AWS region. |
| `operator_cidr` | none | The CIDR that may reach port 80 of the API. Use your own `/32`. |
| `datadog_external_id` | none | The external ID that Datadog shows when you add the AWS integration. |
| `datadog_site` | `datadoghq.com` | The Datadog site. |

## Run it

Run every command from the repository root.

1. Apply the bootstrap configuration once.

   ```bash
   terraform -chdir=infra/terraform/bootstrap init
   terraform -chdir=infra/terraform/bootstrap apply
   ```

2. Initialize the main configuration with the bucket from the bootstrap output.

   ```bash
   terraform -chdir=infra/terraform/main init \
     -backend-config="bucket=$(terraform -chdir=infra/terraform/bootstrap output -raw state_bucket_name)"
   ```

3. Get the external ID from Datadog. Open **Integrations** → **AWS** → **Add AWS Account** → **Role Delegation**. Copy the external ID. Stop before you enter a role ARN. You enter it in step 7.

4. Apply the main configuration.

   ```bash
   terraform -chdir=infra/terraform/main apply \
     -var operator_cidr=<your IP>/32 \
     -var datadog_external_id=<external ID>
   ```

   Add `-var datadog_site=<site>` when your Datadog site is not `datadoghq.com`.

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

7. Finish the Datadog AWS integration. Paste the output of `terraform -chdir=infra/terraform/main output -raw datadog_role_arn` into the role ARN field. Enable the SQS metrics, then save.

8. Run `deploy.yml` by hand. The new ECR repositories are empty, so no merge to `main` has pushed images yet. GitHub runs `deploy.yml` only from `main`, and the deploy role trusts only `main`. First wait until both instances appear in `aws ssm describe-instance-information`. The instances need a few minutes after `apply` to install Docker and register with SSM.

   ```bash
   gh workflow run deploy.yml
   gh run watch
   ```

9. Create the dashboard. Follow [the Datadog README](../datadog/README.md).

## Send one query

Send the request to the Elastic IP of the API. Your IP must match `operator_cidr`.

```bash
API=$(terraform -chdir=infra/terraform/main output -raw api_public_ip)
curl -s -X POST "http://$API/queries" -H 'content-type: application/json' -d '{"text": "ai"}'
```

The API answers `202` with a `query_id`. Poll the query until `status` is `done`.

```bash
curl -s "http://$API/queries/<query_id>"
```

## Teardown

Destroy the main configuration. The bootstrap configuration stays, because the state bucket has `prevent_destroy`.

```bash
terraform -chdir=infra/terraform/main destroy \
  -var operator_cidr=<your IP>/32 \
  -var datadog_external_id=<external ID>
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
| ECR storage, about 1 GB | 0.10 USD/GB-month | 0.10 |
| SQS, four queues | first 1 million requests free, then 0.40 USD per million | 0 to 0.50 |
| Secrets Manager, one secret | 0.40 USD/month | 0.40 |
| S3 state bucket | a few KB | under 0.01 |
| **Total** | | **about 46 to 50** |

- The two instances hold two public IPv4 addresses. A third address appears while you replace an instance.
- Data transfer, CloudWatch Logs, and the Datadog subscription cost extra.
- RDS keeps one day of backups, which the free backup allowance covers.
- The twitterapi.io and Cloudflare charges for real queries cost extra.
- Stop the cost with `terraform destroy`. A destroyed stack costs only the S3 state bucket.

## Failure modes

- **`apply` fails because the GitHub OIDC provider already exists.** Another project in the account created it. Import it with `terraform -chdir=infra/terraform/main import module.iam.aws_iam_openid_connect_provider.github <provider ARN>`, then run `apply` again.
- **`apply` fails with an access error.** Your AWS credentials lack a permission. Fix the credentials and run `apply` again.
- **`deploy.yml` cannot assume the role.** `AWS_DEPLOY_ROLE_ARN` is wrong or missing. Set it again from step 6.
- **The restart script fails on an instance.** Run `aws ssm start-session --target <instance id>`, then read the script output. A secret that still holds `placeholder` is the usual cause. Redo step 5.
- **The API does not answer `curl`.** Your IP changed, so it no longer matches `operator_cidr`. Run `apply` with the new CIDR.
- **Datadog shows no SQS metrics.** The integration does not have the role ARN yet. Redo step 7.

## Decisions

The M3 plan in `docs/refactor-plan.md` shaped this module.
