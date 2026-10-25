# The bucket name comes from the bootstrap configuration. Pass it at init:
#   terraform init -backend-config="bucket=$(terraform -chdir=../bootstrap output -raw state_bucket_name)"
terraform {
  backend "s3" {
    key          = "main/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}
