# Terraform backend — S3 bucket already created
# Run: terraform init -backend-config=backend.tf
terraform {
  backend "s3" {
    bucket  = "llm-platform-terraform-state-659071697671"
    key     = "prod/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}
