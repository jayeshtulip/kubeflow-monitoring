# LLM Platform v2.0 — Production Terraform
# Account: 659071697671  Region: us-east-1
# Instance sizing: scaled for available quotas
#   Standard vCPU: 32 requested (2x t3.xlarge = 8 vCPU used)
#   G-vCPU:        8 requested  (1x g4dn.2xlarge = 8 vCPU used)

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  
}

provider "aws" {
  region = "us-east-1"
}

locals {
  cluster_name = "llm-platform-prod"
  tags = {
    Project     = "llm-platform"
    Environment = "prod"
    ManagedBy   = "terraform"
    Owner       = "jayesh"
  }
}

# ── VPC ───────────────────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.cluster_name}-vpc"
  cidr = "10.0.0.0/16"
  azs  = ["us-east-1a", "us-east-1b"]

  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true   # cost saving — use one NAT GW
  enable_dns_hostnames = true

  public_subnet_tags  = { "kubernetes.io/role/elb" = "1" }
  private_subnet_tags = { "kubernetes.io/role/internal-elb" = "1" }

  tags = local.tags
}

# ── EKS Cluster ───────────────────────────────────────────────────────────────
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = local.cluster_name
  cluster_version = "1.29"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access = true

  eks_managed_node_groups = {

    # CPU workers: 2x t3.xlarge (4 vCPU, 16GB each = 8 vCPU total)
    # Hosts: Kubeflow Pipelines, LangGraph agents, FastAPI, Qdrant, Ollama
    cpu_workers = {
      instance_types = ["t3.xlarge"]
      min_size       = 2
      max_size       = 4
      desired_size   = 2

      disk_size = 100

      labels = {
        role        = "worker"
        Environment = "prod"
      }

      tags = local.tags
    }

    # GPU worker: 1x g4dn.2xlarge (T4 16GB, 8 vCPU)
    # Hosts: vLLM serving Mistral-7B-Instruct-GPTQ-4bit (~4GB VRAM)
    # Also: Triton FIL for XGBoost/CatBoost (anomaly detection)
    gpu_worker = {
      instance_types = ["g4dn.2xlarge"]
      min_size       = 0          # scale to 0 when not needed
      max_size       = 1
      desired_size   = 0

      disk_size = 100             # model weights + container layers

      labels = {
        role                  = "gpu"
        "nvidia.com/gpu.type" = "T4"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]

      tags = merge(local.tags, { GpuNode = "true" })
    }
  }

  tags = local.tags
}

# ── RDS PostgreSQL ─────────────────────────────────────────────────────────────
resource "aws_db_subnet_group" "postgres" {
  name       = "${local.cluster_name}-postgres-subnet"
  subnet_ids = module.vpc.private_subnets
  tags       = local.tags
}

resource "aws_db_instance" "postgres" {
  identifier        = "${local.cluster_name}-postgres"
  engine            = "postgres"
  engine_version    = "15.8"
  instance_class    = "db.t3.medium"   # scaled down from r6g.large
  allocated_storage = 20
  storage_type      = "gp3"

  db_name  = "llm_platform"
  username = "llm_admin"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.postgres.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  multi_az                = false      # single-AZ to save cost
  skip_final_snapshot     = false
  final_snapshot_identifier = "${local.cluster_name}-final-snapshot"
  deletion_protection     = true

  tags = local.tags
}

# ── ElastiCache Redis ──────────────────────────────────────────────────────────
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${local.cluster_name}-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id      = "${local.cluster_name}-redis"
  engine          = "redis"
  engine_version  = "7.0"
  node_type       = "cache.t3.micro"   # scaled down from r7g.large
  num_cache_nodes = 1
  port            = 6379

  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]

  tags = local.tags
}

# ── S3 Buckets ────────────────────────────────────────────────────────────────
resource "aws_s3_bucket" "dvc_remote" {
  bucket = "llm-platform-dvc-remote-659071697671"
  tags   = merge(local.tags, { Purpose = "DVC data versioning" })
}

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "llm-platform-mlflow-artifacts-659071697671"
  tags   = merge(local.tags, { Purpose = "MLflow experiment artifacts" })
}

resource "aws_s3_bucket" "documents" {
  bucket = "llm-platform-documents-659071697671"
  tags   = merge(local.tags, { Purpose = "Source documents for RAG" })
}

resource "aws_s3_bucket_versioning" "dvc" {
  bucket = aws_s3_bucket.dvc_remote.id
  versioning_configuration { status = "Enabled" }
}

# ── Security Groups ───────────────────────────────────────────────────────────
resource "aws_security_group" "rds" {
  name   = "${local.cluster_name}-rds-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.tags
}

resource "aws_security_group" "redis" {
  name   = "${local.cluster_name}-redis-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.tags
}

# ── Variables ────────────────────────────────────────────────────────────────
variable "db_password" {
  description = "RDS PostgreSQL master password"
  type        = string
  sensitive   = true
}

# ── Outputs ──────────────────────────────────────────────────────────────────
output "cluster_name"     { value = module.eks.cluster_name }
output "cluster_endpoint" { value = module.eks.cluster_endpoint }
output "rds_endpoint"     { value = aws_db_instance.postgres.endpoint }
output "redis_endpoint"   { value = aws_elasticache_cluster.redis.cache_nodes[0].address }
output "ecr_api_url"      { value = "659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/api" }
output "ecr_agents_url"   { value = "659071697671.dkr.ecr.us-east-1.amazonaws.com/llm-platform/agents" }
output "dvc_bucket"       { value = aws_s3_bucket.dvc_remote.id }
output "mlflow_bucket"    { value = aws_s3_bucket.mlflow_artifacts.id }
