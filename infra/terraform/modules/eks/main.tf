
# EKS Cluster — 10x c5.4xlarge worker nodes
# Platform: RAGAS-KUBEFLOW-MLOPS v2.0

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "cluster_name"    { default = "llm-platform" }
variable "cluster_version" { default = "1.29" }
variable "region"          { default = "us-east-1" }
variable "node_count"      { default = 10 }
variable "node_type"       { default = "c5.4xlarge" }
variable "vpc_id"          {}
variable "subnet_ids"      { type = list(string) }

resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  version  = var.cluster_version
  role_arn = aws_iam_role.cluster.arn

  vpc_config {
    subnet_ids              = var.subnet_ids
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  tags = {
    Project     = "llm-platform"
    ManagedBy   = "terraform"
  }
}

resource "aws_eks_node_group" "workers" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "llm-workers"
  node_role_arn   = aws_iam_role.node.arn
  subnet_ids      = var.subnet_ids
  instance_types  = [var.node_type]

  scaling_config {
    desired_size = var.node_count
    max_size     = var.node_count + 2
    min_size     = 3
  }

  labels = {
    role = "llm-worker"
  }

  taint {
    key    = "workload"
    value  = "llm"
    effect = "NO_SCHEDULE"
  }
}

# IAM roles (simplified — full policies in modules/iam)
resource "aws_iam_role" "cluster" {
  name = "${var.cluster_name}-cluster-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "eks.amazonaws.com" },
                   Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role" "node" {
  name = "${var.cluster_name}-node-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" },
                   Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "eks_cni" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

output "cluster_endpoint"    { value = aws_eks_cluster.main.endpoint }
output "cluster_name"        { value = aws_eks_cluster.main.name }
output "cluster_certificate" { value = aws_eks_cluster.main.certificate_authority[0].data }
output "node_role_arn"       { value = aws_iam_role.node.arn }
