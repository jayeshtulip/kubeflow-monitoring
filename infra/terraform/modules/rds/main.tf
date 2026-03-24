
# RDS PostgreSQL 15 — Multi-AZ for MLflow + Kubeflow metadata + RAGAS results

variable "identifier"    { default = "llm-platform" }
variable "db_name"       { default = "llm_platform" }
variable "db_user"       { default = "llm_admin" }
variable "instance_class"{ default = "db.r6g.large" }
variable "vpc_id"        {}
variable "subnet_ids"    { type = list(string) }
variable "allowed_sg_id" {}

resource "aws_db_subnet_group" "main" {
  name       = "${var.identifier}-subnet-group"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "rds" {
  name   = "${var.identifier}-rds-sg"
  vpc_id = var.vpc_id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.allowed_sg_id]
  }
}

resource "aws_db_instance" "main" {
  identifier             = var.identifier
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = var.instance_class
  allocated_storage      = 100
  max_allocated_storage  = 500
  storage_encrypted      = true
  db_name                = var.db_name
  username               = var.db_user
  manage_master_user_password = true   # AWS Secrets Manager rotation
  multi_az               = true
  backup_retention_period = 35
  deletion_protection    = true
  skip_final_snapshot    = false
  final_snapshot_identifier = "${var.identifier}-final"
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  parameter_group_name   = "default.postgres15"

  tags = { Project = "llm-platform", ManagedBy = "terraform" }
}

output "endpoint" { value = aws_db_instance.main.address }
output "port"     { value = aws_db_instance.main.port }
output "db_name"  { value = var.db_name }
