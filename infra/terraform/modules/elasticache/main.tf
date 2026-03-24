
# ElastiCache Redis 7 — r7g.large for session store, rate limiting, query cache

variable "cluster_id"    { default = "llm-platform-redis" }
variable "node_type"     { default = "cache.r7g.large" }
variable "vpc_id"        {}
variable "subnet_ids"    { type = list(string) }
variable "allowed_sg_id" {}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.cluster_id}-subnet-group"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "redis" {
  name   = "${var.cluster_id}-sg"
  vpc_id = var.vpc_id
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.allowed_sg_id]
  }
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = var.cluster_id
  description                = "LLM Platform Redis — session store, rate limiting, cache"
  node_type                  = var.node_type
  num_cache_clusters         = 2    # primary + replica
  automatic_failover_enabled = true
  multi_az_enabled           = true
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  engine_version             = "7.0"
  parameter_group_name       = "default.redis7"
  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.redis.id]

  tags = { Project = "llm-platform", ManagedBy = "terraform" }
}

output "primary_endpoint" {
  value = aws_elasticache_replication_group.main.primary_endpoint_address
}
output "reader_endpoint" {
  value = aws_elasticache_replication_group.main.reader_endpoint_address
}
