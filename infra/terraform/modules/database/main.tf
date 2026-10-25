locals {
  name = "real-feel"
}

resource "aws_db_subnet_group" "main" {
  name       = local.name
  subnet_ids = var.private_subnet_ids

  tags = {
    Name = local.name
  }
}

# RDS stores the master password in Secrets Manager. pgvector ships with RDS, so no extension step runs here.
resource "aws_db_instance" "main" {
  identifier = local.name

  engine         = "postgres"
  engine_version = "16"
  instance_class = var.instance_class

  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true

  db_name                     = "realfeel"
  username                    = "realfeel"
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [var.db_sg_id]
  publicly_accessible    = false
  multi_az               = false

  backup_retention_period = 1
  skip_final_snapshot     = true
  deletion_protection     = false
}
