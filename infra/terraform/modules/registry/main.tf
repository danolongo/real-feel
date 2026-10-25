resource "aws_ecr_repository" "api" {
  name         = "real-feel/api"
  force_delete = true
}

resource "aws_ecr_repository" "worker" {
  name         = "real-feel/worker"
  force_delete = true
}
