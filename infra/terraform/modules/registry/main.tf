resource "aws_ecr_repository" "api" {
  name         = "real-feel/api"
  force_delete = true
}

resource "aws_ecr_repository" "worker" {
  name         = "real-feel/worker"
  force_delete = true
}

# Each deploy pushes one image per repository. Keep the last 10, so old images stop costing storage.
resource "aws_ecr_lifecycle_policy" "keep_last_10" {
  for_each   = { api = aws_ecr_repository.api.name, worker = aws_ecr_repository.worker.name }
  repository = each.value
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}
