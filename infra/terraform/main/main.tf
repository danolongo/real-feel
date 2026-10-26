module "network" {
  source = "../modules/network"

  operator_cidr = var.operator_cidr
}

module "database" {
  source = "../modules/database"

  private_subnet_ids = module.network.private_subnet_ids
  db_sg_id           = module.network.db_sg_id
}

module "queues" {
  source = "../modules/queues"
}

module "registry" {
  source = "../modules/registry"
}

module "iam" {
  source = "../modules/iam"

  datadog_external_id = var.datadog_external_id
}

module "compute" {
  source = "../modules/compute"

  public_subnet_ids     = module.network.public_subnet_ids
  api_sg_id             = module.network.api_sg_id
  worker_sg_id          = module.network.worker_sg_id
  repository_arns       = module.registry.repository_arns
  api_repository_url    = module.registry.api_repository_url
  worker_repository_url = module.registry.worker_repository_url
  queue_arns            = module.queues.queue_arns
  db_secret_arn         = module.database.db_secret_arn

  config = {
    SCRAPER_QUEUE_URL    = module.queues.scraper_queue_url
    SCRAPER_DLQ_URL      = module.queues.scraper_dlq_url
    CLASSIFIER_QUEUE_URL = module.queues.classifier_queue_url
    CLASSIFIER_DLQ_URL   = module.queues.classifier_dlq_url
    AWS_REGION           = var.region
    DB_HOST              = module.database.db_host
    DB_PORT              = tostring(module.database.db_port)
    DB_NAME              = module.database.db_name
    DB_SECRET_ARN        = module.database.db_secret_arn
    DD_SITE              = var.datadog_site

    DEFAULT_MODEL_VERSION    = "clef-flash-hosted"
    DEFAULT_METADATA_VERSION = "v6"
    DEFAULT_QUESTION_VERSION = "v4"
    # The best F1 on the tune split of the eval set.
    BOT_THRESHOLD = "0.123"
    # BOT_THRESHOLD = "0.173" # 0.123 plus a 0.05 margin for searches the eval set does not cover.
    BATCH_SIZE = "25"
    # Stays below the 300 s visibility timeout of the queues.
    JOB_DEADLINE_SECS = "240"
  }
}
