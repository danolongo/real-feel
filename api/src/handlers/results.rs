use aws_sdk_dynamodb::Client as DynamoClient;
use lambda_http::{Body, Error, Request, Response};
use std::sync::Arc;

pub async fn handle(_req: Request, _dynamo: Arc<DynamoClient>) -> Result<Response<Body>, Error> {
    Ok(Response::builder()
        .status(200)
        .body("results placeholder".into())?)
}
