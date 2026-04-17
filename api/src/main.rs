mod handlers;

use aws_sdk_dynamodb::Client as DynamoClient;
use lambda_http::http::Method;
use lambda_http::{run, service_fn, Body, Error, Request, Response};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt().json().init();

    let config = aws_config::load_from_env().await;
    let dynamo = Arc::new(DynamoClient::new(&config));

    run(service_fn(move |req| {
        let dynamo = dynamo.clone();
        async move { handler(req, dynamo).await }
    }))
    .await
}

async fn handler(req: Request, dynamo: Arc<DynamoClient>) -> Result<Response<Body>, Error> {
    let path = req.uri().path().trim_start_matches("/Prod");
    match (req.method(), path) {
        (&Method::POST, "/query") => handlers::submit::handle(req, dynamo).await,
        (&Method::GET, p) if p.starts_with("/query/") => {
            handlers::results::handle(req, dynamo).await
        }
        _ => Ok(Response::builder()
            .status(404)
            .body(format!("Not found: {} {}", req.method(), req.uri().path()).into())?),
    }
}
