mod handlers;

use lambda_http::http::Method;
use lambda_http::{run, service_fn, Body, Error, Request, Response};

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt().json().init();
    run(service_fn(handler)).await
}

async fn handler(req: Request) -> Result<Response<Body>, Error> {
    let path = req.uri().path().trim_start_matches("/Prod");
    match (req.method(), path) {
        (&Method::POST, "/query") => handlers::submit::handle(req).await,
        (&Method::GET, p) if p.starts_with("/query/") => handlers::results::handle(req).await,
        _ => Ok(Response::builder().status(404).body(format!("Not found: {} {}", req.method(), req.uri().path()).into())?),
    }
}
