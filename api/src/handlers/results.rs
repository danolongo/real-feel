use lambda_http::{Body, Error, Request, Response};

pub async fn handle(_req: Request) -> Result<Response<Body>, Error> {
    Ok(Response::builder()
        .status(200)
        .body("results placeholder".into())?)
}
