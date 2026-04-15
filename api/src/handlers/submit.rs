use lambda_http::{Body, Error, Request, Response};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// What we expect in the POST body: { "query": "show me AI bot tweets" }
#[derive(Deserialize)]
struct SubmitRequest {
    query: String,
}

// What we return: { "query_id": "...", "status": "processing" }
#[derive(Serialize)]
struct SubmitResponse {
    query_id: String,
    status: String,
}

pub async fn handle(req: Request) -> Result<Response<Body>, Error> {
    // 1. Parse the request body
    // Lambda gives us the body as Body::Text (JSON string) or Body::Binary (bytes)
    let submit_req: SubmitRequest = match req.body() {
        Body::Text(s) => serde_json::from_str(s)?,
        Body::Binary(b) => serde_json::from_slice(b)?,
        _ => {
            return Ok(Response::builder()
                .status(400)
                .body("missing body".into())?)
        }
    };

    // 2. Generate a unique ID for this query
    let query_id = Uuid::new_v4().to_string();

    tracing::info!(query_id = %query_id, query = %submit_req.query, "received query");

    // 3. TODO: write to DynamoDB (status: processing)
    // 4. TODO: publish to Kafka "queries" topic

    // 5. Return immediately, client will poll for results
    let response = SubmitResponse {
        query_id,
        status: "processing".to_string(),
    };

    Ok(Response::builder()
        .status(202) // 202 Accepted, request received, processing async
        .header("content-type", "application/json")
        .body(serde_json::to_string(&response)?.into())?)
}
