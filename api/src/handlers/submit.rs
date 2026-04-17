use aws_sdk_dynamodb::{types::AttributeValue, Client as DynamoClient};
use chrono::Utc;
use lambda_http::{Body, Error, Request, Response};
use rskafka::{
    client::partition::Compression,
    client::{partition::UnknownTopicHandling, ClientBuilder},
    record::Record,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
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

#[derive(Serialize)]
struct KafkaQuery {
    query_id: String,
    search: String,
}

pub async fn handle(req: Request, dynamo: Arc<DynamoClient>) -> Result<Response<Body>, Error> {
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

    // 3. write to DynamoDB (status: processing)
    dynamo
        .put_item()
        .table_name("queries")
        .item("query_id", AttributeValue::S(query_id.clone()))
        .item("status", AttributeValue::S("processing".to_string()))
        .item("query", AttributeValue::S(submit_req.query.clone()))
        .item("created_at", AttributeValue::S(Utc::now().to_rfc3339()))
        .send()
        .await?;

    // 4. TODO: publish to Kafka "queries" topic
    let kafka_msg = KafkaQuery {
        query_id: query_id.clone(),
        search: submit_req.query.clone(),
    };

    let brokers = std::env::var("KAFKA_BROKERS")
        .unwrap_or_else(|_| "localhost:9092".to_string());

    let client = ClientBuilder::new(vec![brokers])
        .build()
        .await?;

    let partition_client = client
        .partition_client("queries", 0, UnknownTopicHandling::Retry)
        .await?;

    partition_client.produce(vec![Record {
        key: None,
        value: Some(serde_json::to_vec(&kafka_msg)?),
        timestamp: Utc::now(),
        headers: Default::default(),
    }], Compression::NoCompression).await?;

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
