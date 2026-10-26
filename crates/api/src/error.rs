use std::time::Duration;

use axum::{
    Json,
    http::{HeaderValue, StatusCode, header::RETRY_AFTER},
    response::{IntoResponse, Response},
};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("the Authorization header must hold a valid API key")]
    Unauthorized,
    #[error("the API key has used its query quota")]
    QuotaExceeded { retry_after: Duration },
    #[error("the query text must hold 1 to 512 characters after trimming")]
    InvalidQuery,
    #[error("the query ID must be a UUID")]
    InvalidQueryId,
    #[error("limit must be 1 to 200, and after must be a tweet ID")]
    InvalidCursor,
    #[error("no query has that ID")]
    QueryNotFound,
    #[error("the API could not send the scraper job")]
    QueueUnavailable,
    #[error("internal error")]
    Internal,
}

impl ApiError {
    fn code(&self) -> &'static str {
        match self {
            Self::Unauthorized => "unauthorized",
            Self::QuotaExceeded { .. } => "quota_exceeded",
            Self::InvalidQuery => "invalid_query",
            Self::InvalidQueryId => "invalid_query_id",
            Self::InvalidCursor => "invalid_cursor",
            Self::QueryNotFound => "query_not_found",
            Self::QueueUnavailable => "queue_unavailable",
            Self::Internal => "internal",
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::QuotaExceeded { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::InvalidQuery | Self::InvalidQueryId | Self::InvalidCursor => {
                StatusCode::BAD_REQUEST
            }
            Self::QueryNotFound => StatusCode::NOT_FOUND,
            Self::QueueUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl From<db::Error> for ApiError {
    fn from(error: db::Error) -> Self {
        tracing::error!(%error, "database call failed");
        Self::Internal
    }
}

#[derive(Serialize)]
struct Body {
    code: &'static str,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Body {
            code: self.code(),
            message: self.to_string(),
        };
        let mut response = (self.status(), Json(body)).into_response();
        if let Self::QuotaExceeded { retry_after } = self {
            let secs = retry_after.as_secs_f64().ceil().max(1.0) as u64;
            response
                .headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from(secs));
        }
        response
    }
}
