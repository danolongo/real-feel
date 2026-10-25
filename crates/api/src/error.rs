use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
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
        (self.status(), Json(body)).into_response()
    }
}
