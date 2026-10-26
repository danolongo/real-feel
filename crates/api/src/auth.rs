use axum::{
    extract::{Request, State},
    http::header::AUTHORIZATION,
    middleware::Next,
    response::Response,
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use sha2::{Digest, Sha256};

use crate::{ApiError, AppState};

/// Returns a new key: 32 random bytes in URL-safe base64.
pub fn generate_key() -> String {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("the operating system has no random bytes");
    URL_SAFE_NO_PAD.encode(bytes)
}

pub fn hash_key(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

/// Looks up the key in the `Authorization: Bearer` header. Stores the `db::ApiKey` in the request
/// extensions. A missing, unknown, or revoked key gets `unauthorized`.
pub async fn require_api_key(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let key = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .ok_or(ApiError::Unauthorized)?;
    let api_key = db::find_active_api_key(&state.db, &hash_key(key))
        .await?
        .ok_or(ApiError::Unauthorized)?;
    request.extensions_mut().insert(api_key);
    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_key_is_url_safe_and_holds_32_bytes() {
        let key = generate_key();
        assert_eq!(key.len(), 43);
        assert!(
            key.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        );
        assert_ne!(key, generate_key());
    }

    #[test]
    fn hash_key_is_the_sha256_of_the_key() {
        let hex: String = hash_key("abc").iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
