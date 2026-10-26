use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Error;
use crate::item::{EvalItem, Label, parse_jsonl};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LabelCounts {
    pub human: usize,
    pub bot: usize,
    pub agent: usize,
}

/// The repository stores this JSON in place of the eval set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub sha256: String,
    pub items: usize,
    pub labels: LabelCounts,
}

/// Hashes the exact bytes of an eval set file and counts its items.
pub fn from_bytes(bytes: &[u8]) -> Result<Manifest, Error> {
    let digest = Sha256::digest(bytes);
    let sha256 = digest.iter().map(|byte| format!("{byte:02x}")).collect();

    let text = String::from_utf8_lossy(bytes);
    let items: Vec<EvalItem> = parse_jsonl(&text)?;
    let count = |label: Label| items.iter().filter(|item| item.label == label).count();
    Ok(Manifest {
        sha256,
        items: items.len(),
        labels: LabelCounts {
            human: count(Label::Human),
            bot: count(Label::Bot),
            agent: count(Label::Agent),
        },
    })
}

pub fn from_file(path: &Path) -> Result<Manifest, Error> {
    let bytes = std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_of_empty_file_is_the_known_digest() {
        let manifest = from_bytes(b"").unwrap();
        assert_eq!(
            manifest.sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(manifest.items, 0);
    }

    #[test]
    fn the_fixture_has_four_human_four_bot_two_agent() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/eval-set.jsonl");
        let manifest = from_file(&path).unwrap();
        assert_eq!(manifest.items, 10);
        assert_eq!(
            manifest.labels,
            LabelCounts {
                human: 4,
                bot: 4,
                agent: 2
            }
        );
        assert_eq!(manifest.sha256.len(), 64);
    }
}
