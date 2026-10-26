/// Percent of accounts that go to the tune split.
const TUNE_PERCENT: u64 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Tune,
    Test,
}

/// FNV-1a over the seed bytes and the account ID bytes, then a 64-bit finalizer.
/// The result is the same on every platform and Rust version.
fn stable_hash(seed: u64, account_id: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in seed.to_le_bytes().into_iter().chain(account_id.bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    // FNV-1a mixes the low bits well and the high bits poorly. The finalizer spreads the change.
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash
}

/// Assigns one account to the tune split (30%) or the test split (70%).
pub fn assign(seed: u64, account_id: &str) -> Split {
    if stable_hash(seed, account_id) % 100 < TUNE_PERCENT {
        Split::Tune
    } else {
        Split::Test
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_and_id_give_the_same_split() {
        for id in ["1", "42", "1850023456789"] {
            assert_eq!(assign(7, id), assign(7, id));
        }
    }

    #[test]
    fn the_hash_is_pinned_so_splits_never_drift() {
        // Both values come from an independent Python implementation of the algorithm.
        assert_eq!(stable_hash(0, ""), 3_530_818_670_739_341_123);
        assert_eq!(stable_hash(42, "12345"), 13_530_000_581_075_853_156);
    }

    #[test]
    fn about_thirty_percent_of_accounts_go_to_tune() {
        let total = 10_000;
        let tune = (0..total)
            .filter(|n| assign(42, &n.to_string()) == Split::Tune)
            .count();
        assert!((2_800..=3_200).contains(&tune), "tune count was {tune}");
    }

    #[test]
    fn a_different_seed_changes_some_splits() {
        let moved = (0..200)
            .filter(|n| assign(1, &n.to_string()) != assign(2, &n.to_string()))
            .count();
        assert!(moved > 0);
    }
}
