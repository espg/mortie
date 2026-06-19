//! Core morton encoding functions
//!
//! The bare-`i64` morton channel carries the packed `decimal_morton` word
//! (issue #48); these two functions are the thin reinterpret-and-(de)code bridge
//! between that word and a HEALPix NESTED `(cell, depth)`. The retired legacy
//! decimal encoder (`fast_norm2mort_scalar`) and its `POWERS_OF_10`/`POWERS_OF_4`
//! tables were removed with the flip; the one-way legacy *decode* needed by the
//! converter now lives inlined in `decimal_morton::from_legacy_decimal`.

/// Decode a packed-u64 morton word (bit-reinterpreted to `i64`) into its
/// HEALPix NESTED cell ID and depth.
///
/// The bare-`i64` morton channel carries the canonical packed word
/// (`decimal_morton`, issue #48); this is a thin reinterpret-and-decode over
/// [`crate::decimal_morton::to_nested`]. Base cells 8-11 set bit 63, so the word
/// is read back as `u64` before decoding (the sign is presentation, not data).
///
/// # Arguments
/// * `morton` - Packed morton word, stored as `i64`
///
/// # Returns
/// `(nested_cell_id, depth)` where depth is the HEALPix order
///
/// # Panics
/// Panics if the word is the empty sentinel (0) or carries an invalid prefix.
pub fn mort2nested(morton: i64) -> (u64, u8) {
    match crate::decimal_morton::to_nested(morton as u64) {
        Some((depth, nested)) => (nested, depth),
        None => panic!("Morton index cannot be zero"),
    }
}

/// Pack a HEALPix NESTED cell ID and depth into a morton word (bit-reinterpreted
/// to `i64`).
///
/// The inverse of [`mort2nested`]: routes through
/// [`crate::decimal_morton::from_nested`] and reinterprets the packed `u64` as
/// `i64`. Reaches order 29 (the kernel's `MAX_ORDER`).
///
/// # Arguments
/// * `nested` - HEALPix NESTED cell ID
/// * `depth` - HEALPix depth/order
///
/// # Returns
/// Packed morton word as `i64`
pub fn nested2mort(nested: u64, depth: u8) -> i64 {
    crate::decimal_morton::from_nested(nested, depth) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- packed-word bridge (nested2mort / mort2nested over the kernel) -------
    // After the issue #48 flip these two functions are a thin reinterpret over
    // `decimal_morton::{from_nested, to_nested}`, so the round-trip is the
    // `(nested, depth)` identity across all base cells and the kernel's full
    // order range (0..=29), not the legacy decimal 0..=18 window.

    #[test]
    fn test_nested_roundtrip_northern() {
        for parent in 0..6u64 {
            for order in 0..=crate::decimal_morton::MAX_ORDER {
                let shift = 2 * order as u32;
                let normed = if order == 0 {
                    0
                } else {
                    0b1011 & ((1u64 << shift) - 1)
                };
                let nested = (parent << shift) | normed;
                let morton = nested2mort(nested, order);
                let (n2, d2) = mort2nested(morton);
                assert_eq!(
                    (d2, n2),
                    (order, nested),
                    "northern roundtrip parent={} order={}",
                    parent,
                    order
                );
            }
        }
    }

    #[test]
    fn test_nested_roundtrip_southern() {
        for parent in 6..12u64 {
            for order in 0..=crate::decimal_morton::MAX_ORDER {
                let shift = 2 * order as u32;
                let normed = if order == 0 {
                    0
                } else {
                    0b1110 & ((1u64 << shift) - 1)
                };
                let nested = (parent << shift) | normed;
                let morton = nested2mort(nested, order);
                // Southern base cells (8..=11) set bit 63 -> the i64 is negative.
                if parent >= 8 {
                    assert!(morton < 0, "base {} should set the sign bit", parent);
                }
                let (n2, d2) = mort2nested(morton);
                assert_eq!(
                    (d2, n2),
                    (order, nested),
                    "southern roundtrip parent={} order={}",
                    parent,
                    order
                );
            }
        }
    }

    #[test]
    fn test_order_zero_is_base_cell() {
        // Order 0 = a base cell kept whole; the packed word decodes to depth 0
        // with nested == the base cell.
        for parent in 0..12u64 {
            let morton = nested2mort(parent, 0);
            let (nested, depth) = mort2nested(morton);
            assert_eq!(depth, 0, "order-0 morton must decode to depth 0");
            assert_eq!(nested, parent, "depth-0 nested == base cell");
            assert_eq!(nested2mort(nested, depth), morton, "order-0 roundtrip");
        }
    }

    #[test]
    fn test_mort2nested_nested_value() {
        // For parent=2, normed=0, order=6: nested = 2 * 4096 + 0 = 8192.
        let morton = nested2mort(2 * 4096, 6);
        let (nested, depth) = mort2nested(morton);
        assert_eq!(depth, 6);
        assert_eq!(nested, 2 * 4096); // parent * nside^2
    }

    #[test]
    fn test_nested2mort_matches_kernel() {
        // nested2mort is exactly the kernel pack, reinterpreted to i64.
        let word = nested2mort(0, 6);
        assert_eq!(word, crate::decimal_morton::from_nested(0, 6) as i64);
    }

    #[test]
    fn test_nested_roundtrip_all_orders() {
        for order in 0..=crate::decimal_morton::MAX_ORDER {
            let nested = 3u64 << (2 * order as u32); // parent 3, normed 0
            let morton = nested2mort(nested, order);
            let (n2, d2) = mort2nested(morton);
            assert_eq!((d2, n2), (order, nested), "roundtrip at order {}", order);
        }
    }

    #[test]
    #[should_panic(expected = "Morton index cannot be zero")]
    fn test_mort2nested_rejects_empty() {
        mort2nested(0);
    }
}
