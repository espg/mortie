//! Full-resolution `decimal_morton` 64-bit MOC datatype kernel (issue #35).
//!
//! This is the phase-1 encode/decode kernel for the packed 64-bit Morton MOC
//! index that captures the HEALPix order explicitly and reaches order 29.
//!
//! # Bit layout (MSB -> LSB)
//!
//! ```text
//! [ 4-bit prefix ][ 54-bit body (27 x 2-bit) ][ 6-bit suffix ]
//!   63 .. 60        59 ..  6                     5 ..  0
//! ```
//!
//! * **prefix** -- base cell stored as `base_id + 1`, so the 12 HEALPix base
//!   cells occupy `1..=12`. `0` is an empty/null sentinel and `13..=15` are
//!   invalid. The `+1` shift is monotonic, so a raw unsigned sort is preserved
//!   as a Z-order curve (the property issue #35 calls paramount).
//! * **body** -- 27 two-bit tuples, one per order `1..=27`. Order 1 occupies the
//!   highest tuple (bits 59..58), order 27 the lowest (bits 7..6). The stored
//!   value is `0..=3` but is *interpreted* as `1..=4` (a decode-time `+1`,
//!   matching the existing decimal Morton convention).
//! * **suffix** -- 6 bits, understood right-to-left (low bit first):
//!     * rightmost bit `0` => variable length; the 5 bits to its left encode the
//!       number of tuples to read, valid `0..=27` (0 = base-cell-only / order 0).
//!     * rightmost two bits `01` => order 28; suffix bits 5..4 hold the order-28
//!       tuple, bits 3..2 are spare (zero-filled).
//!     * rightmost two bits `11` => order 29; suffix bits 5..4 hold the order-28
//!       tuple and bits 3..2 the order-29 tuple.
//!
//! Storage is **unsigned**; the signed "negative = southern" form is a
//! presentation detail applied elsewhere. Encoding **zero-fills** every bit
//! below an element's order, so two encodings of the same cell are bit-equal
//! (canonical) -- integer equality, hashing, dedup and the raw sort all work.

/// Highest HEALPix order this datatype encodes.
pub const MAX_ORDER: u8 = 29;
/// Number of two-bit tuples held in the body (orders 1..=27).
pub const BODY_TUPLES: u8 = 27;

const PREFIX_SHIFT: u32 = 60;
const SUFFIX_BITS: u32 = 6;
const SUFFIX_MASK: u64 = (1 << SUFFIX_BITS) - 1; // low 6 bits
/// Full 54-bit body mask, sitting just above the suffix.
const BODY_MASK: u64 = ((1u64 << 54) - 1) << SUFFIX_BITS;

/// A decoded `decimal_morton` index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecimalMorton {
    /// HEALPix base cell, `0..=11`.
    pub base_cell: u8,
    /// HEALPix order, `0..=29`.
    pub order: u8,
    /// The per-order tuples, `tuples[i]` for order `i + 1`. Each value is the
    /// stored `0..=3` (not the `1..=4` interpretation). Length == `order`.
    pub tuples: Vec<u8>,
}

/// Errors raised while decoding an untrusted packed word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Prefix was `0` (the empty/null sentinel) -- there is no cell to decode.
    Empty,
    /// Prefix was `13..=15`; only `1..=12` map to a base cell.
    InvalidPrefix(u8),
    /// Variable-length suffix pointed at a tuple count of `28..=31`, which is
    /// nonsensical (orders 28/29 use the dedicated flag forms).
    InvalidSuffixCount(u8),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Empty => write!(f, "empty decimal_morton (prefix 0)"),
            DecodeError::InvalidPrefix(p) => {
                write!(f, "invalid base-cell prefix {} (valid 1..=12)", p)
            }
            DecodeError::InvalidSuffixCount(c) => {
                write!(
                    f,
                    "invalid variable-length suffix count {} (valid 0..=27)",
                    c
                )
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Build the 6-bit suffix value for a given order and its order-28/29 tuples.
#[inline]
fn build_suffix(order: u8, t28: u8, t29: u8) -> u64 {
    match order {
        // variable length: low bit 0, count in the 5 bits above it
        0..=27 => ((order as u64) << 1) & SUFFIX_MASK,
        // order 28: bits 5..4 = tuple-28, bits 3..2 spare (0), bits 1..0 = 01
        28 => (((t28 & 3) as u64) << 4) | 0b01,
        // order 29: bits 5..4 = tuple-28, bits 3..2 = tuple-29, bits 1..0 = 11
        29 => (((t28 & 3) as u64) << 4) | (((t29 & 3) as u64) << 2) | 0b11,
        _ => unreachable!("order > 29 is rejected before build_suffix"),
    }
}

/// Encode a base cell, per-order tuples and order into a packed 64-bit word.
///
/// `tuples` supplies the stored `0..=3` value for each order `1..=order`; only
/// the first `order` entries are read. Any bit below the element's order is
/// zero-filled, so the result is canonical.
///
/// # Panics
/// Panics if `base_cell > 11`, `order > 29`, or `tuples` has fewer than `order`
/// entries -- these are programmer errors on the trusted encode path.
pub fn encode(base_cell: u8, tuples: &[u8], order: u8) -> u64 {
    assert!(
        base_cell <= 11,
        "base_cell must be 0..=11, got {}",
        base_cell
    );
    assert!(order <= MAX_ORDER, "order must be 0..=29, got {}", order);
    assert!(
        tuples.len() >= order as usize,
        "need at least {} tuples for order {}, got {}",
        order,
        order,
        tuples.len()
    );

    let prefix = ((base_cell + 1) as u64) << PREFIX_SHIFT;

    // Body: orders 1..=27 only (orders 28/29 live in the suffix). Order n sits
    // at the tuple `BODY_TUPLES - n` counting from the body's low end.
    let body_orders = order.min(BODY_TUPLES);
    let mut body: u64 = 0;
    for n in 1..=body_orders {
        let pair = (tuples[(n - 1) as usize] & 3) as u64;
        // Order 1 is the highest pair; shift it furthest left within the body.
        let shift = SUFFIX_BITS + 2 * (BODY_TUPLES - n) as u32;
        body |= pair << shift;
    }

    let (t28, t29) = match order {
        28 => (tuples[27], 0),
        29 => (tuples[27], tuples[28]),
        _ => (0, 0),
    };
    let suffix = build_suffix(order, t28, t29);

    prefix | body | suffix
}

/// Read just the order encoded by a packed word's suffix (the "fast path").
///
/// Returns `None` for the nonsensical variable-length counts `28..=31`.
#[inline]
pub fn order_of(word: u64) -> Option<u8> {
    let suffix = word & SUFFIX_MASK;
    if suffix & 1 == 0 {
        let count = (suffix >> 1) as u8; // 0..=31
        if count <= BODY_TUPLES {
            Some(count)
        } else {
            None
        }
    } else if suffix & 0b11 == 0b01 {
        Some(28)
    } else {
        Some(29)
    }
}

/// Read just the base cell (`0..=11`) encoded by a packed word's prefix.
///
/// Returns `None` for the empty sentinel (`0`) or an invalid prefix (`13..=15`).
#[inline]
pub fn base_cell_of(word: u64) -> Option<u8> {
    let prefix = (word >> PREFIX_SHIFT) as u8 & 0x0f;
    match prefix {
        1..=12 => Some(prefix - 1),
        _ => None,
    }
}

/// Decode a packed 64-bit word back into its `DecimalMorton` parts.
pub fn decode(word: u64) -> Result<DecimalMorton, DecodeError> {
    let prefix = (word >> PREFIX_SHIFT) as u8 & 0x0f;
    let base_cell = match prefix {
        0 => return Err(DecodeError::Empty),
        1..=12 => prefix - 1,
        other => return Err(DecodeError::InvalidPrefix(other)),
    };

    let suffix = word & SUFFIX_MASK;
    let order = order_of(word).ok_or(DecodeError::InvalidSuffixCount((suffix >> 1) as u8))?;

    let mut tuples = Vec::with_capacity(order as usize);
    let body_orders = order.min(BODY_TUPLES);
    for n in 1..=body_orders {
        let shift = SUFFIX_BITS + 2 * (BODY_TUPLES - n) as u32;
        tuples.push(((word >> shift) & 3) as u8);
    }
    if order >= 28 {
        tuples.push(((suffix >> 4) & 3) as u8); // order-28 tuple
    }
    if order == 29 {
        tuples.push(((suffix >> 2) & 3) as u8); // order-29 tuple
    }

    Ok(DecimalMorton {
        base_cell,
        order,
        tuples,
    })
}

/// Coarsen a packed word to order `k`, discarding finer detail.
///
/// The base cell and the first `k` tuples are kept; every lower bit is
/// zero-filled and the suffix is rewritten for order `k`. Returns the input
/// unchanged when `k >=` the word's own order (nothing to coarsen). Returns
/// `None` if the word does not decode.
pub fn coarsen(word: u64, k: u8) -> Option<u64> {
    let native = order_of(word)?;
    base_cell_of(word)?; // reject empty / invalid prefix
    if k >= native {
        return Some(word);
    }

    let prefix = word & (0x0fu64 << PREFIX_SHIFT);
    // Keep the body tuples for orders 1..=min(k, 27); mask off everything below.
    let kept_body_orders = k.min(BODY_TUPLES);
    let body = if kept_body_orders == 0 {
        0
    } else {
        // Highest `2 * kept_body_orders` bits of the body region.
        let keep_bits = 2 * kept_body_orders as u32;
        let mask = (((1u64 << keep_bits) - 1) << (54 - keep_bits)) << SUFFIX_BITS;
        word & mask & BODY_MASK
    };

    // Build the target suffix. Orders 28/29 keep their tuples, which live in the
    // *source* suffix at the same bit positions (k < native means native == 29
    // when k == 28, so the order-28 tuple is present to preserve). Lower targets
    // use the variable-length form.
    let src_suffix = word & SUFFIX_MASK;
    let suffix = match k {
        28 => build_suffix(28, ((src_suffix >> 4) & 3) as u8, 0),
        // k == 29 implies k >= native, already returned above; unreachable here.
        _ => build_suffix(k, 0, 0),
    };
    Some(prefix | body | suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic tuple vector (stored 0..=3) for property coverage.
    fn sample_tuples(order: u8, seed: u64) -> Vec<u8> {
        (0..order.max(1))
            .map(|i| ((seed.wrapping_mul(2654435761).wrapping_add(i as u64)) % 4) as u8)
            .collect()
    }

    #[test]
    fn roundtrip_all_base_cells_all_orders() {
        for base in 0..=11u8 {
            for order in 0..=MAX_ORDER {
                let tuples = sample_tuples(order, base as u64 + 1);
                let word = encode(base, &tuples, order);
                let dec = decode(word).expect("decode");
                assert_eq!(dec.base_cell, base, "base mismatch order {}", order);
                assert_eq!(dec.order, order, "order mismatch base {}", base);
                assert_eq!(
                    &dec.tuples[..],
                    &tuples[..order as usize],
                    "tuple mismatch base {} order {}",
                    base,
                    order
                );
            }
        }
    }

    #[test]
    fn order_zero_is_base_cell_only() {
        for base in 0..=11u8 {
            let word = encode(base, &[], 0);
            // body and suffix are both zero for order 0
            assert_eq!(word & BODY_MASK, 0, "body not zero-filled, base {}", base);
            assert_eq!(word & SUFFIX_MASK, 0, "suffix not zero, base {}", base);
            assert_eq!(order_of(word), Some(0));
            assert_eq!(base_cell_of(word), Some(base));
            let dec = decode(word).unwrap();
            assert!(dec.tuples.is_empty());
        }
    }

    #[test]
    fn variable_length_orders_1_through_27() {
        let base = 5u8;
        for order in 1..=BODY_TUPLES {
            let tuples = sample_tuples(order, 7);
            let word = encode(base, &tuples, order);
            // low suffix bit must be 0 (variable length) and count == order
            assert_eq!(word & 1, 0, "var-length flag wrong at order {}", order);
            assert_eq!(order_of(word), Some(order));
            assert_eq!(decode(word).unwrap().tuples, &tuples[..order as usize]);
        }
    }

    #[test]
    fn order_28_suffix_form() {
        let base = 3u8;
        let mut tuples = sample_tuples(28, 11);
        for t28 in 0..=3u8 {
            tuples[27] = t28;
            let word = encode(base, &tuples, 28);
            assert_eq!(word & 0b11, 0b01, "order-28 flag wrong for tuple {}", t28);
            // spare bits 3..2 must be zero-filled
            assert_eq!((word >> 2) & 0b11, 0, "spare bits set for tuple {}", t28);
            assert_eq!((word >> 4) & 0b11, t28 as u64, "order-28 tuple wrong");
            assert_eq!(order_of(word), Some(28));
            let dec = decode(word).unwrap();
            assert_eq!(dec.order, 28);
            assert_eq!(dec.tuples[27], t28);
        }
    }

    #[test]
    fn order_29_suffix_form() {
        let base = 9u8;
        let mut tuples = sample_tuples(29, 13);
        for t28 in 0..=3u8 {
            for t29 in 0..=3u8 {
                tuples[27] = t28;
                tuples[28] = t29;
                let word = encode(base, &tuples, 29);
                assert_eq!(word & 0b11, 0b11, "order-29 flag wrong");
                assert_eq!((word >> 4) & 0b11, t28 as u64, "order-28 tuple wrong");
                assert_eq!((word >> 2) & 0b11, t29 as u64, "order-29 tuple wrong");
                assert_eq!(order_of(word), Some(29));
                let dec = decode(word).unwrap();
                assert_eq!(dec.order, 29);
                assert_eq!(dec.tuples[27], t28);
                assert_eq!(dec.tuples[28], t29);
            }
        }
    }

    #[test]
    fn prefix_stores_base_plus_one() {
        for base in 0..=11u8 {
            let word = encode(base, &[], 0);
            assert_eq!((word >> PREFIX_SHIFT) & 0x0f, (base + 1) as u64);
        }
    }

    #[test]
    fn empty_sentinel_decodes_to_error() {
        assert_eq!(decode(0), Err(DecodeError::Empty));
        assert_eq!(base_cell_of(0), None);
    }

    #[test]
    fn invalid_prefix_rejected() {
        for bad in 13..=15u64 {
            let word = bad << PREFIX_SHIFT;
            assert_eq!(decode(word), Err(DecodeError::InvalidPrefix(bad as u8)));
            assert_eq!(base_cell_of(word), None);
        }
    }

    #[test]
    fn invalid_variable_length_counts_rejected() {
        // counts 28..=31 with low bit 0 are nonsensical
        for bad_count in 28..=31u64 {
            let suffix = bad_count << 1; // low bit 0 => variable length
            let word = (1u64 << PREFIX_SHIFT) | suffix;
            assert_eq!(
                order_of(word),
                None,
                "count {} should be invalid",
                bad_count
            );
            assert_eq!(
                decode(word),
                Err(DecodeError::InvalidSuffixCount(bad_count as u8))
            );
        }
    }

    #[test]
    fn canonical_zero_fill_dedups() {
        // Encode reads only the first `order` tuples, so trailing garbage in the
        // input must not change the canonical word: two inputs for the same
        // order-3 cell produce bit-identical results.
        let base = 2u8;
        let canonical = encode(base, &[1, 2, 3], 3);
        let from_noisy = encode(base, &[1, 2, 3, 3, 1, 0, 2], 3);
        assert_eq!(canonical, from_noisy);
        // Every body bit below order 3 is zero.
        let below_mask = ((1u64 << (2 * (BODY_TUPLES - 3) as u32)) - 1) << SUFFIX_BITS;
        assert_eq!(canonical & below_mask, 0);
    }

    #[test]
    fn raw_sort_is_zorder() {
        // A parent (lower order) sorts immediately before its descendants, and
        // siblings sort by tuple value -- the Z-order locality property.
        let base = 4u8;
        let parent = encode(base, &[0, 0], 2); // order-2 cell, path 0->0
        let children: Vec<u64> = (0..4u8).map(|c| encode(base, &[0, 0, c], 3)).collect();
        for &child in &children {
            assert!(parent < child, "parent {parent} !< child {child}");
        }
        let sorted = {
            let mut c = children.clone();
            c.sort_unstable();
            c
        };
        assert_eq!(children, sorted, "children not already Z-sorted");
        // A higher base cell sorts after everything in base 4.
        let other_base = encode(base + 1, &[0, 0, 0], 3);
        assert!(other_base > parent);
        assert!(other_base > *children.last().unwrap());
    }

    #[test]
    fn coarsen_matches_reencode() {
        let base = 7u8;
        let tuples = sample_tuples(29, 99);
        let fine = encode(base, &tuples, 29);
        for k in 0..=29u8 {
            let coarsened = coarsen(fine, k).expect("coarsen");
            let expected = encode(base, &tuples, k);
            assert_eq!(
                coarsened, expected,
                "coarsen to {} != re-encode at {}",
                k, k
            );
        }
    }

    #[test]
    fn coarsen_noop_when_k_ge_order() {
        let base = 1u8;
        let word = encode(base, &[2, 1, 3], 3);
        assert_eq!(coarsen(word, 3), Some(word));
        assert_eq!(coarsen(word, 5), Some(word));
        assert_eq!(coarsen(word, 29), Some(word));
    }

    #[test]
    fn coarsen_rejects_empty() {
        assert_eq!(coarsen(0, 0), None);
    }

    #[test]
    fn order_of_and_base_cell_of_match_decode() {
        for base in [0u8, 5, 11] {
            for order in [0u8, 1, 13, 27, 28, 29] {
                let tuples = sample_tuples(order, base as u64);
                let word = encode(base, &tuples, order);
                let dec = decode(word).unwrap();
                assert_eq!(order_of(word), Some(dec.order));
                assert_eq!(base_cell_of(word), Some(dec.base_cell));
            }
        }
    }
}
