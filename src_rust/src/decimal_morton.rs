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
//! * **suffix** -- 6 bits read as **one plain unsigned integer `0..=63`** that is
//!   a preorder numbering of the path tail past tuple 27. Because it is a single
//!   monotone code, a raw unsigned sort of two words sharing the same body sorts
//!   parent-before-children across the whole order range -- the Z-order property
//!   #35 calls paramount, now holding end-to-end (orders 0..=29), not just 0..=27.
//!
//!   ```text
//!     0 ..=27   variable length            (area; order == suffix; just the count)
//!    28 ..=47   order 28 / 29 AREA cells    (real cell; preorder, div/mod 5)
//!    48 ..=63   order 29 POINT, max-encoded (no area claim; (t28,t29) keyed)
//!   ```
//!
//!   * **`0..=27`** -- self-describing tuple count: order is literally the suffix
//!     value (`0` = base-cell-only / order 0). `27` is the order-27 path tail `[]`.
//!   * **`28..=47`** -- the 20 order-28/29 *area* nodes in preorder. With
//!     `t28,t29 in {1..4}` stored `0..3`, `r = (t28-1)*5 + (t29_present ? t29 : 0)`
//!     (`0..=19`) and `suffix = 28 + r`. Each `t28` owns a 5-block: `[t28]`
//!     (order 28) followed by its four `[t28,t29]` order-29 children, so the
//!     order-28 parent sorts before its order-29 children (parent-first preorder).
//!   * **`48..=63`** -- an order-29 *point* cast to maximum resolution: it carries
//!     no area claim (e.g. a raw lat/lon cast to a Morton index). Keyed by the
//!     `(t28,t29) in {1..4}^2` combination: `r2 = (t28-1)*4 + (t29-1)` (`0..=15`),
//!     `suffix = 48 + r2`. A decoded point sets `kind == Kind::Point`. Z-order
//!     note: a point sorts **after** every area cell sharing the same body (it is
//!     the highest suffix range), so a max-encoded point sorts after the area
//!     cells that contain it -- intended (a point is the finest, last thing there).
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
/// Full 54-bit body mask, sitting just above the suffix (test-only since the
/// `coarsen` cleanup dropped its sole non-test use).
#[cfg(test)]
const BODY_MASK: u64 = ((1u64 << 54) - 1) << SUFFIX_BITS;

/// First suffix value of the order-28/29 AREA preorder region.
const AREA_TAIL_BASE: u64 = 28;
/// First suffix value of the order-29 POINT region.
const POINT_BASE: u64 = 48;

/// Whether a decoded index claims spatial area or is a max-encoded point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A real cell with area (variable-length orders 0..=27, or an explicit
    /// order-28/29 area cell).
    Area,
    /// An order-29 value cast to maximum resolution with **no area claim** --
    /// e.g. a raw lat/lon point. Distinct from a real order-29 area cell.
    Point,
}

/// A decoded `decimal_morton` index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecimalMorton {
    /// HEALPix base cell, `0..=11`.
    pub base_cell: u8,
    /// HEALPix order, `0..=29`.
    pub order: u8,
    /// Whether this is an area cell or a max-encoded point (only order-29 words
    /// are ever `Point`).
    pub kind: Kind,
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
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Empty => write!(f, "empty decimal_morton (prefix 0)"),
            DecodeError::InvalidPrefix(p) => {
                write!(f, "invalid base-cell prefix {} (valid 1..=12)", p)
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// Build the 6-bit suffix for an **area** cell at `order` with order-28/29 tuples.
///
/// The suffix is a single monotone preorder code (see the module docs):
/// `0..=27` is the order itself; `28..=47` is the order-28/29 area region.
/// `t28`/`t29` are the *stored* `0..=3` values (read as `1..=4`).
#[inline]
fn build_suffix(order: u8, t28: u8, t29: u8) -> u64 {
    match order {
        // variable length: the order *is* the suffix value (just the count).
        0..=27 => order as u64,
        // order 28: parent node of its t28 block; pos 0 within the 5-block.
        28 => AREA_TAIL_BASE + ((t28 & 3) as u64) * 5,
        // order 29: child pos (t29 read as 1..=4) within the t28 5-block.
        29 => AREA_TAIL_BASE + ((t28 & 3) as u64) * 5 + ((t29 & 3) as u64) + 1,
        _ => unreachable!("order > 29 is rejected before build_suffix"),
    }
}

/// Build the 6-bit suffix for an order-29 **point** (no area claim).
///
/// Keyed by the `(t28,t29)` combination: `r2 = t28*4 + t29` (stored `0..=3`),
/// `suffix = 48 + r2`, landing in `48..=63`.
#[inline]
fn build_point_suffix(t28: u8, t29: u8) -> u64 {
    POINT_BASE + ((t28 & 3) as u64) * 4 + ((t29 & 3) as u64)
}

/// Pack the prefix and body-27 bits for `base_cell` and the first
/// `min(order, 27)` tuples; the suffix is added by the caller.
#[inline]
fn pack_prefix_body(base_cell: u8, tuples: &[u8], order: u8) -> u64 {
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
    prefix | body
}

/// Encode an **area** cell from a base cell, per-order tuples and order.
///
/// `tuples` supplies the stored `0..=3` value for each order `1..=order`; only
/// the first `order` entries are read. Any bit below the element's order is
/// zero-filled, so the result is canonical. The result is always `Kind::Area`;
/// for a max-resolution point with no area claim use [`encode_point`].
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

    let (t28, t29) = match order {
        28 => (tuples[27], 0),
        29 => (tuples[27], tuples[28]),
        _ => (0, 0),
    };
    pack_prefix_body(base_cell, tuples, order) | build_suffix(order, t28, t29)
}

/// Encode an order-29 **point** (max resolution, no area claim).
///
/// Use this when casting a raw location (e.g. a lat/lon) to a Morton index: the
/// result decodes with `kind == Kind::Point` and `order == 29`, distinct from a
/// real order-29 area cell. `tuples` must hold all 29 stored `0..=3` tuples.
///
/// # Panics
/// Panics if `base_cell > 11` or `tuples` has fewer than 29 entries.
pub fn encode_point(base_cell: u8, tuples: &[u8]) -> u64 {
    assert!(
        base_cell <= 11,
        "base_cell must be 0..=11, got {}",
        base_cell
    );
    assert!(
        tuples.len() >= MAX_ORDER as usize,
        "need {} tuples for an order-29 point, got {}",
        MAX_ORDER,
        tuples.len()
    );
    pack_prefix_body(base_cell, tuples, MAX_ORDER) | build_point_suffix(tuples[27], tuples[28])
}

/// Read just the order encoded by a packed word's suffix (the "fast path").
///
/// Every 6-bit suffix value is valid under the monotone encoding, so this is
/// total: `0..=27` -> the order itself; `28..=47` -> order 28 (block parent) or
/// 29 (block child); `48..=63` -> an order-29 point.
#[inline]
pub fn order_of(word: u64) -> u8 {
    let suffix = word & SUFFIX_MASK;
    if suffix <= BODY_TUPLES as u64 {
        suffix as u8 // 0..=27
    } else if suffix < POINT_BASE {
        // area tail: pos 0 in a 5-block is order 28, pos 1..4 is order 29.
        if (suffix - AREA_TAIL_BASE).is_multiple_of(5) {
            28
        } else {
            29
        }
    } else {
        29 // point region
    }
}

/// Read whether a packed word is an area cell or a max-encoded point.
#[inline]
pub fn kind_of(word: u64) -> Kind {
    if (word & SUFFIX_MASK) >= POINT_BASE {
        Kind::Point
    } else {
        Kind::Area
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

/// Decode the order-28/29 tail (`t28`, `t29`, `kind`) carried by a suffix.
///
/// Returns the stored `0..=3` tuples for orders 28/29 (`t29 == None` at order
/// 28), and whether the suffix is an area cell or a max-encoded point. Suffix
/// values `0..=27` have no tail and are handled by the caller.
#[inline]
fn decode_tail(suffix: u64) -> (u8, Option<u8>, Kind) {
    if suffix >= POINT_BASE {
        // point: r2 = (t28)*4 + t29, both stored 0..=3.
        let r2 = suffix - POINT_BASE;
        ((r2 / 4) as u8, Some((r2 % 4) as u8), Kind::Point)
    } else {
        // area tail: r = t28*5 + (pos), pos 0 => order 28, 1..=4 => order 29.
        let r = suffix - AREA_TAIL_BASE;
        let t28 = (r / 5) as u8;
        let pos = r % 5;
        if pos == 0 {
            (t28, None, Kind::Area)
        } else {
            (t28, Some((pos - 1) as u8), Kind::Area)
        }
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

    let order = order_of(word);
    let mut tuples = Vec::with_capacity(order as usize);
    let body_orders = order.min(BODY_TUPLES);
    for n in 1..=body_orders {
        let shift = SUFFIX_BITS + 2 * (BODY_TUPLES - n) as u32;
        tuples.push(((word >> shift) & 3) as u8);
    }

    let kind = if order >= 28 {
        let (t28, t29, kind) = decode_tail(word & SUFFIX_MASK);
        tuples.push(t28);
        if let Some(t29) = t29 {
            tuples.push(t29);
        }
        kind
    } else {
        Kind::Area
    };

    Ok(DecimalMorton {
        base_cell,
        order,
        kind,
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
    base_cell_of(word)?; // reject empty / invalid prefix
    let native = order_of(word);
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
        word & mask
    };

    // Build the target suffix. Coarsening to order 28 preserves the order-28
    // tuple, recovered from the source tail (k < native here means native is 29,
    // so the tail is present). The result is always an area cell. Lower targets
    // use the variable-length form.
    let suffix = if k == 28 {
        let (t28, _, _) = decode_tail(word & SUFFIX_MASK);
        build_suffix(28, t28, 0)
    } else {
        // k == 29 implies k >= native, already returned above; unreachable here.
        build_suffix(k, 0, 0)
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
                assert_eq!(dec.kind, Kind::Area, "area-encode must decode Area");
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
            assert_eq!(order_of(word), 0);
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
            // suffix is literally the order count in the variable-length region.
            assert_eq!(
                word & SUFFIX_MASK,
                order as u64,
                "suffix != order {}",
                order
            );
            assert_eq!(order_of(word), order);
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
            // order-28 node sits at the head of its t28 5-block: 28 + 5*t28.
            assert_eq!(word & SUFFIX_MASK, 28 + 5 * t28 as u64, "suffix wrong");
            assert_eq!(order_of(word), 28);
            let dec = decode(word).unwrap();
            assert_eq!(dec.order, 28);
            assert_eq!(dec.kind, Kind::Area);
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
                // child pos t29+1 within the t28 5-block.
                assert_eq!(
                    word & SUFFIX_MASK,
                    28 + 5 * t28 as u64 + t29 as u64 + 1,
                    "suffix wrong"
                );
                assert_eq!(order_of(word), 29);
                let dec = decode(word).unwrap();
                assert_eq!(dec.order, 29);
                assert_eq!(dec.kind, Kind::Area);
                assert_eq!(dec.tuples[27], t28);
                assert_eq!(dec.tuples[28], t29);
            }
        }
    }

    #[test]
    fn point_round_trip_and_flag() {
        // A max-encoded point keeps all 29 tuples, decodes as Kind::Point at
        // order 29, and is distinct from the area cell with the same path.
        let base = 6u8;
        let mut tuples = sample_tuples(29, 21);
        for t28 in 0..=3u8 {
            for t29 in 0..=3u8 {
                tuples[27] = t28;
                tuples[28] = t29;
                let point = encode_point(base, &tuples);
                // point region: 48 + 4*t28 + t29.
                assert_eq!(
                    point & SUFFIX_MASK,
                    48 + 4 * t28 as u64 + t29 as u64,
                    "point suffix wrong"
                );
                assert_eq!(order_of(point), 29);
                assert_eq!(kind_of(point), Kind::Point);
                let dec = decode(point).unwrap();
                assert_eq!(dec.order, 29);
                assert_eq!(dec.kind, Kind::Point);
                assert_eq!(dec.tuples[27], t28);
                assert_eq!(dec.tuples[28], t29);
                // The order-29 *area* cell with the same path is a different word.
                let area = encode(base, &tuples, 29);
                assert_ne!(point, area, "point must differ from area cell");
                assert_eq!(kind_of(area), Kind::Area);
            }
        }
    }

    #[test]
    fn point_sorts_after_area_of_same_body() {
        // A max-encoded point sorts after every area cell sharing the same body
        // tuples 1..27 (it is the highest suffix range -- the finest, last thing
        // at that location).
        let base = 2u8;
        let mut tuples = sample_tuples(29, 55);
        // Fix body 1..27; vary only the 28/29 tail.
        let point = encode_point(base, &tuples);
        for t28 in 0..=3u8 {
            for t29 in 0..=3u8 {
                tuples[27] = t28;
                tuples[28] = t29;
                let area28 = encode(base, &tuples, 28);
                let area29 = encode(base, &tuples, 29);
                assert!(point > area28, "point !> area28 ({t28})");
                assert!(point > area29, "point !> area29 ({t28},{t29})");
            }
        }
        // ...and after the order-27 ancestor too.
        assert!(point > encode(base, &tuples, 27));
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
    fn every_suffix_value_decodes() {
        // The monotone code uses all 64 suffix values, so any word with a valid
        // prefix decodes (no nonsensical suffixes remain). Orders stay 0..=29 and
        // only suffix >= 48 is a point.
        for suffix in 0..=SUFFIX_MASK {
            let word = (1u64 << PREFIX_SHIFT) | suffix;
            let dec = decode(word).expect("suffix should decode");
            assert!(dec.order <= MAX_ORDER);
            assert_eq!(order_of(word), dec.order);
            let expected_kind = if suffix >= 48 {
                Kind::Point
            } else {
                Kind::Area
            };
            assert_eq!(dec.kind, expected_kind, "kind wrong for suffix {}", suffix);
            assert_eq!(kind_of(word), expected_kind);
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
    fn zorder_across_27_28_29_seam() {
        // Pin the paramount property across the full 0..=29 seam for AREA cells
        // sharing a fixed body 1..27: the order-27 ancestor sorts before every
        // order-28 node, which sorts before its order-29 children, with t28
        // blocks ascending. The all-minimum continuation makes the suffix the
        // sole tiebreaker.
        let base = 4u8;
        let body: Vec<u8> = (0..27u8).map(|i| i % 4).collect();
        let mut full = body.clone();
        full.push(0); // t28 placeholder
        full.push(0); // t29 placeholder

        let mut chain = vec![encode(base, &body, 27)];
        for t28 in 0..=3u8 {
            full[27] = t28;
            chain.push(encode(base, &full, 28)); // [t28]
            for t29 in 0..=3u8 {
                full[28] = t29;
                chain.push(encode(base, &full, 29)); // [t28, t29]
            }
        }
        // 1 (order27) + 4*(1 + 4) = 21 nodes.
        assert_eq!(chain.len(), 21);
        for w in chain.windows(2) {
            assert!(w[0] < w[1], "seam not monotone: {} !< {}", w[0], w[1]);
        }
        // Already strictly ascending => sorting is a no-op.
        let mut sorted = chain.clone();
        sorted.sort_unstable();
        assert_eq!(chain, sorted, "seam chain not already Z-sorted");
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
    fn coarsen_zero_fills_below_target() {
        // Coarsening an order-29 word to k must leave every body bit below order
        // k zero and recover exactly the order-k re-encoding.
        let base = 10u8;
        let tuples = sample_tuples(29, 71);
        let fine = encode(base, &tuples, 29);
        for k in 0..=27u8 {
            let coarsened = coarsen(fine, k).expect("coarsen");
            if k < BODY_TUPLES {
                let below_bits = 2 * (BODY_TUPLES - k) as u32;
                let below_mask = ((1u64 << below_bits) - 1) << SUFFIX_BITS;
                assert_eq!(coarsened & below_mask, 0, "body not zero-filled at k {}", k);
            }
            assert_eq!(coarsened, encode(base, &tuples, k), "k {}", k);
        }
        // 29 -> 28 preserves the order-28 tuple and yields an area cell.
        let to28 = coarsen(fine, 28).expect("coarsen 28");
        let dec = decode(to28).unwrap();
        assert_eq!(dec.order, 28);
        assert_eq!(dec.kind, Kind::Area);
        assert_eq!(dec.tuples[27], tuples[27]);
    }

    #[test]
    fn coarsen_point_to_28_preserves_tuple() {
        // A max-encoded point coarsened to order 28 becomes an area cell that
        // keeps its order-28 tuple.
        let base = 8u8;
        let tuples = sample_tuples(29, 88);
        let point = encode_point(base, &tuples);
        let to28 = coarsen(point, 28).expect("coarsen point");
        let dec = decode(to28).unwrap();
        assert_eq!(dec.order, 28);
        assert_eq!(dec.kind, Kind::Area);
        assert_eq!(dec.tuples[27], tuples[27]);
    }

    #[test]
    fn order_of_and_base_cell_of_match_decode() {
        for base in [0u8, 5, 11] {
            for order in [0u8, 1, 13, 27, 28, 29] {
                let tuples = sample_tuples(order, base as u64);
                let word = encode(base, &tuples, order);
                let dec = decode(word).unwrap();
                assert_eq!(order_of(word), dec.order);
                assert_eq!(base_cell_of(word), Some(dec.base_cell));
            }
        }
    }
}
