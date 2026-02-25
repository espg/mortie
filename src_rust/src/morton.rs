//! Core morton encoding functions
//!
//! This module implements the morton encoding algorithm for HEALPix grids.
//! It's a direct port of the numba-accelerated fastNorm2Mort function.

/// Precomputed powers of 10 for orders 0-18
const POWERS_OF_10: [i64; 19] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
    10_000_000_000_000,
    100_000_000_000_000,
    1_000_000_000_000_000,
    10_000_000_000_000_000,
    100_000_000_000_000_000,
    1_000_000_000_000_000_000,
];

/// Precomputed powers of 4 for orders 0-18
const POWERS_OF_4: [i64; 19] = [
    1,
    4,
    16,
    64,
    256,
    1_024,
    4_096,
    16_384,
    65_536,
    262_144,
    1_048_576,
    4_194_304,
    16_777_216,
    67_108_864,
    268_435_456,
    1_073_741_824,
    4_294_967_296,
    17_179_869_184,
    68_719_476_736,
];

/// Convert normalized HEALPix address to morton index
///
/// This is a direct port of the Python fastNorm2Mort function.
///
/// # Arguments
/// * `order` - Tessellation order (1-18)
/// * `normed` - Normalized HEALPix address
/// * `parent` - Parent base cell (0-11)
///
/// # Returns
/// Morton index as i64
///
/// # Panics
/// Panics if order > 18 (would overflow i64)
#[inline]
pub fn fast_norm2mort_scalar(order: i64, normed: i64, parent: i64) -> i64 {
    if order > 18 {
        panic!("Max order is 18 (to output to 64-bit int).");
    }

    let order_usize = order as usize;
    let mut mask = 3 * POWERS_OF_4[order_usize - 1];
    let mut num: i64 = 0;

    // Bit manipulation loop - extract 2 bits at a time
    for i in (1..=order).rev() {
        let i_usize = i as usize;
        let next_bit = (normed & mask) >> ((2 * i) - 2);
        num += (next_bit + 1) * POWERS_OF_10[i_usize - 1];
        mask >>= 2;
    }

    // Parent cell handling - conditional based on parent value
    if parent >= 6 {
        // Southern hemisphere (parents 6-11)
        let mut parents = parent - 11;
        parents *= POWERS_OF_10[order_usize];
        num += parents;
        num = -num;
        num -= 6 * POWERS_OF_10[order_usize];
    } else {
        // Northern hemisphere (parents 0-5)
        let parents = (parent + 1) * POWERS_OF_10[order_usize];
        num += parents;
    }

    num
}

/// Convert a morton index to a HEALPix NESTED cell ID and depth.
///
/// # Arguments
/// * `morton` - Morton index (positive for northern hemisphere, negative for southern)
///
/// # Returns
/// `(nested_cell_id, depth)` where depth is the HEALPix order
///
/// # Panics
/// Panics if morton digits are not in range 1-4
pub fn mort2nested(morton: i64) -> (u64, u8) {
    let abs_val = morton.unsigned_abs();
    if abs_val == 0 {
        panic!("Morton index cannot be zero");
    }

    // Count decimal digits to determine order
    let digit_count = decimal_digit_count(abs_val);
    let order = (digit_count - 1) as u8;
    let order_usize = order as usize;
    let divisor = POWERS_OF_10[order_usize] as u64;

    // Extract first digit and remaining digits
    let first_digit = abs_val / divisor as u64;
    let remaining = abs_val % divisor as u64;

    // Determine parent base cell from first digit and sign
    let parent: u64 = if morton > 0 {
        // Northern hemisphere: first_digit = parent + 1
        first_digit - 1
    } else {
        // Southern hemisphere: first_digit = parent - 5
        first_digit + 5
    };

    // Decode remaining digits (values 1-4) to normalized HEALPix address
    // Each digit d maps to 2-bit value (d-1), packed MSB to LSB
    let mut normed: u64 = 0;
    let mut temp = remaining;
    for i in 1..=order as usize {
        let digit = temp % 10;
        debug_assert!((1..=4).contains(&digit), "Invalid morton digit {}", digit);
        let bits = digit - 1;
        normed |= bits << (2 * (i - 1));
        temp /= 10;
    }

    // nested = parent * nside^2 + normed
    let nside_sq = 1u64 << (2 * order as u32);
    let nested = parent * nside_sq + normed;

    (nested, order)
}

/// Convert a HEALPix NESTED cell ID and depth to a morton index.
///
/// # Arguments
/// * `nested` - HEALPix NESTED cell ID
/// * `depth` - HEALPix depth/order
///
/// # Returns
/// Morton index as i64
pub fn nested2mort(nested: u64, depth: u8) -> i64 {
    let nside_sq = 1u64 << (2 * depth as u32);
    let parent = nested / nside_sq;
    let normed = nested % nside_sq;
    fast_norm2mort_scalar(depth as i64, normed as i64, parent as i64)
}

/// Count the number of decimal digits in a u64 value.
#[inline]
fn decimal_digit_count(val: u64) -> usize {
    if val == 0 {
        return 1;
    }
    // Binary search through POWERS_OF_10
    let mut count = 1;
    let mut threshold = 10u64;
    while val >= threshold {
        count += 1;
        if count >= 19 {
            break;
        }
        threshold *= 10;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_norm2mort_basic() {
        // Test a basic conversion
        let result = fast_norm2mort_scalar(6, 100, 2);
        assert!(result > 0); // Northern hemisphere
    }

    #[test]
    fn test_fast_norm2mort_southern_hemisphere() {
        // Test parent >= 6 (southern hemisphere)
        let result = fast_norm2mort_scalar(6, 100, 8);
        assert!(result < 0); // Should be negative
    }

    #[test]
    fn test_fast_norm2mort_northern_hemisphere() {
        // Test parent < 6 (northern hemisphere)
        let result = fast_norm2mort_scalar(6, 100, 2);
        assert!(result > 0); // Should be positive
    }

    #[test]
    #[should_panic(expected = "Max order is 18")]
    fn test_fast_norm2mort_order_too_large() {
        fast_norm2mort_scalar(19, 100, 2);
    }

    #[test]
    fn test_fast_norm2mort_order_18() {
        // Test maximum order
        let result = fast_norm2mort_scalar(18, 1000, 2);
        assert!(result > 0);
    }

    #[test]
    fn test_fast_norm2mort_deterministic() {
        // Same inputs should produce same output
        let r1 = fast_norm2mort_scalar(12, 500, 3);
        let r2 = fast_norm2mort_scalar(12, 500, 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_fast_norm2mort_all_parents() {
        // Test all parent values 0-11
        for parent in 0..12 {
            let result = fast_norm2mort_scalar(10, 1000, parent);
            if parent >= 6 {
                assert!(result < 0, "Parent {} should give negative result", parent);
            } else {
                assert!(result > 0, "Parent {} should give positive result", parent);
            }
        }
    }

    #[test]
    fn test_powers_of_10() {
        // Verify precomputed powers are correct
        for i in 0..19 {
            assert_eq!(POWERS_OF_10[i], 10_i64.pow(i as u32));
        }
    }

    #[test]
    fn test_powers_of_4() {
        // Verify precomputed powers are correct
        for i in 0..19 {
            assert_eq!(POWERS_OF_4[i], 4_i64.pow(i as u32));
        }
    }

    #[test]
    fn test_mort2nested_roundtrip_northern() {
        // Test roundtrip for all northern hemisphere parents
        for parent in 0..6i64 {
            for normed in [0, 1, 15, 100, 1000] {
                let order = 6i64;
                let morton = fast_norm2mort_scalar(order, normed, parent);
                let (nested, depth) = mort2nested(morton);
                let roundtrip = nested2mort(nested, depth);
                assert_eq!(morton, roundtrip,
                    "Roundtrip failed for parent={}, normed={}: morton={} -> nested={}, depth={} -> {}",
                    parent, normed, morton, nested, depth, roundtrip);
            }
        }
    }

    #[test]
    fn test_mort2nested_roundtrip_southern() {
        // Test roundtrip for all southern hemisphere parents
        for parent in 6..12i64 {
            for normed in [0, 1, 15, 100, 1000] {
                let order = 6i64;
                let morton = fast_norm2mort_scalar(order, normed, parent);
                let (nested, depth) = mort2nested(morton);
                let roundtrip = nested2mort(nested, depth);
                assert_eq!(morton, roundtrip,
                    "Roundtrip failed for parent={}, normed={}: morton={} -> nested={}, depth={} -> {}",
                    parent, normed, morton, nested, depth, roundtrip);
            }
        }
    }

    #[test]
    fn test_mort2nested_nested_value() {
        // For parent=2, normed=0, order=6: nested = 2 * 4096 + 0 = 8192
        let morton = fast_norm2mort_scalar(6, 0, 2);
        let (nested, depth) = mort2nested(morton);
        assert_eq!(depth, 6);
        assert_eq!(nested, 2 * 4096); // parent * nside^2
    }

    #[test]
    fn test_nested2mort_basic() {
        // nested=0 at depth=6 means parent=0, normed=0
        let morton = nested2mort(0, 6);
        let expected = fast_norm2mort_scalar(6, 0, 0);
        assert_eq!(morton, expected);
    }

    #[test]
    fn test_mort2nested_all_orders() {
        // Test roundtrip across multiple orders
        for order in 1..=18u8 {
            let morton = fast_norm2mort_scalar(order as i64, 0, 3);
            let (nested, depth) = mort2nested(morton);
            assert_eq!(depth, order);
            let roundtrip = nested2mort(nested, depth);
            assert_eq!(morton, roundtrip, "Roundtrip failed at order {}", order);
        }
    }

    #[test]
    fn test_decimal_digit_count() {
        assert_eq!(decimal_digit_count(1), 1);
        assert_eq!(decimal_digit_count(9), 1);
        assert_eq!(decimal_digit_count(10), 2);
        assert_eq!(decimal_digit_count(99), 2);
        assert_eq!(decimal_digit_count(100), 3);
        assert_eq!(decimal_digit_count(1_000_000), 7);
    }
}
