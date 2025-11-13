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
}
