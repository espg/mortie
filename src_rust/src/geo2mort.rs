//! Rust-native geo2mort: geographic coordinates → morton index
//!
//! Uses the `healpix` crate for the HEALPix hash, then the existing
//! `fast_norm2mort_scalar` for morton encoding.  The entire pipeline
//! runs in Rust with zero Python/HEALPix calls.

use healpix::coords::Degrees;
use healpix::get;

use crate::morton::fast_norm2mort_scalar;

/// Convert a single (lat, lon) pair to a morton index at the given order.
///
/// # Arguments
/// * `lat` – latitude in **degrees**, must be in [-90, 90]
/// * `lon` – longitude in **degrees**
/// * `order` – HEALPix depth / tessellation order (0–18)
///
/// # Returns
/// Morton index as i64 (positive for northern-hemisphere base cells,
/// negative for southern).
#[inline]
pub fn geo2mort_scalar(lat: f64, lon: f64, order: u8) -> i64 {
    let layer = get(order);
    // Degrees(lon, lat) – healpix crate convention is (lon, lat)
    let nest = layer.hash(Degrees(lon, lat)) as i64;

    // nside² = 4^order = 2^(2*order)
    let nside_sq = 1_i64 << (2 * order as u32);

    // Base cell (parent 0–11) and normalized address within that cell
    let parent = nest / nside_sq;
    let normed = nest - parent * nside_sq;

    fast_norm2mort_scalar(order as i64, normed, parent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo2mort_north_pole() {
        let m = geo2mort_scalar(89.999, 0.0, 6);
        assert!(m > 0, "North pole should give positive morton");
    }

    #[test]
    fn test_geo2mort_south_pole() {
        let m = geo2mort_scalar(-89.999, 0.0, 6);
        assert!(m < 0, "South pole should give negative morton");
    }

    #[test]
    fn test_geo2mort_deterministic() {
        let m1 = geo2mort_scalar(45.0, -122.0, 10);
        let m2 = geo2mort_scalar(45.0, -122.0, 10);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_geo2mort_order_range() {
        // Every valid order should work
        for order in 1..=18u8 {
            let m = geo2mort_scalar(45.0, -122.0, order);
            assert!(m != 0, "Order {} should produce non-zero morton", order);
        }
    }

    #[test]
    fn test_geo2mort_equator() {
        let m = geo2mort_scalar(0.0, 0.0, 8);
        assert!(m != 0);
    }
}
