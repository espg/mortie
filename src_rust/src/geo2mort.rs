//! Rust-native HEALPix operations via the `healpix` crate.
//!
//! Provides `geo2mort` (the full coordinate → morton pipeline) and the
//! four primitive HEALPix functions needed by the Python `_healpix.py`
//! backend: `ang2pix`, `pix2ang`, `boundaries`, `vec2ang`.

use healpix::coords::Degrees;
use healpix::dir::Cardinal;
use healpix::get;
use std::f64::consts::TAU;

use crate::morton::fast_norm2mort_scalar;

// ---------------------------------------------------------------------------
// geo2mort
// ---------------------------------------------------------------------------

/// Convert a single (lat, lon) pair to a morton index at the given order.
#[inline]
pub fn geo2mort_scalar(lat: f64, lon: f64, order: u8) -> i64 {
    let layer = get(order);
    let nest = layer.hash(Degrees(lon, lat)) as i64;
    let nside_sq = 1_i64 << (2 * order as u32);
    let parent = nest / nside_sq;
    let normed = nest - parent * nside_sq;
    fast_norm2mort_scalar(order as i64, normed, parent)
}

// ---------------------------------------------------------------------------
// ang2pix: (lon, lat) in degrees → NESTED pixel index
// ---------------------------------------------------------------------------

#[inline]
pub fn ang2pix_scalar(depth: u8, lon_deg: f64, lat_deg: f64) -> u64 {
    get(depth).hash(Degrees(lon_deg, lat_deg))
}

// ---------------------------------------------------------------------------
// pix2ang: NESTED pixel index → (lon, lat) in degrees
// ---------------------------------------------------------------------------

#[inline]
pub fn pix2ang_scalar(depth: u8, pixel: u64) -> (f64, f64) {
    let c = get(depth).center(pixel);
    (c.lon.to_degrees(), c.lat.to_degrees())
}

// ---------------------------------------------------------------------------
// boundaries: NESTED pixel → 3D unit-vector corners
// Returns [x0..x3], [y0..y3], [z0..z3] with healpy vertex order.
// ---------------------------------------------------------------------------

#[inline]
pub fn boundaries_scalar(depth: u8, pixel: u64) -> [[f64; 4]; 3] {
    let verts = get(depth).vertices(pixel);
    // Roll by 2 to match healpy vertex ordering (same fix as cdshealpix backend)
    let mut xyz = [[0.0f64; 4]; 3];
    for i in 0..4usize {
        let src = (i + 2) % 4;
        let cos_lat = verts[src].lat.cos();
        xyz[0][i] = cos_lat * verts[src].lon.cos();
        xyz[1][i] = cos_lat * verts[src].lon.sin();
        xyz[2][i] = verts[src].lat.sin();
    }
    xyz
}

// ---------------------------------------------------------------------------
// boundaries with step: NESTED pixel → 3D unit-vector boundary with
// configurable resolution via path_along_cell_edge.
// Returns Vec<[f64; 3]> with 4*step points.
// ---------------------------------------------------------------------------

#[inline]
pub fn boundaries_step_scalar(depth: u8, pixel: u64, step: u32) -> Vec<[f64; 3]> {
    let layer = get(depth);
    let pts = layer.path_along_cell_edge(pixel, Cardinal::S, true, step);
    // pts has 4*step entries of LonLat (radians).
    // Roll by 2*step to match healpy vertex ordering (S→E→N→W becomes N→W→S→E).
    let n = pts.len();
    let mut xyz = Vec::with_capacity(n);
    for i in 0..n {
        let src = (i + 2 * step as usize) % n;
        let cos_lat = pts[src].lat.cos();
        xyz.push([
            cos_lat * pts[src].lon.cos(),
            cos_lat * pts[src].lon.sin(),
            pts[src].lat.sin(),
        ]);
    }
    xyz
}

// ---------------------------------------------------------------------------
// vec2ang: 3-D unit vector → (theta, phi) in radians
// theta = colatitude (0 at north pole), phi = longitude [0, 2π)
// ---------------------------------------------------------------------------

#[inline]
pub fn vec2ang_single(x: f64, y: f64, z: f64) -> (f64, f64) {
    // Clamp z for numerical safety (mirrors the Python cdshealpix backend)
    let z_clamped = z.clamp(-1.0, 1.0);
    let theta = z_clamped.acos();
    let mut phi = y.atan2(x);
    if phi < 0.0 {
        phi += TAU;
    }
    (theta, phi)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

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
        for order in 1..=18u8 {
            let m = geo2mort_scalar(45.0, -122.0, order);
            assert!(m != 0, "Order {} should produce non-zero morton", order);
        }
    }

    #[test]
    fn test_ang2pix_roundtrip() {
        // ang2pix → pix2ang should return close to original
        let depth = 12u8;
        let lon = 45.0f64;
        let lat = 30.0f64;
        let pix = ang2pix_scalar(depth, lon, lat);
        let (lon2, lat2) = pix2ang_scalar(depth, pix);
        assert!((lon - lon2).abs() < 0.1, "lon roundtrip failed");
        assert!((lat - lat2).abs() < 0.1, "lat roundtrip failed");
    }

    #[test]
    fn test_boundaries_shape() {
        let b = boundaries_scalar(6, 42);
        // 4 vertices, each should be on the unit sphere
        for i in 0..4 {
            let r2 = b[0][i] * b[0][i] + b[1][i] * b[1][i] + b[2][i] * b[2][i];
            assert!((r2 - 1.0).abs() < 1e-10, "vertex {} not on unit sphere", i);
        }
    }

    #[test]
    fn test_vec2ang_equator() {
        let (theta, phi) = vec2ang_single(1.0, 0.0, 0.0);
        assert!((theta - FRAC_PI_2).abs() < 1e-14, "equator theta");
        assert!(phi.abs() < 1e-14, "prime meridian phi");
    }

    #[test]
    fn test_vec2ang_north_pole() {
        let (theta, _phi) = vec2ang_single(0.0, 0.0, 1.0);
        assert!(theta.abs() < 1e-14, "north pole theta should be 0");
    }

    #[test]
    fn test_vec2ang_south_pole() {
        let (theta, _) = vec2ang_single(0.0, 0.0, -1.0);
        assert!((theta - PI).abs() < 1e-14, "south pole theta should be π");
    }
}
