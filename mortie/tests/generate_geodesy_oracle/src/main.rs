//! Independent authalic-latitude oracle for mortie issue #186 (phase 2).
//!
//! Prints `mortie/tests/data/geodesy_oracle.json` on stdout: both directions
//! of the authalic map at the reference latitudes, computed by the `geodesy`
//! crate — the implementation healpix-geo's `ReferenceEllipsoid` wraps — with
//! the pinned WGS84 constants.  `test_authalic.py::TestGeodesyOracle` asserts
//! mortie's series against the committed values.
//!
//! Offline only: this crate is never part of the mortie build.  Nothing in
//! `Cargo.toml` at the repo root references it, and `maturin` builds only
//! `src_rust/src/lib.rs`, so `geodesy` is not a dependency of the wheel.
//!
//! Regenerating (from the repo root)::
//!
//!     cd mortie/tests/generate_geodesy_oracle
//!     cargo run --release > ../data/geodesy_oracle.json
use geodesy::ellps::{Ellipsoid, Latitudes};

/// Semi-major axis of WGS84, metres.
const A: f64 = 6378137.0;
/// Inverse flattening of WGS84.
const INV_F: f64 = 298.257223563;
/// The `geodesy` release these values were captured from.
const GEODESY_VERSION: &str = "0.15.0";

/// Reference latitudes in degrees, matching the probe set in the JSON.
const LATS: [f64; 20] = [
    -90.0,
    -60.0,
    -45.0,
    -30.0,
    -0.5,
    0.0,
    1e-7,
    0.5,
    5.0,
    15.0,
    30.0,
    35.26438968275,
    44.9999999,
    45.0,
    45.0000001,
    60.0,
    75.0,
    89.0,
    89.9999999,
    90.0,
];

fn rows(pairs: &[(f64, f64)]) -> String {
    pairs
        .iter()
        .map(|(x, y)| format!("  [{x:.17e}, {y:.17e}]"))
        .collect::<Vec<_>>()
        .join(",\n")
}

fn main() {
    let ellps = Ellipsoid::new(A, 1.0 / INV_F);
    let coeffs = ellps.coefficients_for_authalic_latitude_computations();

    let fwd: Vec<(f64, f64)> = LATS
        .iter()
        .map(|&lat| {
            (
                lat,
                ellps
                    .latitude_geographic_to_authalic(lat.to_radians(), &coeffs)
                    .to_degrees(),
            )
        })
        .collect();
    let inv: Vec<(f64, f64)> = LATS
        .iter()
        .map(|&lat| {
            (
                lat,
                ellps
                    .latitude_authalic_to_geographic(lat.to_radians(), &coeffs)
                    .to_degrees(),
            )
        })
        .collect();

    println!("{{");
    println!(
        " \"comment\": \"Independent authalic-latitude oracle for issue #186, \
         generated offline by mortie/tests/generate_geodesy_oracle (see its \
         header for the cargo invocation). Degrees, both directions, printed \
         to 18 significant digits. This is an interop/provenance pin, not the \
         numerical correctness anchor -- that stays authalic_reference.json.\","
    );
    println!(" \"geodesy\": \"{GEODESY_VERSION}\",");
    println!(" \"generator\": \"mortie/tests/generate_geodesy_oracle\",");
    println!(" \"wgs84\": {{\"a\": {A:.1}, \"inv_f\": {INV_F}}},");
    println!(" \"forward_deg\": [\n{}\n ],", rows(&fwd));
    println!(" \"inverse_deg\": [\n{}\n ]", rows(&inv));
    println!("}}");
}
