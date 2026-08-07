//! Batch polygon coverage: one Python↔Rust crossing, rayon across polygons
//! (issue #153).
//!
//! The scalar MOC entry points already release the GIL and parallelize the
//! descent internally, but a per-polygon Python loop pays the boundary cost —
//! argument conversion, allocation, result wrapping — once per polygon, which
//! dominates wall time on catalog-scale batches (~556k footprints).  This
//! module crosses the boundary once: ragged input in arrow list layout
//! (`polygon i` is `lats[offsets[i]..offsets[i+1]]`), `par_iter` over the
//! polygons around the existing scalar kernels, ragged output in the same
//! layout.  Identity-preserving by construction: result `i` ↔ input polygon
//! `i` (unlike [`super::multipolygon_to_morton_moc`], which unions rings).
//!
//! # Error posture
//!
//! Fail-fast with the offending polygon named: structural problems (bad
//! offsets, short rings, non-finite coordinates) are rejected in a serial
//! pre-validation pass, and kernel panics are caught per polygon.  In both
//! regimes the error reported is the **lowest-index** offending polygon —
//! pre-validation scans in index order, and the parallel pass materializes
//! every per-polygon result before scanning for the first failure — so the
//! error is deterministic regardless of rayon's schedule.

use rayon::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::polygon_to_morton_moc;

/// Validate the ragged layout, returning the polygon count.
///
/// Checks, in order: `order` range, `lats`/`lons` length agreement, a
/// non-empty `offsets`, then per polygon (lowest index first) offset
/// monotonicity and bounds, the 3-vertex ring minimum, and coordinate
/// finiteness.  Errors name the offending polygon.
fn validate_batch(lats: &[f64], lons: &[f64], offsets: &[i64], order: u8) -> Result<usize, String> {
    if !(1..=29).contains(&order) {
        return Err("Order must be between 1 and 29".to_string());
    }
    if lats.len() != lons.len() {
        return Err("lats and lons must have the same length".to_string());
    }
    if offsets.is_empty() {
        return Err("offsets must have at least one element".to_string());
    }
    if offsets[0] < 0 {
        return Err(format!("offsets must be non-negative, got {}", offsets[0]));
    }
    let n_polys = offsets.len() - 1;
    for i in 0..n_polys {
        let (s, e) = (offsets[i], offsets[i + 1]);
        if e < s {
            return Err(format!(
                "polygon {i}: offsets must be monotonically non-decreasing \
                 ({e} < {s})"
            ));
        }
        if e as usize > lats.len() {
            return Err(format!(
                "polygon {i}: offset {e} exceeds vertex array length {}",
                lats.len()
            ));
        }
        if e - s < 3 {
            return Err(format!(
                "polygon {i}: needs at least 3 vertices, got {}",
                e - s
            ));
        }
        let (s, e) = (s as usize, e as usize);
        if lats[s..e].iter().any(|v| !v.is_finite()) || lons[s..e].iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "polygon {i}: lats and lons must not contain NaN or infinity"
            ));
        }
    }
    Ok(n_polys)
}

/// Concatenate per-polygon covers into `(values, offsets)` arrow list layout,
/// or surface the lowest-index error.
fn assemble(covers: Vec<Result<Vec<u64>, String>>) -> Result<(Vec<u64>, Vec<i64>), String> {
    // Scanning in index order makes the surfaced error the lowest-index
    // failure — deterministic under any rayon schedule.
    let covers: Vec<Vec<u64>> = covers.into_iter().collect::<Result<_, _>>()?;
    let total: usize = covers.iter().map(Vec::len).sum();
    let mut values = Vec::with_capacity(total);
    let mut out_offsets = Vec::with_capacity(covers.len() + 1);
    out_offsets.push(0i64);
    for cover in &covers {
        values.extend_from_slice(cover);
        out_offsets.push(values.len() as i64);
    }
    Ok((values, out_offsets))
}

/// MOC coverage of many independent polygons in one call.
///
/// Plural *MOCs*: one MOC per input polygon (many→many), against the
/// many→one union of [`super::multipolygon_to_morton_moc`].
///
/// `polygon i` is the ring `lats[offsets[i]..offsets[i+1]]` /
/// `lons[offsets[i]..offsets[i+1]]` (arrow list layout; `offsets[0]` need not
/// be 0, so a sliced arrow array's offsets pass straight through).  Returns
/// `(values, out_offsets)` in the same layout: polygon `i`'s compact MOC is
/// `values[out_offsets[i]..out_offsets[i+1]]`, each identical to what
/// [`polygon_to_morton_moc`] returns for that ring alone.
///
/// # Errors
/// The lowest-index offending polygon, named in the message (see the module
/// docs for the determinism argument).
pub fn polygons_to_morton_mocs(
    lats: &[f64],
    lons: &[f64],
    offsets: &[i64],
    order: u8,
    normalize: bool,
) -> Result<(Vec<u64>, Vec<i64>), String> {
    let n_polys = validate_batch(lats, lons, offsets, order)?;
    let covers: Vec<Result<Vec<u64>, String>> = (0..n_polys)
        .into_par_iter()
        .map(|i| {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            catch_unwind(AssertUnwindSafe(|| {
                polygon_to_morton_moc(&lats[s..e], &lons[s..e], order, normalize)
            }))
            .map_err(|e| {
                format!(
                    "polygon {i}: {}",
                    crate::panic_msg(e, "polygon coverage panicked")
                )
            })
        })
        .collect();
    assemble(covers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ragged() -> (Vec<f64>, Vec<f64>, Vec<i64>) {
        // Three rings: a mid-latitude triangle, a quad, a southern triangle.
        let lats = vec![
            20.0, 30.0, 25.0, // triangle
            40.0, 40.0, 50.0, 50.0, // quad
            -60.0, -70.0, -65.0, // southern triangle
        ];
        let lons = vec![
            -120.0, -120.0, -110.0, //
            -125.0, -115.0, -115.0, -125.0, //
            10.0, 10.0, 20.0,
        ];
        (lats, lons, vec![0, 3, 7, 10])
    }

    #[test]
    fn batch_matches_scalar_per_polygon() {
        let (lats, lons, offsets) = ragged();
        let (values, out) = polygons_to_morton_mocs(&lats, &lons, &offsets, 6, true).unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 0);
        assert_eq!(*out.last().unwrap() as usize, values.len());
        for i in 0..3 {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            let scalar = polygon_to_morton_moc(&lats[s..e], &lons[s..e], 6, true);
            assert_eq!(&values[out[i] as usize..out[i + 1] as usize], &scalar[..]);
        }
    }

    #[test]
    fn empty_batch() {
        let (values, out) = polygons_to_morton_mocs(&[], &[], &[0], 6, true).unwrap();
        assert!(values.is_empty());
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn nonzero_start_offset() {
        let (lats, lons, _) = ragged();
        // Only the middle quad, addressed by a sliced-array style offset pair.
        let (values, out) = polygons_to_morton_mocs(&lats, &lons, &[3, 7], 6, true).unwrap();
        let scalar = polygon_to_morton_moc(&lats[3..7], &lons[3..7], 6, true);
        assert_eq!(out, vec![0, scalar.len() as i64]);
        assert_eq!(values, scalar);
    }

    #[test]
    fn errors_name_lowest_index_polygon() {
        let (lats, lons, _) = ragged();
        // Polygon 1 is 2 vertices; polygon 2 also invalid (0 vertices).
        let err = polygons_to_morton_mocs(&lats, &lons, &[0, 3, 5, 5, 10], 6, true).unwrap_err();
        assert!(err.starts_with("polygon 1:"), "{err}");
        // Non-monotone offsets name the polygon that shrinks.
        let err = polygons_to_morton_mocs(&lats, &lons, &[0, 7, 3, 10], 6, true).unwrap_err();
        assert!(err.starts_with("polygon 1:"), "{err}");
        // Out-of-bounds end offset.
        let err = polygons_to_morton_mocs(&lats, &lons, &[0, 3, 99], 6, true).unwrap_err();
        assert!(err.starts_with("polygon 1:"), "{err}");
        // NaN coordinate.
        let mut bad = lats.clone();
        bad[4] = f64::NAN;
        let err = polygons_to_morton_mocs(&bad, &lons, &[0, 3, 7, 10], 6, true).unwrap_err();
        assert!(err.starts_with("polygon 1:"), "{err}");
    }

    #[test]
    fn bad_order_and_layout_rejected() {
        let (lats, lons, offsets) = ragged();
        assert!(polygons_to_morton_mocs(&lats, &lons, &offsets, 0, true).is_err());
        assert!(polygons_to_morton_mocs(&lats, &lons, &offsets, 30, true).is_err());
        assert!(polygons_to_morton_mocs(&lats, &lons[..9], &offsets, 6, true).is_err());
        assert!(polygons_to_morton_mocs(&lats, &lons, &[], 6, true).is_err());
        assert!(polygons_to_morton_mocs(&lats, &lons, &[-1, 3], 6, true).is_err());
    }
}
