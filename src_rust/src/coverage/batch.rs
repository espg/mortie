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
//! # Memory posture
//!
//! Polygons are covered a chunk at a time and each chunk is copied into the
//! ragged output as it lands, so the whole batch's per-polygon covers never
//! coexist with the concatenated result — peak ≈ result + one chunk, against
//! the ~2.5x of a cover-everything-then-concatenate pass.
//!
//! # Error posture
//!
//! Fail-fast with the offending polygon named: structural problems (bad
//! offsets, short rings, non-finite coordinates) are rejected in a serial
//! pre-validation pass, and kernel panics are caught per polygon.  In both
//! regimes the error reported is the **lowest-index** offending polygon —
//! pre-validation scans in index order, and the parallel pass materializes a
//! whole chunk's results, then walks them in index order (chunks themselves
//! in index order), before any is allowed to fail the call — so the error is
//! deterministic regardless of rayon's schedule.

use rayon::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::{polygon_to_morton_moc, polygon_to_morton_moc_budget, polygon_to_morton_moc_tolerance};

/// Polygons covered per parallel chunk.
///
/// The batch runs chunk by chunk so only one chunk's per-polygon `Vec`s are
/// ever resident: covering all polygons up front and *then* concatenating
/// peaks at ~2.5x the returned array, since every cover stays live while the
/// output buffer is filled (review of issue #153).  Large enough that the
/// per-chunk fork-join is noise against the covering work: a sweep of
/// 1024/2048/8192 on 100k order-8 footprints put 2048 at the pre-chunking
/// throughput with the peak still ~1.1x the result.
const CHUNK: usize = 2048;

/// A batch MOC build: ragged `values` / `offsets` (arrow list layout) plus the
/// budget-raise telemetry of the `max_cells` variant.
#[derive(Debug)]
pub struct BatchMocs {
    /// All polygons' MOC words, concatenated.
    pub values: Vec<u64>,
    /// Arrow list offsets into `values`; MOC `i` is
    /// `values[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<i64>,
    /// `(count, first_index, first_effective)` when `max_cells` was below the
    /// representable floor for `count` polygons; `first_index` is the lowest
    /// such polygon index and `first_effective` its raised budget.  `None`
    /// without a budget or when no polygon was raised.
    pub raised: Option<(usize, usize, usize)>,
}

/// Validate the ragged layout, returning the polygon count.
///
/// Checks, in order: `order` range, `lats`/`lons` length agreement, a
/// non-empty `offsets` starting at 0, then per polygon (lowest index first)
/// offset monotonicity and bounds, the 3-vertex ring minimum, and coordinate
/// finiteness, and finally the endpoint requirement `offsets[n_polys] ==
/// lats.len()`.  The offsets must **exactly cover** the vertex arrays — both
/// endpoints are checked and each names which one failed — so stale or
/// misaligned offsets that happen to stay in bounds cannot silently produce a
/// valid-looking wrong covering.  Per-polygon errors name the offending
/// polygon, and are raised ahead of the endpoint check so the lowest-index
/// polygon is still what a caller sees first.
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
    if offsets[0] != 0 {
        return Err(format!("offsets must start at 0, got {}", offsets[0]));
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
    // Exact coverage: the last offset must land on the end of the vertex
    // arrays.  Checked after the per-polygon pass so an out-of-bounds polygon
    // is still reported by index (the lowest-index rule).
    if offsets[n_polys] as usize != lats.len() {
        return Err(format!(
            "offsets must end at the vertex count: offsets[{n_polys}] is {} but \
             lats/lons have {} vertices",
            offsets[n_polys],
            lats.len()
        ));
    }
    Ok(n_polys)
}

impl BatchMocs {
    /// An empty batch result sized for `n_polys` polygons.
    fn new(n_polys: usize) -> Self {
        let mut offsets = Vec::with_capacity(n_polys + 1);
        offsets.push(0i64);
        BatchMocs {
            values: Vec::new(),
            offsets,
            raised: None,
        }
    }

    /// Append one chunk's covers, or surface the lowest-index error.
    ///
    /// `base` is the chunk's first polygon index and `budget` the requested
    /// `max_cells` (if any), folded into the raise telemetry.  The covers are
    /// consumed in index order — each is copied and then **dropped**, so a
    /// chunk's allocations are released as the copy walks it — which also
    /// makes the surfaced error the lowest-index failure, deterministic under
    /// any rayon schedule.
    fn extend_chunk(
        &mut self,
        covers: Vec<Result<(Vec<u64>, usize), String>>,
        base: usize,
        budget: Option<usize>,
    ) -> Result<(), String> {
        for (k, cover) in covers.into_iter().enumerate() {
            let (cover, effective) = cover?;
            self.values.extend_from_slice(&cover);
            self.offsets.push(self.values.len() as i64);
            if budget.is_some_and(|b| effective > b) {
                self.raised = match self.raised {
                    None => Some((1, base + k, effective)),
                    Some((n, i0, e0)) => Some((n + 1, i0, e0)),
                };
            }
        }
        Ok(())
    }

    /// Reserve for the polygons still to come, from the covers seen so far.
    ///
    /// A `Vec` realloc holds the old and new buffers at once, so growing the
    /// output by pure doubling puts a 1.5x-of-result transient in the peak.
    /// Extrapolating the mean cover size (with a margin) makes late reallocs
    /// rare; over-reserving is cheap because untouched pages cost address
    /// space, not resident memory.
    fn reserve_estimate(&mut self, done: usize, remaining: usize) {
        if done == 0 || remaining == 0 {
            return;
        }
        // Mean cover so far, scaled by the polygons left, plus a 1/16 margin.
        // `reserve` is a no-op when the spare capacity already covers it.
        let est = (self.values.len() / done).saturating_mul(remaining) * 17 / 16;
        self.values.reserve(est);
    }
}

/// Run one polygon's kernel, turning a panic into a polygon-named error.
///
/// Defensive: `validate_batch` already screens exactly what the scalar
/// kernels assert on, so no known input reaches the panic arm — which is why
/// the tests drive this with an injected panicking kernel instead, keeping
/// the capture-and-name mechanism itself covered against future kernel
/// changes.
fn run_polygon<F>(i: usize, kernel: F) -> Result<(Vec<u64>, usize), String>
where
    F: FnOnce() -> (Vec<u64>, usize),
{
    catch_unwind(AssertUnwindSafe(kernel)).map_err(|e| {
        format!(
            "polygon {i}: {}",
            crate::panic_msg(e, "polygon coverage panicked")
        )
    })
}

/// MOC coverage of many independent polygons in one call.
///
/// Plural *MOCs*: one MOC per input polygon (many→many), against the
/// many→one union of [`super::multipolygon_to_morton_moc`].
///
/// `polygon i` is the ring `lats[offsets[i]..offsets[i+1]]` /
/// `lons[offsets[i]..offsets[i+1]]` (arrow list layout), one ring per entry.
/// The offsets must **exactly cover** the vertex arrays — `offsets[0] == 0`
/// and `offsets[n] == lats.len() == lons.len()` — so a sliced arrow array is
/// re-based by the caller (the Arrow skin does this) rather than passed
/// through verbatim; anything else is an error naming the failing endpoint.
/// Returns
/// a [`BatchMocs`] in the same layout: polygon `i`'s compact MOC is
/// `values[offsets[i]..offsets[i+1]]`, each identical to what the scalar
/// kernel — [`polygon_to_morton_moc`], or its `_tolerance` / `_budget`
/// variant when `tolerance` / `max_cells` is set — returns for that ring
/// alone.  `tolerance` (radians) and `max_cells` are **single shared
/// settings** applied to every polygon, mutually exclusive; a `max_cells`
/// below some polygon's representable floor is raised per polygon exactly as
/// in the scalar path and reported in [`BatchMocs::raised`].
///
/// # Errors
/// The lowest-index offending polygon, named in the message (see the module
/// docs for the determinism argument), or both stop criteria set.
pub fn polygons_to_morton_mocs(
    lats: &[f64],
    lons: &[f64],
    offsets: &[i64],
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
    normalize: bool,
) -> Result<BatchMocs, String> {
    if tolerance.is_some() && max_cells.is_some() {
        return Err("pass at most one of tolerance / max_cells".to_string());
    }
    let n_polys = validate_batch(lats, lons, offsets, order)?;
    let mut out = BatchMocs::new(n_polys);
    // Chunk the parallel pass so only one chunk's covers are resident, and
    // copy each chunk out (in index order) before the next one is built.
    for base in (0..n_polys).step_by(CHUNK) {
        let end = (base + CHUNK).min(n_polys);
        let covers: Vec<Result<(Vec<u64>, usize), String>> = (base..end)
            .into_par_iter()
            .map(|i| {
                let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
                let (la, lo) = (&lats[s..e], &lons[s..e]);
                run_polygon(i, || {
                    if let Some(tol) = tolerance {
                        (
                            polygon_to_morton_moc_tolerance(la, lo, order, tol, normalize),
                            0,
                        )
                    } else if let Some(budget) = max_cells {
                        polygon_to_morton_moc_budget(la, lo, order, budget, normalize)
                    } else {
                        (polygon_to_morton_moc(la, lo, order, normalize), 0)
                    }
                })
            })
            .collect();
        out.extend_chunk(covers, base, max_cells)?;
        out.reserve_estimate(end, n_polys - end);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact-MOC batch call (no stop criteria) with results unpacked.
    fn batch(lats: &[f64], lons: &[f64], offsets: &[i64], order: u8) -> Result<BatchMocs, String> {
        polygons_to_morton_mocs(lats, lons, offsets, order, None, None, true)
    }

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
        let out = batch(&lats, &lons, &offsets, 6).unwrap();
        assert_eq!(out.offsets.len(), 4);
        assert_eq!(out.offsets[0], 0);
        assert_eq!(*out.offsets.last().unwrap() as usize, out.values.len());
        assert!(out.raised.is_none());
        for i in 0..3 {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            let scalar = polygon_to_morton_moc(&lats[s..e], &lons[s..e], 6, true);
            let got = &out.values[out.offsets[i] as usize..out.offsets[i + 1] as usize];
            assert_eq!(got, &scalar[..]);
        }
    }

    #[test]
    fn tolerance_and_budget_match_scalar_variants() {
        let (lats, lons, offsets) = ragged();
        let tol = 0.02_f64;
        let out =
            polygons_to_morton_mocs(&lats, &lons, &offsets, 8, Some(tol), None, true).unwrap();
        for i in 0..3 {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            let scalar = polygon_to_morton_moc_tolerance(&lats[s..e], &lons[s..e], 8, tol, true);
            let got = &out.values[out.offsets[i] as usize..out.offsets[i + 1] as usize];
            assert_eq!(got, &scalar[..]);
        }
        let out = polygons_to_morton_mocs(&lats, &lons, &offsets, 8, None, Some(64), true).unwrap();
        for i in 0..3 {
            let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
            let (scalar, _) = polygon_to_morton_moc_budget(&lats[s..e], &lons[s..e], 8, 64, true);
            let got = &out.values[out.offsets[i] as usize..out.offsets[i + 1] as usize];
            assert_eq!(got, &scalar[..]);
        }
    }

    #[test]
    fn budget_raise_reports_lowest_index() {
        let (lats, lons, offsets) = ragged();
        // A 1-cell budget is below every polygon's representable floor, so all
        // three are raised; the telemetry names polygon 0 first.
        let out = polygons_to_morton_mocs(&lats, &lons, &offsets, 8, None, Some(1), true).unwrap();
        let (count, first, effective) = out.raised.unwrap();
        assert_eq!(count, 3);
        assert_eq!(first, 0);
        let (_, scalar_eff) = polygon_to_morton_moc_budget(&lats[..3], &lons[..3], 8, 1, true);
        assert_eq!(effective, scalar_eff);
    }

    #[test]
    fn both_stop_criteria_rejected() {
        let (lats, lons, offsets) = ragged();
        let err = polygons_to_morton_mocs(&lats, &lons, &offsets, 8, Some(0.1), Some(64), true)
            .unwrap_err();
        assert!(err.contains("at most one"), "{err}");
    }

    #[test]
    fn empty_batch() {
        let out = batch(&[], &[], &[0], 6).unwrap();
        assert!(out.values.is_empty());
        assert_eq!(out.offsets, vec![0]);
    }

    #[test]
    fn offsets_must_exactly_cover_the_vertices() {
        let (lats, lons, _) = ragged();
        // A sliced-array style offset pair: the core rejects it (the Arrow
        // skin re-bases such inputs before they get here).
        let err = batch(&lats, &lons, &[3, 7], 6).unwrap_err();
        assert!(err.contains("must start at 0"), "{err}");
        // In-bounds but short of the end: trailing unused vertices rejected.
        let err = batch(&lats, &lons, &[0, 3], 6).unwrap_err();
        assert!(err.contains("must end at the vertex count"), "{err}");
        // The exactly-covering spelling of that middle quad is accepted.
        let out = batch(&lats[3..7], &lons[3..7], &[0, 4], 6).unwrap();
        let scalar = polygon_to_morton_moc(&lats[3..7], &lons[3..7], 6, true);
        assert_eq!(out.offsets, vec![0, scalar.len() as i64]);
        assert_eq!(out.values, scalar);
    }

    #[test]
    fn errors_name_lowest_index_polygon() {
        let (lats, lons, _) = ragged();
        // Polygon 1 is 2 vertices; polygon 2 also invalid (0 vertices).
        let err = batch(&lats, &lons, &[0, 3, 5, 5, 10], 6).unwrap_err();
        assert!(err.starts_with("polygon 1:"), "{err}");
        // Non-monotone offsets name the polygon that shrinks.
        let err = batch(&lats, &lons, &[0, 7, 3, 10], 6).unwrap_err();
        assert!(err.starts_with("polygon 1:"), "{err}");
        // Out-of-bounds end offset: the per-polygon bounds check must fire
        // ahead of the batch-level endpoint check, so the polygon is named.
        let err = batch(&lats, &lons, &[0, 3, 99], 6).unwrap_err();
        assert!(err.starts_with("polygon 1:"), "{err}");
        // NaN coordinate.
        let mut bad = lats.clone();
        bad[4] = f64::NAN;
        let err = batch(&bad, &lons, &[0, 3, 7, 10], 6).unwrap_err();
        assert!(err.starts_with("polygon 1:"), "{err}");
    }

    #[test]
    fn kernel_panic_is_caught_and_named_by_lowest_index() {
        // No input reaches the kernels' asserts (validate_batch screens them),
        // so the panic-capture half of the lowest-index guarantee is driven
        // here with injected panicking kernels through the same machinery.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // keep the test output clean
        let covers = vec![
            run_polygon(0, || (vec![7u64], 0)),
            run_polygon(1, || panic!("kernel exploded")),
            run_polygon(2, || panic!("also exploded")),
        ];
        // A chunk that does not start at polygon 0 still names the true index.
        let late = vec![run_polygon(9, || panic!("boom"))];
        std::panic::set_hook(hook);

        assert!(covers[0].is_ok());
        let err = BatchMocs::new(3).extend_chunk(covers, 0, None).unwrap_err();
        assert!(err.starts_with("polygon 1:"), "{err}");
        assert!(err.contains("kernel exploded"), "{err}");
        let err = BatchMocs::new(10).extend_chunk(late, 9, None).unwrap_err();
        assert!(err.starts_with("polygon 9:"), "{err}");
    }

    #[test]
    fn bad_order_and_layout_rejected() {
        let (lats, lons, offsets) = ragged();
        assert!(batch(&lats, &lons, &offsets, 0).is_err());
        assert!(batch(&lats, &lons, &offsets, 30).is_err());
        assert!(batch(&lats, &lons[..9], &offsets, 6).is_err());
        assert!(batch(&lats, &lons, &[], 6).is_err());
        let err = batch(&lats, &lons, &[-1, 3], 6).unwrap_err();
        assert!(err.contains("must start at 0"), "{err}");
    }
}
