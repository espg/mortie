//! Batch WKB coverage: many blobs → one MOC each, in one crossing (issue #157).
//!
//! The scalar path pays the Python↔Rust boundary once per blob — argument
//! conversion, a parse, a cover, a result wrap — which is what dominates wall
//! time on a catalog-scale WKB column (~556k ATL03 footprints).  This module
//! covers a whole chunk of blobs in one `par_iter`: parse, validate and cover
//! each blob independently, returning per-blob results the caller assembles
//! into the ragged output.
//!
//! # Why a chunk, and not the whole batch
//!
//! [`super::parse`] needs `&[u8]`, and a `&[u8]` borrowed from a Python
//! `bytes` object is GIL-bound — it cannot cross `py.allow_threads`.  So the
//! caller copies **one chunk** of blobs into a contiguous buffer while it
//! still holds the GIL, releases the GIL, and hands that buffer here.  The
//! copy is therefore bounded by the chunk (a few MB), not by the column
//! (~280 MB for the ATL03 corpus at ~500 B/blob).  `Py<PyBytes>` handles were
//! the alternative and are worse: they would force re-acquiring the GIL for
//! every blob inside the parallel region.
//!
//! # Error posture
//!
//! Every blob's outcome is materialized as a `Result`, in index order, and
//! the caller scans them in that order before letting any fail the call — so
//! the surfaced error is the **lowest-index** offending blob regardless of
//! rayon's schedule, matching [`crate::coverage::batch`].  Panics from the
//! coverage kernel are caught per blob and named the same way.

use rayon::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::{parse, Kind};
use crate::coverage::multipolygon_to_morton_moc;

/// The per-ring `(lats, lons)` arrays [`multipolygon_to_morton_moc`] takes.
type RingArrays = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Split a parsed geometry into the per-ring arrays the kernel takes, with
/// the validation the scalar Python path applies before it calls in.
///
/// Mirrors `mortie.coverage._prep_rings` exactly — the 3-vertex minimum and
/// the finiteness check, with its messages — so a blob refused here is
/// refused with the same words `from_wkb` uses for the same geometry.  The
/// closing vertex is **not** dropped: the multipart kernel keeps it, and
/// `from_wkb` always takes the multipart path (its rings arrive as a list),
/// so dropping it here would break byte parity.
fn rings_to_kernel_arrays(rings: &super::Rings) -> Result<RingArrays, String> {
    let n = rings.offsets.len() - 1;
    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);
    for i in 0..n {
        let (s, e) = (rings.offsets[i] as usize, rings.offsets[i + 1] as usize);
        if e - s < 3 {
            return Err("each ring needs at least 3 vertices".to_string());
        }
        let (la, lo) = (&rings.lats[s..e], &rings.lons[s..e]);
        if la.iter().chain(lo.iter()).any(|v| !v.is_finite()) {
            return Err("lats and lons must not contain NaN or infinity".to_string());
        }
        lats.push(la.to_vec());
        lons.push(lo.to_vec());
    }
    Ok((lats, lons))
}

/// Parse and cover one blob, returning its MOC and the effective cell budget.
///
/// Linear geometry is refused: `linestring_coverage` returns one array *per
/// line*, which has no single-MOC-per-blob spelling, so a MultiLineString
/// cannot ride the ragged output at all.  Cover those with `from_wkb`.
fn cover_blob(
    bytes: &[u8],
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
    normalize: bool,
) -> Result<(Vec<u64>, usize), String> {
    let rings = parse(bytes)?;
    if rings.kind != Kind::Polygonal {
        return Err(
            "linear geometry (LineString / MultiLineString) has no ragged MOC form, \
             since a cover is one array per line; use from_wkb for it"
                .to_string(),
        );
    }
    let (lats, lons) = rings_to_kernel_arrays(&rings)?;
    Ok(multipolygon_to_morton_moc(
        &lats, &lons, order, tolerance, max_cells, normalize,
    ))
}

/// Cover one chunk of blobs in parallel, naming failures by global index.
///
/// `buf` holds the chunk's blobs back to back and `offsets` indexes them
/// (`blob k` is `buf[offsets[k]..offsets[k + 1]]`); `base` is the global index
/// of blob 0 of the chunk, so error messages carry the caller's numbering
/// rather than the chunk's.  Results come back in index order, one per blob.
pub fn cover_chunk(
    buf: &[u8],
    offsets: &[usize],
    base: usize,
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
    normalize: bool,
) -> Vec<Result<(Vec<u64>, usize), String>> {
    (0..offsets.len().saturating_sub(1))
        .into_par_iter()
        .map(|k| {
            let bytes = &buf[offsets[k]..offsets[k + 1]];
            let run = catch_unwind(AssertUnwindSafe(|| {
                cover_blob(bytes, order, tolerance, max_cells, normalize)
            }));
            match run {
                Ok(result) => result,
                Err(e) => Err(crate::panic_msg(e, "WKB coverage panicked")),
            }
            .map_err(|msg| format!("blob {}: {msg}", base + k))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::batch::BatchMocs;

    /// Pack a little-endian Polygon WKB from `(lon, lat)` rings.
    fn polygon(rings: &[Vec<(f64, f64)>]) -> Vec<u8> {
        let mut out = vec![1u8];
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(rings.len() as u32).to_le_bytes());
        for r in rings {
            out.extend_from_slice(&(r.len() as u32).to_le_bytes());
            for (x, y) in r {
                out.extend_from_slice(&x.to_le_bytes());
                out.extend_from_slice(&y.to_le_bytes());
            }
        }
        out
    }

    /// A closed quad at `(lon0, lat0)` spanning one degree.
    fn quad(lon0: f64, lat0: f64) -> Vec<(f64, f64)> {
        vec![
            (lon0, lat0),
            (lon0 + 1.0, lat0),
            (lon0 + 1.0, lat0 + 1.0),
            (lon0, lat0 + 1.0),
            (lon0, lat0),
        ]
    }

    /// Lay blobs out the way the pyfunction does, then cover them.
    fn run(blobs: &[Vec<u8>], order: u8) -> Vec<Result<(Vec<u64>, usize), String>> {
        let mut buf = Vec::new();
        let mut offsets = vec![0usize];
        for b in blobs {
            buf.extend_from_slice(b);
            offsets.push(buf.len());
        }
        cover_chunk(&buf, &offsets, 0, order, None, None, true)
    }

    #[test]
    fn each_blob_matches_the_scalar_kernel() {
        let blobs: Vec<Vec<u8>> = (0..5)
            .map(|i| polygon(&[quad(10.0 * i as f64, -70.0 + 2.0 * i as f64)]))
            .collect();
        let got = run(&blobs, 7);
        for (i, blob) in blobs.iter().enumerate() {
            let rings = parse(blob).unwrap();
            let (la, lo) = rings_to_kernel_arrays(&rings).unwrap();
            let (want, _) = multipolygon_to_morton_moc(&la, &lo, 7, None, None, true);
            assert_eq!(got[i].as_ref().unwrap().0, want, "blob {i}");
        }
    }

    #[test]
    fn holes_and_multipart_go_to_the_one_union() {
        // A quad with a hole is one blob and one MOC — the multipart kernel
        // unions the rings, exactly as `from_wkb` does.
        let hole = vec![
            (0.25, 0.25),
            (0.75, 0.25),
            (0.75, 0.75),
            (0.25, 0.75),
            (0.25, 0.25),
        ];
        let blob = polygon(&[quad(0.0, 0.0), hole]);
        let got = run(std::slice::from_ref(&blob), 8);
        let rings = parse(&blob).unwrap();
        let (la, lo) = rings_to_kernel_arrays(&rings).unwrap();
        let (want, _) = multipolygon_to_morton_moc(&la, &lo, 8, None, None, true);
        assert_eq!(got[0].as_ref().unwrap().0, want);
        // The hole really is carved — densified, the holed cover is the
        // smaller region (its *MOC* is longer, since a hole fragments the
        // compact spelling, so compare covered cells rather than words).
        let solid = run(&[polygon(&[quad(0.0, 0.0)])], 8);
        let covered = |m: &[u64]| crate::moc::to_order(m, 8).len();
        assert!(covered(&got[0].as_ref().unwrap().0) < covered(&solid[0].as_ref().unwrap().0));
    }

    #[test]
    fn errors_are_named_by_global_index() {
        let good = polygon(&[quad(0.0, 0.0)]);
        let mut truncated = good.clone();
        truncated.truncate(9);
        let blobs = vec![good.clone(), truncated, good];
        let got = run(&blobs, 6);
        assert!(got[0].is_ok() && got[2].is_ok());
        let err = got[1].as_ref().unwrap_err();
        assert!(err.starts_with("blob 1:"), "{err}");
        assert!(err.contains("truncated WKB"), "{err}");

        // `base` carries the caller's numbering into a later chunk.
        let mut buf = Vec::new();
        let mut offsets = vec![0usize];
        buf.extend_from_slice(&blobs[1]);
        offsets.push(buf.len());
        let got = cover_chunk(&buf, &offsets, 4096, 6, None, None, true);
        assert!(got[0].as_ref().unwrap_err().starts_with("blob 4096:"),);
    }

    #[test]
    fn linear_geometry_is_refused() {
        let mut line = vec![1u8];
        line.extend_from_slice(&2u32.to_le_bytes());
        line.extend_from_slice(&3u32.to_le_bytes());
        for (x, y) in [(0.0f64, 0.0f64), (1.0, 0.0), (1.0, 1.0)] {
            line.extend_from_slice(&x.to_le_bytes());
            line.extend_from_slice(&y.to_le_bytes());
        }
        let got = run(&[polygon(&[quad(0.0, 0.0)]), line], 6);
        let err = got[1].as_ref().unwrap_err();
        assert!(err.starts_with("blob 1:"), "{err}");
        assert!(err.contains("linear geometry"), "{err}");
    }

    #[test]
    fn ring_validation_matches_the_scalar_messages() {
        // A 2-vertex ring, and a mid-ring NaN: the two refusals the scalar
        // path raises from `_prep_rings`, word for word.  The short ring is
        // spelled *closed*, so it reaches the vertex-count check rather than
        // being turned away by the closure check first.
        let short = polygon(&[vec![(0.0, 0.0), (0.0, 0.0)]]);
        let err = run(&[short], 6)[0].as_ref().unwrap_err().clone();
        assert!(err.contains("each ring needs at least 3 vertices"), "{err}");

        let mut ring = quad(0.0, 0.0);
        ring[1].1 = f64::NAN;
        let err = run(&[polygon(&[ring])], 6)[0].as_ref().unwrap_err().clone();
        assert!(
            err.contains("lats and lons must not contain NaN or infinity"),
            "{err}"
        );
    }

    #[test]
    fn empty_chunk_and_assembly_contract() {
        assert!(cover_chunk(&[], &[0], 0, 6, None, None, true).is_empty());
        // The ragged assembly this feeds keeps the strict offsets contract.
        let blobs = vec![polygon(&[quad(0.0, 0.0)]), polygon(&[quad(5.0, 5.0)])];
        let covers = run(&blobs, 6);
        let mut out = BatchMocs::new(2);
        out.extend_chunk(covers, 0, None).unwrap();
        assert_eq!(out.offsets[0], 0);
        assert_eq!(*out.offsets.last().unwrap() as usize, out.values.len());
        assert_eq!(out.offsets.len(), 3);
    }

    #[test]
    fn budget_is_shared_and_raises_are_reported() {
        let blobs = vec![polygon(&[quad(0.0, 0.0)]), polygon(&[quad(5.0, 5.0)])];
        let mut buf = Vec::new();
        let mut offsets = vec![0usize];
        for b in &blobs {
            buf.extend_from_slice(b);
            offsets.push(buf.len());
        }
        let covers = cover_chunk(&buf, &offsets, 0, 8, None, Some(1), true);
        let mut out = BatchMocs::new(2);
        out.extend_chunk(covers, 0, Some(1)).unwrap();
        let (count, first, _) = out.raised.unwrap();
        assert_eq!((count, first), (2, 0));
    }
}
