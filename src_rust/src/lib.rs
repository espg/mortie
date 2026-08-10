//! Rust-accelerated morton indexing for mortie
//!
//! This module provides Python bindings for fast morton encoding operations,
//! replacing the numba-accelerated functions to eliminate Dask conflicts.

// False positives on the pyo3/numpy `?` bridges, where the "useless" error conversion is load-bearing (issue #108).
#![allow(clippy::useless_conversion)]

pub mod arrow_ffi;
pub mod buffer;
pub mod cell_geom;
pub mod coverage;
pub mod decimal_morton;
pub mod dissolve;
pub mod geo2mort;
pub mod linestring;
pub mod moc;
pub mod morton;
pub mod prefix_trie;
pub mod rank_xy;
pub mod sphere;
pub mod toc;
pub mod wkb;

use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBytes, PyModule};
use rayon::prelude::*;

/// Extract a 1-D `i64` buffer from a scalar-or-array Python object, returning
/// `(values, is_scalar)`.  A Python int, a numpy integer scalar, **or a 0-d numpy
/// array** is treated as a scalar (single-element `Vec`, `is_scalar = true`);
/// otherwise the object is read as a contiguous 1-D `i64` array.  Centralizing
/// this keeps scalar-vs-array detection consistent — in particular a 0-d array
/// always classifies as a scalar, instead of falling through to a 1-D extract
/// that would fail with a cryptic dtype error.
fn extract_i64_input(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<i64>, bool)> {
    if let Ok(v) = obj.extract::<i64>() {
        return Ok((vec![v], true));
    }
    let arr = obj.extract::<PyReadonlyArray1<i64>>()?;
    Ok((arr.to_vec()?, false))
}

/// `f64` counterpart of [`extract_i64_input`]: a Python float, numpy float
/// scalar, or 0-d numpy array is a scalar; otherwise a contiguous 1-D `f64`
/// array.
fn extract_f64_input(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, bool)> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok((vec![v], true));
    }
    let arr = obj.extract::<PyReadonlyArray1<f64>>()?;
    Ok((arr.to_vec()?, false))
}

/// Decode morton indices to HEALPix NESTED cell ids and depths (vectorized).
///
/// # Arguments
/// * `morton_array` - Morton indices (u64 NumPy array)
///
/// # Returns
/// Tuple of two NumPy arrays: (nested cell ids as u64, depths as u8).
///
/// Each morton word is the packed `decimal_morton` word (`u64`, issue #58);
/// decoding is total over valid words (issue #48). The empty sentinel (0) or a
/// word with an invalid base-cell prefix raises a `ValueError`.
#[pyfunction]
fn rust_mort2nested(py: Python<'_>, morton_array: PyReadonlyArray1<u64>) -> PyResult<PyObject> {
    // Borrow the (GIL-held) numpy buffer directly when it is contiguous — the
    // common case from the Python wrappers — instead of copying it into a Vec.
    // This stays GIL-bound (no `allow_threads`), so the borrow is sound.
    let owned;
    let data: &[u64] = match morton_array.as_slice() {
        Ok(s) => s,
        Err(_) => {
            owned = morton_array.to_vec()?;
            &owned
        }
    };

    let result = std::panic::catch_unwind(|| {
        let mut nested = Vec::with_capacity(data.len());
        let mut depths = Vec::with_capacity(data.len());
        for &m in data {
            let (cell, depth) = morton::mort2nested(m);
            nested.push(cell);
            depths.push(depth);
        }
        (nested, depths)
    });

    match result {
        Ok((nested, depths)) => {
            let py_nested = nested.into_pyarray_bound(py).into_any().unbind();
            let py_depths = depths.into_pyarray_bound(py).into_any().unbind();
            let tuple = pyo3::types::PyTuple::new_bound(py, &[py_nested, py_depths]);
            Ok(tuple.to_object(py))
        }
        Err(e) => Err(PyValueError::new_err(panic_msg(e, "mort2nested panicked"))),
    }
}

/// Encode HEALPix NESTED cell ids and depths to morton indices (vectorized).
///
/// # Arguments
/// * `nested_array` - HEALPix NESTED cell ids (u64 NumPy array)
/// * `depth_array` - HEALPix depths/orders (u8 NumPy array), same length
///
/// # Returns
/// Morton indices as a u64 NumPy array.
#[pyfunction]
fn rust_nested2mort(
    py: Python<'_>,
    nested_array: PyReadonlyArray1<u64>,
    depth_array: PyReadonlyArray1<u8>,
) -> PyResult<PyObject> {
    // Borrow the contiguous numpy buffers directly (GIL-held, no copy); fall
    // back to a copy only for the rare non-contiguous input.
    let nested_owned;
    let nested: &[u64] = match nested_array.as_slice() {
        Ok(s) => s,
        Err(_) => {
            nested_owned = nested_array.to_vec()?;
            &nested_owned
        }
    };
    let depths_owned;
    let depths: &[u8] = match depth_array.as_slice() {
        Ok(s) => s,
        Err(_) => {
            depths_owned = depth_array.to_vec()?;
            &depths_owned
        }
    };

    if nested.len() != depths.len() {
        return Err(PyValueError::new_err(
            "nested and depth arrays must have the same length",
        ));
    }

    let result = std::panic::catch_unwind(|| {
        nested
            .iter()
            .zip(depths.iter())
            .map(|(&n, &d)| morton::nested2mort(n, d))
            .collect::<Vec<u64>>()
    });

    match result {
        Ok(morton) => Ok(morton.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(e, "nested2mort panicked"))),
    }
}

/// Build compacted prefix trie over morton indices (Python binding)
///
/// Returns `(nodes, permutation)` where `nodes` is a list of tuples
/// `(characteristic, count, idx_start, idx_len, child_node_ids, depth)` and
/// `permutation` is one flat int64 numpy array of original positions. Each
/// node's membership is the slice `permutation[idx_start : idx_start+idx_len]`
/// — no per-node index list is materialised under the GIL (issue #34 item 8).
#[pyfunction]
#[pyo3(signature = (morton_array, max_depth=None))]
fn split_children_rust(
    py: Python<'_>,
    morton_array: PyReadonlyArray1<u64>,
    max_depth: Option<usize>,
) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    // The prefix trie is a path-domain structure: it branches on the decimal
    // repr, and a point word's terminal 'p' would be miscounted as an extra
    // order digit (order-30 area -> 4x-off cell_area). Points do not live in
    // paths (spec section 2, issue #120), so refuse them loudly here -- the
    // same contract hive_path enforces -- rather than emit a corrupt trie.
    if data
        .iter()
        .any(|&w| decimal_morton::kind_of(w) == decimal_morton::Kind::Point)
    {
        return Err(PyValueError::new_err(
            "split_children is undefined for point ids: points do not live in \
             paths (spec section 2, issue #120); pass area words only",
        ));
    }
    let (flat, perm) = py.allow_threads(|| prefix_trie::split_children_flat(&data, max_depth));

    // Node metadata: (characteristic, count, idx_start, idx_len, child_ids, depth)
    let py_list = pyo3::types::PyList::empty_bound(py);
    for (characteristic, count, idx_start, idx_len, child_ids, depth) in flat {
        let py_child_ids = pyo3::types::PyList::new_bound(py, &child_ids);
        let tuple = pyo3::types::PyTuple::new_bound(
            py,
            &[
                characteristic.to_object(py),
                count.to_object(py),
                idx_start.to_object(py),
                idx_len.to_object(py),
                py_child_ids.to_object(py),
                depth.to_object(py),
            ],
        );
        py_list.append(tuple)?;
    }

    // Single flat permutation buffer as a numpy int64 array.
    let perm_i64: Vec<i64> = perm.into_iter().map(|i| i as i64).collect();
    let py_perm = perm_i64.into_pyarray_bound(py);

    let out = pyo3::types::PyTuple::new_bound(py, &[py_list.to_object(py), py_perm.to_object(py)]);
    Ok(out.to_object(py))
}

/// Convert geographic coordinates to morton indices entirely in Rust
///
/// Uses the `healpix` crate for HEALPix hashing — no Python HEALPix
/// backend needed.
///
/// # Arguments
/// * `lats` - Latitude(s) in degrees (scalar or NumPy array)
/// * `lons` - Longitude(s) in degrees (scalar or NumPy array)
/// * `order` - HEALPix order (default 29)
/// * `points` - When true, encode order-29 `Kind::Point` words instead of area
///   cells (order-29-only; any other `order` raises `ValueError`).
///
/// These low-level binding defaults (`order=29`, `points=false`) are a plain
/// area primitive; the public point-by-default ergonomics live in the
/// `mortie.convert.geo2mort` wrapper, which resolves `order`/`points` and always
/// passes them explicitly here.
#[pyfunction]
#[pyo3(signature = (lats, lons, order=29, points=false))]
fn rust_geo2mort<'py>(
    py: Python<'py>,
    lats: &Bound<'py, PyAny>,
    lons: &Bound<'py, PyAny>,
    order: u8,
    points: bool,
) -> PyResult<PyObject> {
    if order > decimal_morton::MAX_ORDER {
        return Err(PyValueError::new_err(
            "Max order is 29 (the packed-u64 decimal_morton limit).",
        ));
    }
    if points && order != decimal_morton::MAX_ORDER {
        return Err(PyValueError::new_err(
            "points=True encodes an order-29 point; pass order=29 (the default) or omit it",
        ));
    }

    let (lat_arr, lats_is_scalar) = extract_f64_input(lats)?;
    let (lon_arr, lons_is_scalar) = extract_f64_input(lons)?;

    // Both scalars → return scalar
    if lats_is_scalar && lons_is_scalar {
        let result = geo2mort::geo2mort_word(lat_arr[0], lon_arr[0], order, points);
        return Ok(result.to_object(py));
    }

    let max_len = lat_arr.len().max(lon_arr.len());

    if (lat_arr.len() != 1 && lat_arr.len() != max_len)
        || (lon_arr.len() != 1 && lon_arr.len() != max_len)
    {
        return Err(PyValueError::new_err(
            "lats and lons must have the same length",
        ));
    }

    let lat_bcast = lat_arr.len() == 1;
    let lon_bcast = lon_arr.len() == 1;
    let results: Vec<u64> = py.allow_threads(|| {
        (0..max_len)
            .into_par_iter()
            .map(|i| {
                let lat = lat_arr[if lat_bcast { 0 } else { i }];
                let lon = lon_arr[if lon_bcast { 0 } else { i }];
                geo2mort::geo2mort_word(lat, lon, order, points)
            })
            .collect()
    });

    Ok(results.into_pyarray_bound(py).into_any().unbind())
}

// ---------------------------------------------------------------------------
// HEALPix backend functions (ang2pix, pix2ang, boundaries, vec2ang)
// ---------------------------------------------------------------------------

/// Convert (lon, lat) in degrees to NESTED pixel index.
///
/// # Arguments
/// * `depth` - HEALPix depth/order
/// * `lon` - Longitude(s) in degrees (scalar or array)
/// * `lat` - Latitude(s) in degrees (scalar or array)
#[pyfunction]
fn rust_ang2pix<'py>(
    py: Python<'py>,
    depth: u8,
    lon: &Bound<'py, PyAny>,
    lat: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let (lon_arr, lon_is_scalar) = extract_f64_input(lon)?;
    let (lat_arr, lat_is_scalar) = extract_f64_input(lat)?;

    if lon_is_scalar && lat_is_scalar {
        let result = geo2mort::ang2pix_scalar(depth, lon_arr[0], lat_arr[0]);
        return Ok((result as i64).to_object(py));
    }

    let max_len = lon_arr.len().max(lat_arr.len());
    if (lon_arr.len() != 1 && lon_arr.len() != max_len)
        || (lat_arr.len() != 1 && lat_arr.len() != max_len)
    {
        return Err(PyValueError::new_err(
            "lon and lat must have the same length",
        ));
    }

    let lon_bcast = lon_arr.len() == 1;
    let lat_bcast = lat_arr.len() == 1;
    let results: Vec<i64> = py.allow_threads(|| {
        (0..max_len)
            .into_par_iter()
            .map(|i| {
                let lo = lon_arr[if lon_bcast { 0 } else { i }];
                let la = lat_arr[if lat_bcast { 0 } else { i }];
                geo2mort::ang2pix_scalar(depth, lo, la) as i64
            })
            .collect()
    });

    Ok(results.into_pyarray_bound(py).into_any().unbind())
}

/// Convert NESTED pixel index to (lon, lat) in degrees.
///
/// # Arguments
/// * `depth` - HEALPix depth/order
/// * `pixel` - Pixel index(es) (scalar or array)
///
/// # Returns
/// Tuple of (lon, lat) as scalars or arrays (degrees)
#[pyfunction]
fn rust_pix2ang<'py>(py: Python<'py>, depth: u8, pixel: &Bound<'py, PyAny>) -> PyResult<PyObject> {
    let (pixel_arr, pixel_is_scalar) = extract_i64_input(pixel)?;

    if pixel_is_scalar {
        let (lon, lat) = geo2mort::pix2ang_scalar(depth, pixel_arr[0] as u64);
        return Ok((lon, lat).to_object(py));
    }

    let n = pixel_arr.len();

    let results: Vec<(f64, f64)> = py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .map(|i| geo2mort::pix2ang_scalar(depth, pixel_arr[i] as u64))
            .collect()
    });

    let mut lons = Vec::with_capacity(n);
    let mut lats = Vec::with_capacity(n);
    for (lo, la) in results {
        lons.push(lo);
        lats.push(la);
    }
    let lon_arr = lons.into_pyarray_bound(py);
    let lat_arr = lats.into_pyarray_bound(py);
    Ok((lon_arr, lat_arr).to_object(py))
}

/// Get boundary vertices of NESTED pixel(s) as 3-D unit vectors.
///
/// # Arguments
/// * `depth` - HEALPix depth/order
/// * `pixel` - Pixel index(es) (scalar or array)
/// * `step` - Points per side (default 1 = 4 corners only; step=32 gives 128 points)
///
/// # Returns
/// For scalar: ndarray shape (3, 4*step)
/// For array of N pixels: ndarray shape (N, 3, 4*step)
#[pyfunction]
#[pyo3(signature = (depth, pixel, step=1))]
fn rust_boundaries<'py>(
    py: Python<'py>,
    depth: u8,
    pixel: &Bound<'py, PyAny>,
    step: u32,
) -> PyResult<PyObject> {
    let (pixel_arr, pixel_is_scalar) = extract_i64_input(pixel)?;
    let ncols = 4 * step as usize;

    if step == 1 {
        // Fast path: original 4-corner code
        if pixel_is_scalar {
            let xyz = geo2mort::boundaries_scalar(depth, pixel_arr[0] as u64);
            let arr = numpy::ndarray::Array2::from_shape_fn((3, 4), |(r, c)| xyz[r][c]);
            return Ok(PyArray2::from_owned_array_bound(py, arr)
                .into_any()
                .unbind());
        }
        let n = pixel_arr.len();
        let results: Vec<[[f64; 4]; 3]> = py.allow_threads(|| {
            (0..n)
                .into_par_iter()
                .map(|i| geo2mort::boundaries_scalar(depth, pixel_arr[i] as u64))
                .collect()
        });
        let mut flat = Vec::with_capacity(n * 3 * 4);
        for xyz in &results {
            for row in xyz {
                for &val in row {
                    flat.push(val);
                }
            }
        }
        let arr = numpy::ndarray::Array3::from_shape_vec((n, 3, 4), flat)
            .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;
        return Ok(PyArray3::from_owned_array_bound(py, arr)
            .into_any()
            .unbind());
    }

    // step > 1: use path_along_cell_edge
    if pixel_is_scalar {
        let pts = geo2mort::boundaries_step_scalar(depth, pixel_arr[0] as u64, step);
        // pts is Vec<[f64; 3]> with ncols entries → shape (3, ncols)
        let arr = numpy::ndarray::Array2::from_shape_fn((3, ncols), |(r, c)| pts[c][r]);
        return Ok(PyArray2::from_owned_array_bound(py, arr)
            .into_any()
            .unbind());
    }

    let n = pixel_arr.len();
    let results: Vec<Vec<[f64; 3]>> = py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .map(|i| geo2mort::boundaries_step_scalar(depth, pixel_arr[i] as u64, step))
            .collect()
    });
    // Shape (N, 3, ncols): transpose each point list (ncols points × 3 coords)
    // into 3 coord-rows of ncols.
    let mut flat = Vec::with_capacity(n * 3 * ncols);
    for pts in &results {
        for r in 0..3 {
            flat.extend(pts.iter().map(|p| p[r]));
        }
    }
    let arr = numpy::ndarray::Array3::from_shape_vec((n, 3, ncols), flat)
        .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;
    Ok(PyArray3::from_owned_array_bound(py, arr)
        .into_any()
        .unbind())
}

/// Convert 3-D unit vectors to (theta, phi) in radians.
///
/// # Arguments
/// * `vectors` - Array of shape (N, 3) containing unit vectors
///
/// # Returns
/// Tuple of (theta, phi) arrays. theta = colatitude (0 at N pole),
/// phi = longitude [0, 2π).
#[pyfunction]
fn rust_vec2ang<'py>(py: Python<'py>, vectors: PyReadonlyArray2<'py, f64>) -> PyResult<PyObject> {
    let shape = vectors.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err("vectors must have shape (N, 3)"));
    }
    let n = shape[0];
    let data = vectors.to_vec()?;

    let results: Vec<(f64, f64)> = py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let x = data[i * 3];
                let y = data[i * 3 + 1];
                let z = data[i * 3 + 2];
                geo2mort::vec2ang_single(x, y, z)
            })
            .collect()
    });

    let mut thetas = Vec::with_capacity(n);
    let mut phis = Vec::with_capacity(n);
    for (t, p) in results {
        thetas.push(t);
        phis.push(p);
    }
    let theta_arr = thetas.into_pyarray_bound(py);
    let phi_arr = phis.into_pyarray_bound(py);
    Ok((theta_arr, phi_arr).to_object(py))
}

/// Symbolic minor-arc crossing test `sphere::arcs_cross_sos`, exported with a
/// plain C ABI for the S2 differential-fixture generator (issue #107 phase 2).
///
/// Not a Python function and not a public-API commitment: the generator
/// (`mortie/tests/generate_s2_crossing_fixtures.py`) reaches it with `ctypes`
/// on the built `mortie._rustie` shared object — the same route it uses for
/// the C++ s2geometry reference — so the committed fixture records both
/// libraries' verdicts from one run.  A C symbol rather than a `#[pyfunction]`
/// because pyo3's argument/return conversion glue references CPython *data*
/// symbols (`Py_True`, `Py_None`, type objects) that the macOS `cargo test`
/// link cannot resolve; this signature is f64/u64/u8 throughout and links
/// everywhere.
///
/// `a..d` point at 3 f64s each (unit vectors); `ia..id` are the
/// pairwise-distinct SoS identities.  Returns `1` for a crossing, else `0`.
///
/// # Safety
/// Each of `a`, `b`, `c`, `d` must be non-null and point at (at least) three
/// readable `f64`s.
///
/// `ia`, `ib`, `ic`, `id` must be **pairwise distinct**: `arcs_cross_sos` is
/// total and reorder-invariant only under that precondition — a duplicated
/// identity makes the symbolic perturbation ill-defined and voids the
/// invariance.  Violating it is not undefined behaviour, so it fails silently
/// with a wrong verdict rather than loudly; a `ctypes` caller reading this
/// block for the contract has no other guard.
#[no_mangle]
pub unsafe extern "C" fn mortie_arcs_cross_sos_ffi(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    d: *const f64,
    ia: u64,
    ib: u64,
    ic: u64,
    id: u64,
) -> u8 {
    let p = |q: *const f64| [*q, *q.add(1), *q.add(2)];
    sphere::arcs_cross_sos(&p(a), &p(b), &p(c), &p(d), ia, ib, ic, id) as u8
}

/// First self-intersecting edge pair of a polygon ring, or an empty array.
///
/// Issue #145: the bucketed transversal-crossing check
/// (`sphere::ring_is_simple`).  Returns a NumPy `uint64` array — empty when
/// the ring is simple, else the two crossing edges' start-vertex indices.
/// Vertex prep matches coverage ingest: a duplicated closing vertex is
/// dropped, so indices refer to the open ring.
///
/// # Arguments
/// * `lats`, `lons` - Vertex coordinates in degrees (NumPy arrays)
#[pyfunction]
fn rust_ring_is_simple(
    py: Python<'_>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let la = lats.to_vec()?;
    let lo = lons.to_vec()?;
    if la.len() != lo.len() {
        return Err(PyValueError::new_err("lats and lons must have same length"));
    }
    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            let mut ring: Vec<sphere::Vec3> = la
                .iter()
                .zip(lo.iter())
                .map(|(&a, &o)| sphere::latlon_to_unit_vec(a, o))
                .collect();
            if ring.len() > 3 {
                let (f, l) = (ring[0], ring[ring.len() - 1]);
                if (f[0] - l[0]).abs() < 1e-12
                    && (f[1] - l[1]).abs() < 1e-12
                    && (f[2] - l[2]).abs() < 1e-12
                {
                    ring.pop();
                }
            }
            match sphere::ring_is_simple(&ring) {
                None => Vec::new(),
                Some((i, j)) => vec![i as u64, j as u64],
            }
        })
    });
    match result {
        Ok(pair) => Ok(pair.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "ring_is_simple panicked",
        ))),
    }
}

/// Both ring-validity verdicts (issue #145, option (b) per espg).
///
/// Returns a NumPy `uint64` array `[crossing, identity_conflict]` of 0/1
/// flags from `sphere::ring_set_validity` over the single ring.  Ring prep
/// matches `rust_ring_is_simple`.
#[pyfunction]
fn rust_ring_validity(
    py: Python<'_>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let la = lats.to_vec()?;
    let lo = lons.to_vec()?;
    if la.len() != lo.len() {
        return Err(PyValueError::new_err("lats and lons must have same length"));
    }
    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            let mut ring: Vec<sphere::Vec3> = la
                .iter()
                .zip(lo.iter())
                .map(|(&a, &o)| sphere::latlon_to_unit_vec(a, o))
                .collect();
            if ring.len() > 3 {
                let (f, l) = (ring[0], ring[ring.len() - 1]);
                if (f[0] - l[0]).abs() < 1e-12
                    && (f[1] - l[1]).abs() < 1e-12
                    && (f[2] - l[2]).abs() < 1e-12
                {
                    ring.pop();
                }
            }
            let v = sphere::ring_set_validity(&[ring]);
            vec![
                u64::from(v.crossing.is_some()),
                u64::from(v.identity_conflict.is_some()),
            ]
        })
    });
    match result {
        Ok(flags) => Ok(flags.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "ring_validity panicked",
        ))),
    }
}

/// Compute the k-cell border around a set of morton indices.
///
/// Returns only cells NOT in the input set (the expansion ring).
/// All input indices must be at the same order.
///
/// # Arguments
/// * `morton_array` - NumPy array of morton indices (u64)
/// * `k` - Border width in cells (default 1, 8-connected neighbors)
///
/// # Returns
/// NumPy array of border morton indices (sorted)
#[pyfunction]
#[pyo3(signature = (morton_array, k=1))]
fn rust_morton_buffer(
    py: Python<'_>,
    morton_array: PyReadonlyArray1<u64>,
    k: u32,
) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;

    let result = py.allow_threads(|| std::panic::catch_unwind(|| buffer::morton_buffer(&data, k)));

    match result {
        Ok(border) => Ok(border.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "morton_buffer panicked",
        ))),
    }
}

/// Compute morton indices that completely cover a polygon.
///
/// # Arguments
/// * `lats` - Vertex latitudes in degrees (NumPy array)
/// * `lons` - Vertex longitudes in degrees (NumPy array)
/// * `order` - HEALPix order/depth (default 18)
/// * `normalize` - auto-correct a sub-hemisphere CW ring to CCW (default true);
///   pass false to trust the supplied vertex order exactly
///
/// # Returns
/// Sorted NumPy array of morton indices (u64)
#[pyfunction]
#[pyo3(signature = (lats, lons, order=18, normalize=true))]
fn rust_polygon_coverage(
    py: Python<'_>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    order: u8,
    normalize: bool,
) -> PyResult<PyObject> {
    let lat_data = lats.to_vec()?;
    let lon_data = lons.to_vec()?;

    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            coverage::polygon_to_morton_coverage(&lat_data, &lon_data, order, normalize)
        })
    });

    match result {
        Ok(cells) => Ok(cells.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "polygon_coverage panicked",
        ))),
    }
}

/// Compute polygon coverage as a compact Multi-Order Coverage map (mixed-order
/// morton indices), with optional adaptive stop criteria.
///
/// # Arguments
/// * `lats`, `lons` - Vertex coordinates in degrees (NumPy arrays)
/// * `order` - finest HEALPix order/depth
/// * `tolerance` - optional: stop refining a boundary cell once its angular
///   radius (radians) drops to this (coarser, approximate boundary)
/// * `max_cells` - optional: best-first budget; refine the largest boundary
///   cells until this many cells, giving an adaptive mixed-order boundary
///
/// `tolerance` and `max_cells` are mutually exclusive; passing neither gives the
/// exact MOC at `order`.
///
/// `normalize` toggles the ingest orientation auto-correction exactly as on
/// `rust_polygon_coverage`; `false` is the escape hatch for expressing a
/// big-side interior as a lone ring (issue #144 decision (A)).
#[pyfunction]
#[pyo3(signature = (lats, lons, order=18, tolerance=None, max_cells=None, normalize=true))]
fn rust_polygon_coverage_moc(
    py: Python<'_>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
    normalize: bool,
) -> PyResult<PyObject> {
    if tolerance.is_some() && max_cells.is_some() {
        return Err(PyValueError::new_err(
            "pass at most one of tolerance / max_cells",
        ));
    }
    let lat_data = lats.to_vec()?;
    let lon_data = lons.to_vec()?;

    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            if let Some(tol) = tolerance {
                (
                    coverage::polygon_to_morton_moc_tolerance(
                        &lat_data, &lon_data, order, tol, normalize,
                    ),
                    None,
                )
            } else if let Some(budget) = max_cells {
                let (cells, effective) = coverage::polygon_to_morton_moc_budget(
                    &lat_data, &lon_data, order, budget, normalize,
                );
                let warn = (effective > budget).then_some((budget, effective));
                (cells, warn)
            } else {
                (
                    coverage::polygon_to_morton_moc(&lat_data, &lon_data, order, normalize),
                    None,
                )
            }
        })
    });

    match result {
        Ok((cells, warn)) => {
            if let Some((requested, effective)) = warn {
                let warnings = py.import_bound("warnings")?;
                warnings.call_method1(
                    "warn",
                    (format!(
                        "max_cells={requested} is below the minimum to represent this \
                         polygon; using {effective}"
                    ),),
                )?;
            }
            Ok(cells.into_pyarray_bound(py).into_any().unbind())
        }
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "polygon_coverage_moc panicked",
        ))),
    }
}

/// Drain the cause-tagged `node_straddles` instrumentation (issue #90).
///
/// Compiled only under the `descent-stats` cargo feature
/// (`maturin develop --release --features descent-stats`).  Take-and-reset:
/// call once to clear, run one descent, call again to read it.  Returns a
/// dict with `causes` (the taxonomy names, indexed by cause id),
/// `leaf_counts` / `internal_counts` (per-cause straddle counters), and the
/// per-leaf table `morton`, `depth`, `cause`, `fill`, `cx`/`cy`/`cz` (cell
/// centre unit vector), `circ` (densified-boundary circumradius, radians).
#[cfg(feature = "descent-stats")]
#[pyfunction]
fn rust_descent_stats_take(py: Python<'_>) -> PyResult<PyObject> {
    use coverage::descent_stats as ds;
    let stats = ds::take();
    let n = stats.leaves.len();
    let mut morton = Vec::with_capacity(n);
    let mut depth = Vec::with_capacity(n);
    let mut cause = Vec::with_capacity(n);
    let mut fill = Vec::with_capacity(n);
    let (mut cx, mut cy, mut cz) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut circ = Vec::with_capacity(n);
    for l in &stats.leaves {
        morton.push(l.morton);
        depth.push(l.depth);
        cause.push(l.cause as u8);
        fill.push(l.fill);
        cx.push(l.center[0]);
        cy.push(l.center[1]);
        cz.push(l.center[2]);
        circ.push(l.circ);
    }
    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item(
        "causes",
        [
            "vertex_leaf",
            "quad_cross",
            "quad_touch",
            "corner_parity",
            "near_pole_bulge",
        ]
        .to_vec(),
    )?;
    dict.set_item("leaf_counts", stats.leaf.to_vec())?;
    dict.set_item("internal_counts", stats.internal.to_vec())?;
    dict.set_item("morton", morton.into_pyarray_bound(py))?;
    dict.set_item("depth", depth.into_pyarray_bound(py))?;
    dict.set_item("cause", cause.into_pyarray_bound(py))?;
    dict.set_item("fill", fill.into_pyarray_bound(py))?;
    dict.set_item("cx", cx.into_pyarray_bound(py))?;
    dict.set_item("cy", cy.into_pyarray_bound(py))?;
    dict.set_item("cz", cz.into_pyarray_bound(py))?;
    dict.set_item("circ", circ.into_pyarray_bound(py))?;
    Ok(dict.into_any().unbind())
}

/// Extract a readable message from a caught panic payload, falling back to
/// `fallback` when the payload is neither a `String` nor a `&str`.
pub(crate) fn panic_msg(e: Box<dyn std::any::Any + Send>, fallback: &str) -> String {
    if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else {
        fallback.to_string()
    }
}

/// Re-raise `err` prefixed with the blob's global index, keeping its type.
///
/// The batch's fail-fast contract names the offending blob, and coercion is
/// the one gate that runs inside the chunk loop rather than in the wrapper's
/// pre-pass; without this an entry that changed underfoot between the two
/// (a `memoryview` released, say) would surface unnumbered.
fn index_error(py: Python<'_>, err: PyErr, index: usize) -> PyErr {
    let msg = format!("blob {index}: {}", err.value_bound(py));
    PyErr::from_type_bound(err.get_type_bound(py).clone(), (msg,))
}

/// Coverage of a ring-set (multipart polygons and/or holes) as a flat list at
/// `order`.  All rings go to one even-odd descent: a point is covered iff it is
/// inside an odd number of rings (so nested rings carve holes, and disjoint
/// parts union with no internal seams).
#[pyfunction]
#[pyo3(signature = (lats, lons, order=18, normalize=true))]
fn rust_multipolygon_coverage(
    py: Python<'_>,
    lats: Vec<PyReadonlyArray1<f64>>,
    lons: Vec<PyReadonlyArray1<f64>>,
    order: u8,
    normalize: bool,
) -> PyResult<PyObject> {
    let la: Vec<Vec<f64>> = lats.iter().map(|a| a.to_vec()).collect::<Result<_, _>>()?;
    let lo: Vec<Vec<f64>> = lons.iter().map(|a| a.to_vec()).collect::<Result<_, _>>()?;
    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            coverage::multipolygon_to_morton_coverage(&la, &lo, order, normalize)
        })
    });
    match result {
        Ok(cells) => Ok(cells.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "multipolygon_coverage panicked",
        ))),
    }
}

/// MOC coverage of a ring-set (multipart / holes) with optional adaptive stop.
/// See `rust_polygon_coverage_moc` for `tolerance` / `max_cells`.
#[pyfunction]
#[pyo3(signature = (lats, lons, order=18, tolerance=None, max_cells=None, normalize=true))]
fn rust_multipolygon_coverage_moc(
    py: Python<'_>,
    lats: Vec<PyReadonlyArray1<f64>>,
    lons: Vec<PyReadonlyArray1<f64>>,
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
    normalize: bool,
) -> PyResult<PyObject> {
    if tolerance.is_some() && max_cells.is_some() {
        return Err(PyValueError::new_err(
            "pass at most one of tolerance / max_cells",
        ));
    }
    let la: Vec<Vec<f64>> = lats.iter().map(|a| a.to_vec()).collect::<Result<_, _>>()?;
    let lo: Vec<Vec<f64>> = lons.iter().map(|a| a.to_vec()).collect::<Result<_, _>>()?;
    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            coverage::multipolygon_to_morton_moc(&la, &lo, order, tolerance, max_cells, normalize)
        })
    });
    match result {
        Ok((cells, effective)) => {
            if let Some(requested) = max_cells {
                if effective > requested {
                    let warnings = py.import_bound("warnings")?;
                    warnings.call_method1(
                        "warn",
                        (format!(
                            "max_cells={requested} is below the minimum to represent this \
                             polygon; using {effective}"
                        ),),
                    )?;
                }
            }
            Ok(cells.into_pyarray_bound(py).into_any().unbind())
        }
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "multipolygon_coverage_moc panicked",
        ))),
    }
}

/// MOC coverage of many independent polygons in one call (issue #153).
///
/// Ragged input in arrow list layout: polygon `i` is
/// `lats[offsets[i]..offsets[i+1]]` / `lons[..]`.  Returns
/// `(values, out_offsets)` in the same layout, each polygon's MOC identical
/// to the scalar `rust_polygon_coverage_moc` output for that ring (including
/// its `tolerance` / `max_cells` variants — both **shared** across the batch
/// and mutually exclusive, `tolerance` in radians).  The GIL is released for
/// the whole batch; rayon parallelizes across polygons.  Errors name the
/// lowest-index offending polygon.
#[pyfunction]
#[pyo3(signature = (lats, lons, offsets, order=18, tolerance=None, max_cells=None, normalize=true))]
#[allow(clippy::too_many_arguments)]
fn rust_polygons_coverage_mocs(
    py: Python<'_>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    offsets: PyReadonlyArray1<i64>,
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
    normalize: bool,
) -> PyResult<(PyObject, PyObject)> {
    let la = lats.to_vec()?;
    let lo = lons.to_vec()?;
    let off = offsets.to_vec()?;

    let result = py.allow_threads(|| {
        coverage::batch::polygons_to_morton_mocs(
            &la, &lo, &off, order, tolerance, max_cells, normalize,
        )
    });

    match result {
        Ok(batch) => {
            if let Some((count, first, effective)) = batch.raised {
                let requested = max_cells.unwrap_or(0);
                let warnings = py.import_bound("warnings")?;
                warnings.call_method1(
                    "warn",
                    (format!(
                        "max_cells={requested} is below the minimum to represent \
                         {count} polygon(s); e.g. polygon {first} uses {effective}"
                    ),),
                )?;
            }
            Ok((
                batch.values.into_pyarray_bound(py).into_any().unbind(),
                batch.offsets.into_pyarray_bound(py).into_any().unbind(),
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

/// Compress a (mixed-order) morton set into its canonical compact MOC: merge
/// any 4 complete sibling cells into their parent, and drop any cell already
/// contained in a coarser one.  Use after unioning per-part covers.
#[pyfunction]
fn rust_moc_normalize(py: Python<'_>, morton: PyReadonlyArray1<u64>) -> PyResult<PyObject> {
    let data = morton.to_vec()?;
    let normalized = py.allow_threads(|| moc::normalize(&data));
    Ok(normalized.into_pyarray_bound(py).into_any().unbind())
}

/// Densify a (mixed-order) morton set to a flat list at `order`.
///
/// `order` above 29 raises `ValueError` — the densify shift is undefined there
/// and used to wrap mod 64 into a `PanicException` (issue #161).
#[pyfunction]
#[pyo3(signature = (morton, order))]
fn rust_moc_to_order(
    py: Python<'_>,
    morton: PyReadonlyArray1<u64>,
    order: u8,
) -> PyResult<PyObject> {
    let data = morton.to_vec()?;
    let densified = py
        .allow_threads(|| moc::to_order(&data, order))
        .map_err(PyValueError::new_err)?;
    Ok(densified.into_pyarray_bound(py).into_any().unbind())
}

/// Exact flat cell count `rust_moc_to_order` would produce at `order`, computed
/// from the compact MOC without materializing the flat list (issue #80).
///
/// Shares `rust_moc_to_order`'s `order` domain and raises the same `ValueError`
/// past it, so no `order` can make the guard's estimate a fabricated one
/// (issue #161).  A malformed *word* is a separate matter — it still panics in
/// `mort2nested`, as it does on every kernel that decodes one.
#[pyfunction]
#[pyo3(signature = (morton, order))]
fn rust_moc_to_order_count(
    py: Python<'_>,
    morton: PyReadonlyArray1<u64>,
    order: u8,
) -> PyResult<u64> {
    let data = morton.to_vec()?;
    py.allow_threads(|| moc::to_order_count(&data, order))
        .map_err(PyValueError::new_err)
}

/// Densify many (mixed-order) MOCs to a flat `order` in one call (issue #156).
///
/// Ragged input in arrow list layout — MOC `i` is
/// `values[offsets[i]..offsets[i+1]]`, exactly the pair
/// `rust_polygons_coverage_mocs` returns.  Gives back `(values, out_offsets)`
/// in the same layout, each MOC's flat list identical to the scalar
/// `rust_moc_to_order` output for that MOC.  `max_cells` is the per-MOC
/// pre-emptive densify budget (issue #80's guard, applied per item): a MOC over
/// budget raises `ValueError` naming the lowest-index offender, before anything
/// is densified.  The GIL is released for the whole batch; rayon parallelizes
/// across MOCs.
#[pyfunction]
#[pyo3(signature = (values, offsets, order, max_cells=None))]
fn rust_mocs_to_orders(
    py: Python<'_>,
    values: PyReadonlyArray1<u64>,
    offsets: PyReadonlyArray1<i64>,
    order: u8,
    max_cells: Option<u64>,
) -> PyResult<(PyObject, PyObject)> {
    let vals = values.to_vec()?;
    let off = offsets.to_vec()?;

    let result = py.allow_threads(|| moc::batch::mocs_to_orders(&vals, &off, order, max_cells));

    match result {
        Ok(batch) => Ok((
            batch.values.into_pyarray_bound(py).into_any().unbind(),
            batch.offsets.into_pyarray_bound(py).into_any().unbind(),
        )),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

/// Union (OR) of two morton covers, backed by the healpix-crate BMOC.
#[pyfunction]
fn rust_moc_or(
    py: Python<'_>,
    a: PyReadonlyArray1<u64>,
    b: PyReadonlyArray1<u64>,
) -> PyResult<PyObject> {
    let (da, db) = (a.to_vec()?, b.to_vec()?);
    let out = py.allow_threads(|| moc::moc_or(&da, &db));
    Ok(out.into_pyarray_bound(py).into_any().unbind())
}

/// Intersection (AND) of two morton covers, backed by the healpix-crate BMOC.
#[pyfunction]
fn rust_moc_and(
    py: Python<'_>,
    a: PyReadonlyArray1<u64>,
    b: PyReadonlyArray1<u64>,
) -> PyResult<PyObject> {
    let (da, db) = (a.to_vec()?, b.to_vec()?);
    let out = py.allow_threads(|| moc::moc_and(&da, &db));
    Ok(out.into_pyarray_bound(py).into_any().unbind())
}

/// Difference (`a \ b`) of two morton covers, backed by the healpix-crate BMOC.
#[pyfunction]
fn rust_moc_minus(
    py: Python<'_>,
    a: PyReadonlyArray1<u64>,
    b: PyReadonlyArray1<u64>,
) -> PyResult<PyObject> {
    let (da, db) = (a.to_vec()?, b.to_vec()?);
    let out = py.allow_threads(|| moc::moc_minus(&da, &db));
    Ok(out.into_pyarray_bound(py).into_any().unbind())
}

/// Symmetric difference (`a △ b`) of two morton covers, backed by the
/// healpix-crate BMOC.
#[pyfunction]
fn rust_moc_xor(
    py: Python<'_>,
    a: PyReadonlyArray1<u64>,
    b: PyReadonlyArray1<u64>,
) -> PyResult<PyObject> {
    let (da, db) = (a.to_vec()?, b.to_vec()?);
    let out = py.allow_threads(|| moc::moc_xor(&da, &db));
    Ok(out.into_pyarray_bound(py).into_any().unbind())
}

/// Whether two morton covers intersect, without materializing the intersection
/// (issue #173).  The predicate twin of `rust_moc_and`: equal to
/// `rust_moc_and(a, b).size > 0`, computed as a range-overlap walk over the
/// normalized covers with an early exit on the first overlap.
#[pyfunction]
fn rust_moc_intersects(
    py: Python<'_>,
    a: PyReadonlyArray1<u64>,
    b: PyReadonlyArray1<u64>,
) -> PyResult<bool> {
    let (da, db) = (a.to_vec()?, b.to_vec()?);
    Ok(py.allow_threads(|| moc::moc_intersects(&da, &db)))
}

/// Intersect one shared morton cover with many ragged MOCs in one call
/// (issue #173).  The 1×N broadcast of `rust_moc_and`: the shared operand's
/// BMOC is built once and borrowed per item, and item `i` of the ragged result
/// is byte-identical to `rust_moc_and(a, values[offsets[i]..offsets[i+1]])`.
/// The GIL is released for the whole batch; rayon parallelizes across MOCs.
#[pyfunction]
fn rust_mocs_and(
    py: Python<'_>,
    a: PyReadonlyArray1<u64>,
    values: PyReadonlyArray1<u64>,
    offsets: PyReadonlyArray1<i64>,
) -> PyResult<(PyObject, PyObject)> {
    let da = a.to_vec()?;
    let vals = values.to_vec()?;
    let off = offsets.to_vec()?;

    let result = py.allow_threads(|| moc::batch::mocs_and(&da, &vals, &off));

    match result {
        Ok(batch) => Ok((
            batch.values.into_pyarray_bound(py).into_any().unbind(),
            batch.offsets.into_pyarray_bound(py).into_any().unbind(),
        )),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

/// Which of many ragged MOCs intersect one shared cover — `bool` per MOC
/// (issue #173).  The batch form of `rust_moc_intersects`: item `i` is exactly
/// "is `rust_mocs_and`'s slot `i` non-empty", computed without materializing
/// any intersection.  The GIL is released for the whole batch; rayon
/// parallelizes across MOCs.
#[pyfunction]
fn rust_mocs_intersect(
    py: Python<'_>,
    a: PyReadonlyArray1<u64>,
    values: PyReadonlyArray1<u64>,
    offsets: PyReadonlyArray1<i64>,
) -> PyResult<PyObject> {
    let da = a.to_vec()?;
    let vals = values.to_vec()?;
    let off = offsets.to_vec()?;

    let result = py.allow_threads(|| moc::batch::mocs_intersect(&da, &vals, &off));

    match result {
        Ok(hits) => Ok(hits.into_pyarray_bound(py).into_any().unbind()),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

/// Deepest common ancestor (`moc_min`) of a morton cover: the highest-order cell
/// that contains every input word, returned as a scalar u64.  Raises
/// `ValueError` on empty input, an empty/invalid word, or inputs spanning more
/// than one base cell (no common ancestor exists).
#[pyfunction]
fn rust_moc_min(py: Python<'_>, morton: PyReadonlyArray1<u64>) -> PyResult<u64> {
    let data = morton.to_vec()?;
    py.allow_threads(|| decimal_morton::common_ancestor(&data))
        .map_err(|e| PyValueError::new_err(format!("moc_min: {}", e)))
}

/// Deepest common ancestor of each of many groups of words, in one call
/// (issue #156).
///
/// Ragged input in arrow list layout — group `i` is
/// `values[offsets[i]..offsets[i+1]]`, the same pair `rust_polygons_coverage_mocs`
/// returns.  The output is **dense**: one `u64` per group, `offsets.len() - 1` of
/// them, each identical to the scalar `rust_moc_min` on that group alone.  Raises
/// `ValueError` naming the lowest-index offending group (a layout problem, or a
/// group that is empty / undecodable / spanning more than one base cell).  The
/// GIL is released for the whole batch; rayon parallelizes across groups.
#[pyfunction]
fn rust_common_ancestors(
    py: Python<'_>,
    values: PyReadonlyArray1<u64>,
    offsets: PyReadonlyArray1<i64>,
) -> PyResult<PyObject> {
    let vals = values.to_vec()?;
    let off = offsets.to_vec()?;

    match py.allow_threads(|| decimal_morton::batch::common_ancestors(&vals, &off)) {
        Ok(words) => Ok(words.into_pyarray_bound(py).into_any().unbind()),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

/// Children of many parent words at a target order, in one call (issue #156).
///
/// The parents must all sit at one order `p <= order`, so every one yields the
/// same `4**(order - p)` children and the result is a **dense** `(n, 4**d)`
/// row-major array — no ragged offsets.  Row `i` is identical to the scalar
/// `generate_morton_children` on `words[i]` alone, the `d == 0` case included
/// (the parent comes back verbatim, preserving a point word's kind).  Raises
/// `ValueError` naming the lowest-index offending word (undecodable, finer than
/// `order`, or at a different order from word 0), or naming the byte count when
/// the `4**d` result is a size the allocator refuses — the block is allocated
/// fallibly so an unservable `order` is catchable rather than an `abort()`.  The
/// GIL is released for the whole batch; rayon parallelizes across parents.
///
/// `max_cells` is an **opt-in** budget on the result's cell count, `None` by
/// default — the opposite default from `rust_mocs_to_orders`' per-MOC budget,
/// because this op's output size is exactly `n * 4**d` and therefore predictable
/// by the caller (see the batch module's allocation posture).  When set, an
/// over-budget result is refused before anything is allocated.
#[pyfunction]
#[pyo3(signature = (words, order, max_cells=None))]
fn rust_children_of(
    py: Python<'_>,
    words: PyReadonlyArray1<u64>,
    order: u8,
    max_cells: Option<u64>,
) -> PyResult<PyObject> {
    let data = words.to_vec()?;
    let n = data.len();

    let children = py
        .allow_threads(|| decimal_morton::batch::children_of(&data, order, max_cells))
        .map_err(PyValueError::new_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((n, children.width), children.values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array_bound(py, arr)
        .into_any()
        .unbind())
}

/// Compute morton indices tracing a linestring (open polyline).
///
/// # Arguments
/// * `lats` - Vertex latitudes in degrees (NumPy array, >=2)
/// * `lons` - Vertex longitudes in degrees (NumPy array, >=2)
/// * `order` - HEALPix order/depth (default 18)
///
/// # Returns
/// Sorted, unique NumPy array of morton indices (u64) tracing the line
/// as a contiguous cell chain at the given order.
#[pyfunction]
#[pyo3(signature = (lats, lons, order=18))]
fn rust_linestring_coverage(
    py: Python<'_>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    order: u8,
) -> PyResult<PyObject> {
    let lat_data = lats.to_vec()?;
    let lon_data = lons.to_vec()?;

    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            linestring::linestring_to_morton_coverage(&lat_data, &lon_data, order)
        })
    });

    match result {
        Ok(cells) => Ok(cells.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "linestring_coverage panicked",
        ))),
    }
}

/// Parse WKB (or EWKB) bytes into mortie coverage inputs, backend-free
/// (issue #157).
///
/// Returns `(kind, lats, lons, offsets)` — `kind` is `"polygonal"` or
/// `"linear"`, and ring `i` is `lats[offsets[i]:offsets[i+1]]` /
/// `lons[...]` in degrees (arrow list layout).  Polygonal geometries yield the
/// exterior **and** interior rings of every part, flattened, exactly as
/// `mortie.geometry.decompose` documents; `(x, y)` is unswapped to
/// `(lats, lons)` here.  Both byte orders, the ISO and EWKB dimension
/// spellings (Z/M dropped), and an EWKB SRID prefix (stripped) are accepted.
///
/// # Errors
/// `ValueError` for a truncated or malformed blob, an unsupported geometry
/// type, or an empty geometry.
#[pyfunction]
fn rust_wkb_rings(
    py: Python<'_>,
    data: &[u8],
) -> PyResult<(&'static str, PyObject, PyObject, PyObject)> {
    let rings = wkb::parse(data).map_err(PyValueError::new_err)?;
    Ok((
        rings.kind.as_str(),
        rings.lats.into_pyarray_bound(py).into_any().unbind(),
        rings.lons.into_pyarray_bound(py).into_any().unbind(),
        rings.offsets.into_pyarray_bound(py).into_any().unbind(),
    ))
}

/// MOC coverage of many WKB blobs in one call, backend-free (issue #157).
///
/// `blobs` is a sequence of WKB/EWKB geometries already screened against the
/// Python wrapper's input contract; `coerce` is that contract's one-blob
/// coercion (`mortie.geometry._wkb_bytes`), applied here to any entry that is
/// not already `bytes`.  Returns `(values, out_offsets)` in arrow list layout,
/// blob `i`'s MOC being `values[out_offsets[i]:out_offsets[i+1]]` and
/// byte-identical to what `from_wkb(blobs[i], moc=True)` returns for it —
/// `tolerance` / `max_cells` (in radians / cells) are **shared** across the
/// batch and mutually exclusive.  Errors name the lowest-index offending blob.
///
/// The GIL is released for the covering work, a chunk at a time: `bytes`
/// buffers are GIL-bound, so each chunk is copied into one contiguous buffer
/// while the GIL is held and only that buffer crosses into the parallel
/// region.  Two things keep that copy bounded by the chunk rather than by the
/// column: the chunk ends at whichever comes first, `CHUNK` blobs or
/// [`wkb::batch::CHUNK_BYTES`] bytes; and a non-`bytes` entry is coerced
/// *inside* the chunk loop, so the `bytes` it produces dies with the chunk
/// instead of standing for the whole call.
#[pyfunction]
#[pyo3(signature = (blobs, coerce, order=18, tolerance=None, max_cells=None, normalize=true))]
fn rust_wkbs_coverage_mocs(
    py: Python<'_>,
    blobs: Vec<Bound<'_, PyAny>>,
    coerce: Bound<'_, PyAny>,
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
    normalize: bool,
) -> PyResult<(PyObject, PyObject)> {
    if tolerance.is_some() && max_cells.is_some() {
        return Err(PyValueError::new_err(
            "pass at most one of tolerance / max_cells",
        ));
    }
    if !(1..=29).contains(&order) {
        return Err(PyValueError::new_err("Order must be between 1 and 29"));
    }
    let n = blobs.len();
    let mut out = coverage::batch::BatchMocs::new(n);
    let mut buf: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(coverage::batch::CHUNK + 1);
    let mut base = 0usize;
    while base < n {
        // Copy this chunk's blobs contiguously while the GIL is held; the
        // buffers are reused across chunks, so the copy peaks at one chunk.
        // The chunk ends at CHUNK blobs or CHUNK_BYTES bytes, whichever comes
        // first, so a column of fat geometries cannot turn "one chunk" into
        // gigabytes; a blob larger than the budget still forms a chunk of one.
        buf.clear();
        offsets.clear();
        offsets.push(0);
        let mut end = base;
        while end < n && !wkb::batch::chunk_full(end - base, buf.len()) {
            let entry = &blobs[end];
            // Keeps a coerced blob alive for the copy below.  Coercion happens
            // here, not in the wrapper's pre-pass: the `bytes` it makes for a
            // hex string or a byte buffer dies at the end of this iteration, so
            // a non-`bytes` column costs the same peak a `bytes` column does.
            let coerced;
            let bytes = match entry.downcast::<PyBytes>() {
                Ok(b) => b,
                Err(_) => {
                    coerced = coerce
                        .call1((entry,))
                        .map_err(|e| index_error(py, e, end))?;
                    coerced.downcast::<PyBytes>().map_err(|_| {
                        PyValueError::new_err(format!(
                            "blob {end}: WKB coercion did not return bytes"
                        ))
                    })?
                }
            };
            let blob = bytes.as_bytes();
            // `Vec` grows by doubling, which would turn a 64 MiB chunk into a
            // 128 MiB allocation and make "one chunk of copied bytes" mean
            // twice the budget.  Once a chunk is clearly heading for the
            // budget, take the budget exactly; a footprint column's ~1 MiB
            // chunks never reach this and keep doubling from small.
            let need = buf.len() + blob.len();
            if need > buf.capacity() && need > wkb::batch::CHUNK_BYTES / 2 {
                buf.reserve_exact(wkb::batch::CHUNK_BYTES.max(need) - buf.len());
            }
            buf.extend_from_slice(blob);
            offsets.push(buf.len());
            end += 1;
        }
        let covers = py.allow_threads(|| {
            wkb::batch::cover_chunk(&buf, &offsets, base, order, tolerance, max_cells, normalize)
        });
        out.extend_chunk(covers, base, max_cells)
            .map_err(PyValueError::new_err)?;
        out.reserve_estimate(end, n - end);
        base = end;
    }
    if let Some((count, first, effective)) = out.raised {
        let requested = max_cells.unwrap_or(0);
        let warnings = py.import_bound("warnings")?;
        warnings.call_method1(
            "warn",
            (format!(
                "max_cells={requested} is below the minimum to represent \
                 {count} geometry/ies; e.g. blob {first} uses {effective}"
            ),),
        )?;
    }
    Ok((
        out.values.into_pyarray_bound(py).into_any().unbind(),
        out.offsets.into_pyarray_bound(py).into_any().unbind(),
    ))
}

// ---------------------------------------------------------------------------
// morton_index datatype bindings (issue #35, phase 5)
//
// Vectorized batch wrappers over the `decimal_morton` kernel. The morton WORD is
// a native `u64` (issue #58): these bindings take and return `u64` numpy arrays
// directly, so the Z-order is simply the unsigned word order -- base cells 7..=11
// (prefix 8..=12) set bit 63 and sort after the northern cells with no special
// casing. (`rust_mi_from_legacy` is the lone exception: its INPUT stays `i64`
// because retired legacy decimal values were genuine signed i64.)
// These work with numpy only.
// ---------------------------------------------------------------------------

/// Vectorized `from_nested`: pack HEALPix NESTED ids at `depth` into
/// `morton_index` words (u64 numpy array out).
#[pyfunction]
fn rust_mi_from_nested(
    py: Python<'_>,
    nested_array: PyReadonlyArray1<u64>,
    depth: u8,
) -> PyResult<PyObject> {
    let nested = nested_array.to_vec()?;
    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            nested
                .par_iter()
                .map(|&n| decimal_morton::from_nested(n, depth))
                .collect::<Vec<u64>>()
        })
    });
    match result {
        Ok(words) => Ok(words.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "mi_from_nested panicked",
        ))),
    }
}

/// Vectorized `from_nested_point`: pack order-29 HEALPix NESTED ids into
/// `morton_index` **point** words (`Kind::Point`, u64 numpy array out). Point
/// encoding is order-29-only, so this takes no `depth` argument.
#[pyfunction]
fn rust_mi_from_nested_point(
    py: Python<'_>,
    nested_array: PyReadonlyArray1<u64>,
) -> PyResult<PyObject> {
    let nested = nested_array.to_vec()?;
    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            nested
                .par_iter()
                .map(|&n| decimal_morton::from_nested_point(n))
                .collect::<Vec<u64>>()
        })
    });
    match result {
        Ok(words) => Ok(words.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "mi_from_nested_point panicked",
        ))),
    }
}

/// Vectorized `to_nested`: unpack `morton_index` words (u64) back into
/// `(nested ids u64, depths u8)`. Raises `ValueError` if any word is the empty
/// sentinel or carries an invalid prefix.
#[pyfunction]
fn rust_mi_to_nested(py: Python<'_>, morton_array: PyReadonlyArray1<u64>) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let result: Result<(Vec<u64>, Vec<u8>), ()> = py.allow_threads(|| {
        let mut nested = Vec::with_capacity(data.len());
        let mut depths = Vec::with_capacity(data.len());
        for &w in &data {
            match decimal_morton::to_nested(w) {
                Some((depth, n)) => {
                    nested.push(n);
                    depths.push(depth);
                }
                None => return Err(()),
            }
        }
        Ok((nested, depths))
    });
    match result {
        Ok((nested, depths)) => {
            let py_nested = nested.into_pyarray_bound(py).into_any().unbind();
            let py_depths = depths.into_pyarray_bound(py).into_any().unbind();
            let tuple = pyo3::types::PyTuple::new_bound(py, &[py_nested, py_depths]);
            Ok(tuple.to_object(py))
        }
        Err(()) => Err(PyValueError::new_err(
            "morton_index array contains an empty or invalid word",
        )),
    }
}

/// Vectorized `coarsen`: coarsen every `morton_index` word (u64) to order `k`.
/// Raises `ValueError` if any word is empty or has an invalid prefix.
#[pyfunction]
fn rust_mi_coarsen(
    py: Python<'_>,
    morton_array: PyReadonlyArray1<u64>,
    k: u8,
) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let result: Result<Vec<u64>, ()> = py.allow_threads(|| {
        data.par_iter()
            .map(|&w| decimal_morton::coarsen(w, k).ok_or(()))
            .collect()
    });
    match result {
        Ok(words) => Ok(words.into_pyarray_bound(py).into_any().unbind()),
        Err(()) => Err(PyValueError::new_err(
            "morton_index array contains an empty or invalid word",
        )),
    }
}

/// Vectorized `order_of`: read the HEALPix order of every word (u8 array out).
#[pyfunction]
fn rust_mi_order_of(py: Python<'_>, morton_array: PyReadonlyArray1<u64>) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let orders: Vec<u8> = py.allow_threads(|| {
        data.par_iter()
            .map(|&w| decimal_morton::order_of(w))
            .collect()
    });
    Ok(orders.into_pyarray_bound(py).into_any().unbind())
}

/// Vectorized `base_cell_of`: read the base cell `0..=11` of every word.
/// The empty sentinel / invalid prefix maps to `255` (no valid base cell).
#[pyfunction]
fn rust_mi_base_cell_of(py: Python<'_>, morton_array: PyReadonlyArray1<u64>) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let bases: Vec<u8> = py.allow_threads(|| {
        data.par_iter()
            .map(|&w| decimal_morton::base_cell_of(w).unwrap_or(255))
            .collect()
    });
    Ok(bases.into_pyarray_bound(py).into_any().unbind())
}

/// Vectorized `encode` from base cells, packed tuples and orders.
///
/// `tuples` is a flat `(n, 29)` row-major u8 array; row `i` holds the stored
/// `0..=3` tuples for element `i` (only the first `orders[i]` entries are read).
/// Returns the u64 `morton_index` words.
#[pyfunction]
fn rust_mi_encode(
    py: Python<'_>,
    base_cells: PyReadonlyArray1<u8>,
    tuples: PyReadonlyArray2<u8>,
    orders: PyReadonlyArray1<u8>,
) -> PyResult<PyObject> {
    let bases = base_cells.to_vec()?;
    let orders = orders.to_vec()?;
    let shape = tuples.shape();
    let (n, ncols) = (shape[0], shape[1]);
    if bases.len() != n || orders.len() != n {
        return Err(PyValueError::new_err(
            "base_cells, tuples and orders must share the same length",
        ));
    }
    if ncols < decimal_morton::MAX_ORDER as usize {
        return Err(PyValueError::new_err(
            "tuples must have at least 29 columns",
        ));
    }
    let flat = tuples.to_vec()?;
    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            (0..n)
                .map(|i| {
                    let row = &flat[i * ncols..i * ncols + ncols];
                    decimal_morton::encode(bases[i], row, orders[i])
                })
                .collect::<Vec<u64>>()
        })
    });
    match result {
        Ok(words) => Ok(words.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(e, "mi_encode panicked"))),
    }
}

/// Vectorized `decode`: unpack each word into its base cell, order, kind flag
/// (0 = area, 1 = point) and its full tuple row.
///
/// Returns `(base_cells u8, orders u8, kinds u8, tuples (n,29) u8)`; tuple
/// columns past an element's order are zero. Raises `ValueError` on any empty /
/// invalid word.
#[pyfunction]
fn rust_mi_decode(py: Python<'_>, morton_array: PyReadonlyArray1<u64>) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let n = data.len();
    let ncols = decimal_morton::MAX_ORDER as usize;
    type Decoded = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);
    let result: Result<Decoded, String> = py.allow_threads(|| {
        let mut bases = Vec::with_capacity(n);
        let mut orders = Vec::with_capacity(n);
        let mut kinds = Vec::with_capacity(n);
        let mut flat = vec![0u8; n * ncols];
        for (i, &w) in data.iter().enumerate() {
            let dec = decimal_morton::decode(w).map_err(|e| e.to_string())?;
            bases.push(dec.base_cell);
            orders.push(dec.order);
            kinds.push(matches!(dec.kind, decimal_morton::Kind::Point) as u8);
            for (j, &t) in dec.tuples.iter().enumerate() {
                flat[i * ncols + j] = t;
            }
        }
        Ok((bases, orders, kinds, flat))
    });
    match result {
        Ok((bases, orders, kinds, flat)) => {
            let arr = numpy::ndarray::Array2::from_shape_vec((n, ncols), flat)
                .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;
            let py_tuples = PyArray2::from_owned_array_bound(py, arr)
                .into_any()
                .unbind();
            let out = pyo3::types::PyTuple::new_bound(
                py,
                &[
                    bases.into_pyarray_bound(py).into_any().unbind(),
                    orders.into_pyarray_bound(py).into_any().unbind(),
                    kinds.into_pyarray_bound(py).into_any().unbind(),
                    py_tuples,
                ],
            );
            Ok(out.to_object(py))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

/// Vectorized one-way `legacy_decimal_i64 -> packed_u64` converter (issue #48).
///
/// Maps each retired legacy decimal Morton index to its canonical packed word
/// (returned as a `u64`). Kept for testing new output against old pinned values;
/// there is no packed -> legacy inverse beyond the render-only repr. The INPUT
/// stays `i64` because legacy decimal values were genuine signed i64 (possibly
/// negative). Raises `ValueError` if any input is `0` (not a legacy Morton).
#[pyfunction]
fn rust_mi_from_legacy(py: Python<'_>, legacy_array: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let data = legacy_array.to_vec()?;
    let result = py.allow_threads(|| {
        std::panic::catch_unwind(|| {
            data.par_iter()
                .map(|&m| decimal_morton::from_legacy_decimal(m))
                .collect::<Vec<u64>>()
        })
    });
    match result {
        Ok(words) => Ok(words.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(
            e,
            "mi_from_legacy panicked",
        ))),
    }
}

/// Vectorized decode-through-kernel decimal repr (issue #48).
///
/// Renders each packed word as its human-readable decimal string (the canonical
/// render-only repr; up to 30 chars at order 29, which is why it is a string and
/// not an integer). Returns a Python list of `str`. Raises `ValueError` on any
/// empty / invalid word.
#[pyfunction]
fn rust_mi_decimal_repr(py: Python<'_>, morton_array: PyReadonlyArray1<u64>) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let result: Result<Vec<String>, String> = py.allow_threads(|| {
        data.iter()
            .map(|&w| {
                decimal_morton::to_decimal_repr(w).ok_or_else(|| {
                    "morton_index array contains an empty or invalid word".to_string()
                })
            })
            .collect()
    });
    match result {
        Ok(strings) => Ok(pyo3::types::PyList::new_bound(py, &strings)
            .into_any()
            .unbind()),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

/// Vectorized decimal-string -> packed word parse (issue #114).
///
/// The inverse of [`rust_mi_decimal_repr`]: parses each decimal Morton id back
/// to its packed word (u64 numpy array out). A `p`-marked id yields the POINT
/// word, an unmarked one the AREA word (the spec section 4 tie-break). Raises
/// `ValueError` naming the first malformed id, in input order.
///
/// Serial like the emit side rather than rayon-parallel, for two reasons. The
/// reported error stays deterministic (the *first* bad id, not whichever thread
/// failed first); and parallelism has little to win here anyway -- extracting
/// `Vec<String>` copies every id into Rust memory *while holding the GIL*,
/// before `allow_threads` runs, and that extraction dominates. Measured at
/// N=200k order-29 ids: 57 ms from a `<U32` numpy array vs 27 ms from a Python
/// list, where the delta is pure `str` boxing, so the parse `par_iter` would
/// split is a minority of the total. Cutting the extraction (reading the
/// fixed-width UCS-4 buffer directly) is the win worth having; see issue #114.
#[pyfunction]
fn rust_mi_from_decimal(py: Python<'_>, decimals: Vec<String>) -> PyResult<PyObject> {
    let result: Result<Vec<u64>, String> = py.allow_threads(|| {
        decimals
            .iter()
            .map(|s| decimal_morton::from_decimal_repr(s).map_err(|e| e.to_string()))
            .collect()
    });
    match result {
        Ok(words) => Ok(words.into_pyarray_bound(py).into_any().unbind()),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

/// Dissolve a morton cover into the classified planar rings of its outline.
///
/// Returns `(shells, holes)`: two Python lists of `(N, 2)` f64 arrays of
/// `(lon, lat)` degrees.  Crossing rings are cut at +/-180 and reconnected by
/// the GeoJSON convention (explicit +/-90 pole vertices stitched down the
/// antimeridian for a pole-enclosing region).  The Python side builds the
/// backend Polygons and nests holes — see `mortie/geometry.py`.  Raises
/// `ValueError` for a cover spanning near or over a hemisphere, where the
/// exterior/hole winding sign is ambiguous (issue #108).
#[pyfunction]
#[pyo3(signature = (morton, step=1))]
fn rust_dissolve(py: Python<'_>, morton: PyReadonlyArray1<u64>, step: u32) -> PyResult<PyObject> {
    let data = morton.to_vec()?;
    let result = py.allow_threads(|| std::panic::catch_unwind(|| dissolve::dissolve(&data, step)));
    let classified = match result {
        Ok(Ok(c)) => c,
        Ok(Err(msg)) => return Err(PyValueError::new_err(msg)),
        Err(e) => return Err(PyValueError::new_err(panic_msg(e, "dissolve panicked"))),
    };
    let to_list = |rings: Vec<Vec<(f64, f64)>>| {
        let lst = pyo3::types::PyList::empty_bound(py);
        for ring in rings {
            let mut flat = Vec::with_capacity(ring.len() * 2);
            for (lon, lat) in ring {
                flat.push(lon);
                flat.push(lat);
            }
            let n = flat.len() / 2;
            let arr = numpy::ndarray::Array2::from_shape_vec((n, 2), flat).unwrap();
            lst.append(PyArray2::from_owned_array_bound(py, arr))
                .unwrap();
        }
        lst.into_any().unbind()
    };
    let shells = to_list(classified.shells);
    let holes = to_list(classified.holes);
    Ok(pyo3::types::PyTuple::new_bound(py, &[shells, holes]).to_object(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _rustie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_mort2nested, m)?)?;
    m.add_function(wrap_pyfunction!(rust_nested2mort, m)?)?;
    m.add_function(wrap_pyfunction!(rank_xy::rust_rank_to_xy, m)?)?;
    m.add_function(wrap_pyfunction!(rank_xy::rust_xy_to_rank, m)?)?;
    m.add_function(wrap_pyfunction!(split_children_rust, m)?)?;
    m.add_function(wrap_pyfunction!(rust_geo2mort, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ang2pix, m)?)?;
    m.add_function(wrap_pyfunction!(rust_pix2ang, m)?)?;
    m.add_function(wrap_pyfunction!(rust_boundaries, m)?)?;
    m.add_function(wrap_pyfunction!(rust_dissolve, m)?)?;
    m.add_function(wrap_pyfunction!(rust_vec2ang, m)?)?;
    m.add_function(wrap_pyfunction!(rust_morton_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ring_is_simple, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ring_validity, m)?)?;
    m.add_function(wrap_pyfunction!(rust_polygon_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(rust_polygon_coverage_moc, m)?)?;
    m.add_function(wrap_pyfunction!(rust_multipolygon_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(rust_multipolygon_coverage_moc, m)?)?;
    m.add_function(wrap_pyfunction!(rust_polygons_coverage_mocs, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_to_order, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_to_order_count, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mocs_to_orders, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_or, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_and, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_minus, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_xor, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_intersects, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mocs_and, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mocs_intersect, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_min, m)?)?;
    m.add_function(wrap_pyfunction!(rust_common_ancestors, m)?)?;
    m.add_function(wrap_pyfunction!(rust_children_of, m)?)?;
    m.add_function(wrap_pyfunction!(rust_linestring_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(toc::rust_time2toc, m)?)?;
    m.add_function(wrap_pyfunction!(toc::rust_span2toc, m)?)?;
    m.add_function(wrap_pyfunction!(toc::rust_toc2time, m)?)?;
    m.add_function(wrap_pyfunction!(toc::rust_toc_merge, m)?)?;
    m.add_function(wrap_pyfunction!(toc::rust_toc_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(toc::rust_toc_is_range, m)?)?;
    m.add_function(wrap_pyfunction!(toc::rust_toc_window, m)?)?;
    m.add_function(wrap_pyfunction!(rust_wkb_rings, m)?)?;
    m.add_function(wrap_pyfunction!(rust_wkbs_coverage_mocs, m)?)?;
    #[cfg(feature = "descent-stats")]
    m.add_function(wrap_pyfunction!(rust_descent_stats_take, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_from_nested, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_from_nested_point, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_to_nested, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_coarsen, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_order_of, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_base_cell_of, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_encode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_decode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_from_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_decimal_repr, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_from_decimal, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_ffi::rust_mi_export_c_schema, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_ffi::rust_mi_export_c_array, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_ffi::rust_mi_import_c_array, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_ffi::rust_mi_import_c_stream, m)?)?;
    Ok(())
}
