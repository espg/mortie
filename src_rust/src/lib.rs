//! Rust-accelerated morton indexing for mortie
//!
//! This module provides Python bindings for fast morton encoding operations,
//! replacing the numba-accelerated functions to eliminate Dask conflicts.

pub mod buffer;
pub mod cell_geom;
pub mod coverage;
pub mod decimal_morton;
pub mod geo2mort;
pub mod linestring;
pub mod moc;
pub mod morton;
pub mod prefix_trie;
pub mod sphere;

use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyModule};
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
/// * `order` - HEALPix order (default 18)
#[pyfunction]
#[pyo3(signature = (lats, lons, order=18))]
fn rust_geo2mort<'py>(
    py: Python<'py>,
    lats: &Bound<'py, PyAny>,
    lons: &Bound<'py, PyAny>,
    order: u8,
) -> PyResult<PyObject> {
    if order > decimal_morton::MAX_ORDER {
        return Err(PyValueError::new_err(
            "Max order is 29 (the packed-u64 decimal_morton limit).",
        ));
    }

    let (lat_arr, lats_is_scalar) = extract_f64_input(lats)?;
    let (lon_arr, lons_is_scalar) = extract_f64_input(lons)?;

    // Both scalars → return scalar
    if lats_is_scalar && lons_is_scalar {
        let result = geo2mort::geo2mort_scalar(lat_arr[0], lon_arr[0], order);
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
                geo2mort::geo2mort_scalar(lat, lon, order)
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
#[pyfunction]
#[pyo3(signature = (lats, lons, order=18, tolerance=None, max_cells=None))]
fn rust_polygon_coverage_moc(
    py: Python<'_>,
    lats: PyReadonlyArray1<f64>,
    lons: PyReadonlyArray1<f64>,
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
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
                    coverage::polygon_to_morton_moc_tolerance(&lat_data, &lon_data, order, tol),
                    None,
                )
            } else if let Some(budget) = max_cells {
                let (cells, effective) =
                    coverage::polygon_to_morton_moc_budget(&lat_data, &lon_data, order, budget);
                let warn = (effective > budget).then_some((budget, effective));
                (cells, warn)
            } else {
                (
                    coverage::polygon_to_morton_moc(&lat_data, &lon_data, order),
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

/// Extract a readable message from a caught panic payload, falling back to
/// `fallback` when the payload is neither a `String` nor a `&str`.
fn panic_msg(e: Box<dyn std::any::Any + Send>, fallback: &str) -> String {
    if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else {
        fallback.to_string()
    }
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
#[pyo3(signature = (lats, lons, order=18, tolerance=None, max_cells=None))]
fn rust_multipolygon_coverage_moc(
    py: Python<'_>,
    lats: Vec<PyReadonlyArray1<f64>>,
    lons: Vec<PyReadonlyArray1<f64>>,
    order: u8,
    tolerance: Option<f64>,
    max_cells: Option<usize>,
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
            coverage::multipolygon_to_morton_moc(&la, &lo, order, tolerance, max_cells)
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
#[pyfunction]
#[pyo3(signature = (morton, order))]
fn rust_moc_to_order(
    py: Python<'_>,
    morton: PyReadonlyArray1<u64>,
    order: u8,
) -> PyResult<PyObject> {
    let data = morton.to_vec()?;
    let densified = py.allow_threads(|| moc::to_order(&data, order));
    Ok(densified.into_pyarray_bound(py).into_any().unbind())
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

/// A Python module implemented in Rust.
#[pymodule]
fn _rustie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_mort2nested, m)?)?;
    m.add_function(wrap_pyfunction!(rust_nested2mort, m)?)?;
    m.add_function(wrap_pyfunction!(split_children_rust, m)?)?;
    m.add_function(wrap_pyfunction!(rust_geo2mort, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ang2pix, m)?)?;
    m.add_function(wrap_pyfunction!(rust_pix2ang, m)?)?;
    m.add_function(wrap_pyfunction!(rust_boundaries, m)?)?;
    m.add_function(wrap_pyfunction!(rust_vec2ang, m)?)?;
    m.add_function(wrap_pyfunction!(rust_morton_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(rust_polygon_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(rust_polygon_coverage_moc, m)?)?;
    m.add_function(wrap_pyfunction!(rust_multipolygon_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(rust_multipolygon_coverage_moc, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_to_order, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_or, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_and, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_minus, m)?)?;
    m.add_function(wrap_pyfunction!(rust_moc_xor, m)?)?;
    m.add_function(wrap_pyfunction!(rust_linestring_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_from_nested, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_to_nested, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_coarsen, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_order_of, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_base_cell_of, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_encode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_decode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_from_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_decimal_repr, m)?)?;
    Ok(())
}
