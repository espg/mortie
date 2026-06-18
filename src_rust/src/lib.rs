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

use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods,
            PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAnyMethods, PyModule};
use rayon::prelude::*;

/// Convert normalized HEALPix addresses to morton indices (vectorized)
///
/// This function accepts either scalar values or NumPy arrays and returns
/// the corresponding morton indices.
///
/// # Arguments
/// * `order` - Tessellation order (scalar or array)
/// * `normed` - Normalized HEALPix address (scalar or array)
/// * `parents` - Parent base cell (scalar or array)
///
/// # Returns
/// Morton indices as i64 scalar or NumPy array
///
/// # Examples
/// ```python
/// import numpy as np
/// from mortie_rs import fast_norm2mort
///
/// # Scalar inputs
/// result = fast_norm2mort(18, 1000, 2)
///
/// # Array inputs
/// orders = np.array([18, 18, 18], dtype=np.int64)
/// normed = np.array([100, 200, 300], dtype=np.int64)
/// parents = np.array([2, 3, 8], dtype=np.int64)
/// results = fast_norm2mort(orders, normed, parents)
/// ```
#[pyfunction]
fn fast_norm2mort<'py>(
    py: Python<'py>,
    order: &Bound<'py, PyAny>,
    normed: &Bound<'py, PyAny>,
    parents: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    // Check if inputs are scalars or arrays
    let order_is_scalar = order.extract::<i64>().is_ok();
    let normed_is_scalar = normed.extract::<i64>().is_ok();
    let parents_is_scalar = parents.extract::<i64>().is_ok();

    // All scalars - return scalar
    if order_is_scalar && normed_is_scalar && parents_is_scalar {
        let order_val = order.extract::<i64>()?;
        let normed_val = normed.extract::<i64>()?;
        let parents_val = parents.extract::<i64>()?;

        if order_val > 18 {
            return Err(PyValueError::new_err("Max order is 18 (to output to 64-bit int)."));
        }

        let result = morton::fast_norm2mort_scalar(order_val, normed_val, parents_val);
        return Ok(result.to_object(py));
    }

    // At least one array - extract arrays
    let order_arr = if order_is_scalar {
        let val = order.extract::<i64>()?;
        vec![val]
    } else {
        order.extract::<PyReadonlyArray1<i64>>()?.to_vec()?
    };

    let normed_arr = if normed_is_scalar {
        let val = normed.extract::<i64>()?;
        vec![val]
    } else {
        normed.extract::<PyReadonlyArray1<i64>>()?.to_vec()?
    };

    let parents_arr = if parents_is_scalar {
        let val = parents.extract::<i64>()?;
        vec![val]
    } else {
        parents.extract::<PyReadonlyArray1<i64>>()?.to_vec()?
    };

    // Determine output length (broadcast scalars to match array length)
    let lengths = vec![order_arr.len(), normed_arr.len(), parents_arr.len()];
    let max_len = *lengths.iter().max().unwrap();

    // Check all non-scalar inputs have same length or length 1
    for &len in &lengths {
        if len != 1 && len != max_len {
            return Err(PyValueError::new_err(
                "All array inputs must have the same length"
            ));
        }
    }

    // Validate max order
    let max_order = *order_arr.iter().max().unwrap();
    if max_order > 18 {
        return Err(PyValueError::new_err("Max order is 18 (to output to 64-bit int)."));
    }

    // Parallel computation using rayon (GIL released for the pure-Rust region)
    let results: Vec<i64> = py.allow_threads(|| {
        (0..max_len)
            .into_par_iter()
            .map(|i| {
                let order_val = order_arr[if order_arr.len() == 1 { 0 } else { i }];
                let normed_val = normed_arr[if normed_arr.len() == 1 { 0 } else { i }];
                let parents_val = parents_arr[if parents_arr.len() == 1 { 0 } else { i }];
                morton::fast_norm2mort_scalar(order_val, normed_val, parents_val)
            })
            .collect()
    });

    // Return as NumPy array
    Ok(results.into_pyarray_bound(py).into_any().unbind())
}

/// Decode morton indices to HEALPix NESTED cell ids and depths (vectorized).
///
/// # Arguments
/// * `morton_array` - Morton indices (i64 NumPy array)
///
/// # Returns
/// Tuple of two NumPy arrays: (nested cell ids as u64, depths as u8).
///
/// Callers must pre-validate inputs: each digit of every morton index is
/// expected to be in 1-4 and the parent in 0-11. A zero morton raises a
/// `ValueError`; other malformed indices are not checked in release builds
/// (the digit check is a debug assertion) and will silently mis-decode, so
/// validate upstream (as ``mort2norm`` does via ``validate_morton``).
#[pyfunction]
fn rust_mort2nested(
    py: Python<'_>,
    morton_array: PyReadonlyArray1<i64>,
) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;

    let result = std::panic::catch_unwind(|| {
        let mut nested = Vec::with_capacity(data.len());
        let mut depths = Vec::with_capacity(data.len());
        for &m in &data {
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
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "mort2nested panicked".to_string()
            };
            Err(PyValueError::new_err(msg))
        }
    }
}

/// Encode HEALPix NESTED cell ids and depths to morton indices (vectorized).
///
/// # Arguments
/// * `nested_array` - HEALPix NESTED cell ids (u64 NumPy array)
/// * `depth_array` - HEALPix depths/orders (u8 NumPy array), same length
///
/// # Returns
/// Morton indices as an i64 NumPy array.
#[pyfunction]
fn rust_nested2mort(
    py: Python<'_>,
    nested_array: PyReadonlyArray1<u64>,
    depth_array: PyReadonlyArray1<u8>,
) -> PyResult<PyObject> {
    let nested = nested_array.to_vec()?;
    let depths = depth_array.to_vec()?;

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
            .collect::<Vec<i64>>()
    });

    match result {
        Ok(morton) => Ok(morton.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "nested2mort panicked".to_string()
            };
            Err(PyValueError::new_err(msg))
        }
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
    morton_array: PyReadonlyArray1<i64>,
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
    if order > 18 {
        return Err(PyValueError::new_err("Max order is 18 (to output to 64-bit int)."));
    }

    let lats_is_scalar = lats.extract::<f64>().is_ok();
    let lons_is_scalar = lons.extract::<f64>().is_ok();

    // Both scalars → return scalar
    if lats_is_scalar && lons_is_scalar {
        let lat = lats.extract::<f64>()?;
        let lon = lons.extract::<f64>()?;
        let result = geo2mort::geo2mort_scalar(lat, lon, order);
        return Ok(result.to_object(py));
    }

    // At least one array
    let lat_arr = if lats_is_scalar {
        vec![lats.extract::<f64>()?]
    } else {
        lats.extract::<PyReadonlyArray1<f64>>()?.to_vec()?
    };

    let lon_arr = if lons_is_scalar {
        vec![lons.extract::<f64>()?]
    } else {
        lons.extract::<PyReadonlyArray1<f64>>()?.to_vec()?
    };

    let max_len = lat_arr.len().max(lon_arr.len());

    if (lat_arr.len() != 1 && lat_arr.len() != max_len)
        || (lon_arr.len() != 1 && lon_arr.len() != max_len)
    {
        return Err(PyValueError::new_err(
            "lats and lons must have the same length",
        ));
    }

    let results: Vec<i64> = py.allow_threads(|| {
        (0..max_len)
            .into_par_iter()
            .map(|i| {
                let lat = lat_arr[if lat_arr.len() == 1 { 0 } else { i }];
                let lon = lon_arr[if lon_arr.len() == 1 { 0 } else { i }];
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
    let lon_is_scalar = lon.extract::<f64>().is_ok();
    let lat_is_scalar = lat.extract::<f64>().is_ok();

    if lon_is_scalar && lat_is_scalar {
        let lon_val = lon.extract::<f64>()?;
        let lat_val = lat.extract::<f64>()?;
        let result = geo2mort::ang2pix_scalar(depth, lon_val, lat_val);
        return Ok((result as i64).to_object(py));
    }

    let lon_arr = if lon_is_scalar {
        vec![lon.extract::<f64>()?]
    } else {
        lon.extract::<PyReadonlyArray1<f64>>()?.to_vec()?
    };
    let lat_arr = if lat_is_scalar {
        vec![lat.extract::<f64>()?]
    } else {
        lat.extract::<PyReadonlyArray1<f64>>()?.to_vec()?
    };

    let max_len = lon_arr.len().max(lat_arr.len());
    if (lon_arr.len() != 1 && lon_arr.len() != max_len)
        || (lat_arr.len() != 1 && lat_arr.len() != max_len)
    {
        return Err(PyValueError::new_err("lon and lat must have the same length"));
    }

    let results: Vec<i64> = py.allow_threads(|| {
        (0..max_len)
            .into_par_iter()
            .map(|i| {
                let lo = lon_arr[if lon_arr.len() == 1 { 0 } else { i }];
                let la = lat_arr[if lat_arr.len() == 1 { 0 } else { i }];
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
fn rust_pix2ang<'py>(
    py: Python<'py>,
    depth: u8,
    pixel: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let pixel_is_scalar = pixel.extract::<i64>().is_ok();

    if pixel_is_scalar {
        let pix = pixel.extract::<i64>()? as u64;
        let (lon, lat) = geo2mort::pix2ang_scalar(depth, pix);
        return Ok((lon, lat).to_object(py));
    }

    let pixel_arr = pixel.extract::<PyReadonlyArray1<i64>>()?.to_vec()?;
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
    let pixel_is_scalar = pixel.extract::<i64>().is_ok();
    let ncols = 4 * step as usize;

    if step == 1 {
        // Fast path: original 4-corner code
        if pixel_is_scalar {
            let pix = pixel.extract::<i64>()? as u64;
            let xyz = geo2mort::boundaries_scalar(depth, pix);
            let arr = numpy::ndarray::Array2::from_shape_fn((3, 4), |(r, c)| xyz[r][c]);
            return Ok(PyArray2::from_owned_array_bound(py, arr).into_any().unbind());
        }
        let pixel_arr = pixel.extract::<PyReadonlyArray1<i64>>()?.to_vec()?;
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
        return Ok(PyArray3::from_owned_array_bound(py, arr).into_any().unbind());
    }

    // step > 1: use path_along_cell_edge
    if pixel_is_scalar {
        let pix = pixel.extract::<i64>()? as u64;
        let pts = geo2mort::boundaries_step_scalar(depth, pix, step);
        // pts is Vec<[f64; 3]> with ncols entries → shape (3, ncols)
        let arr = numpy::ndarray::Array2::from_shape_fn((3, ncols), |(r, c)| pts[c][r]);
        return Ok(PyArray2::from_owned_array_bound(py, arr).into_any().unbind());
    }

    let pixel_arr = pixel.extract::<PyReadonlyArray1<i64>>()?.to_vec()?;
    let n = pixel_arr.len();
    let results: Vec<Vec<[f64; 3]>> = py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .map(|i| geo2mort::boundaries_step_scalar(depth, pixel_arr[i] as u64, step))
            .collect()
    });
    // Shape (N, 3, ncols)
    let mut flat = Vec::with_capacity(n * 3 * ncols);
    for pts in &results {
        for r in 0..3 {
            for c in 0..ncols {
                flat.push(pts[c][r]);
            }
        }
    }
    let arr = numpy::ndarray::Array3::from_shape_vec((n, 3, ncols), flat)
        .map_err(|e| PyValueError::new_err(format!("shape error: {}", e)))?;
    Ok(PyArray3::from_owned_array_bound(py, arr).into_any().unbind())
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
fn rust_vec2ang<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f64>,
) -> PyResult<PyObject> {
    let shape = vectors.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(
            "vectors must have shape (N, 3)"
        ));
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
/// * `morton_array` - NumPy array of morton indices (i64)
/// * `k` - Border width in cells (default 1, 8-connected neighbors)
///
/// # Returns
/// NumPy array of border morton indices (sorted)
#[pyfunction]
#[pyo3(signature = (morton_array, k=1))]
fn rust_morton_buffer(
    py: Python<'_>,
    morton_array: PyReadonlyArray1<i64>,
    k: u32,
) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;

    let result = py.allow_threads(|| std::panic::catch_unwind(|| buffer::morton_buffer(&data, k)));

    match result {
        Ok(border) => Ok(border.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "morton_buffer panicked".to_string()
            };
            Err(PyValueError::new_err(msg))
        }
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
/// Sorted NumPy array of morton indices (i64)
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
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "polygon_coverage panicked".to_string()
            };
            Err(PyValueError::new_err(msg))
        }
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

    let result = py.allow_threads(|| std::panic::catch_unwind(|| {
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
            (coverage::polygon_to_morton_moc(&lat_data, &lon_data, order), None)
        }
    }));

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
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "polygon_coverage_moc panicked".to_string()
            };
            Err(PyValueError::new_err(msg))
        }
    }
}

/// Extract a readable message from a caught panic payload.
fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "coverage panicked".to_string()
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
        Err(e) => Err(PyValueError::new_err(panic_msg(e))),
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
        return Err(PyValueError::new_err("pass at most one of tolerance / max_cells"));
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
        Err(e) => Err(PyValueError::new_err(panic_msg(e))),
    }
}

/// Compress a (mixed-order) morton set into its canonical compact MOC: merge
/// any 4 complete sibling cells into their parent, and drop any cell already
/// contained in a coarser one.  Use after unioning per-part covers.
#[pyfunction]
fn rust_moc_normalize(py: Python<'_>, morton: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let data = morton.to_vec()?;
    let normalized = py.allow_threads(|| moc::normalize(&data));
    Ok(normalized.into_pyarray_bound(py).into_any().unbind())
}

/// Densify a (mixed-order) morton set to a flat list at `order`.
#[pyfunction]
#[pyo3(signature = (morton, order))]
fn rust_moc_to_order(py: Python<'_>, morton: PyReadonlyArray1<i64>, order: u8) -> PyResult<PyObject> {
    let data = morton.to_vec()?;
    let densified = py.allow_threads(|| moc::to_order(&data, order));
    Ok(densified.into_pyarray_bound(py).into_any().unbind())
}

/// Compute morton indices tracing a linestring (open polyline).
///
/// # Arguments
/// * `lats` - Vertex latitudes in degrees (NumPy array, >=2)
/// * `lons` - Vertex longitudes in degrees (NumPy array, >=2)
/// * `order` - HEALPix order/depth (default 18)
///
/// # Returns
/// Sorted, unique NumPy array of morton indices (i64) tracing the line
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
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "linestring_coverage panicked".to_string()
            };
            Err(PyValueError::new_err(msg))
        }
    }
}

// ---------------------------------------------------------------------------
// morton_index datatype bindings (issue #35, phase 5)
//
// Vectorized batch wrappers over the `decimal_morton` kernel. Storage is i64
// (the signed presentation form); the raw bit pattern is the kernel's u64 word,
// so every binding reinterprets `i64 <-> u64` with a bitwise `as` cast and the
// raw-i64 sort still equals the raw-u64 Z-order (no valid base cell 0..=11 /
// prefix 1..=12 sets the sign bit). These work with numpy only.
// ---------------------------------------------------------------------------

/// Vectorized `from_nested`: pack HEALPix NESTED ids at `depth` into
/// `morton_index` words (i64 numpy array out).
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
                .map(|&n| decimal_morton::from_nested(n, depth) as i64)
                .collect::<Vec<i64>>()
        })
    });
    match result {
        Ok(words) => Ok(words.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(e))),
    }
}

/// Vectorized `to_nested`: unpack `morton_index` words (i64) back into
/// `(nested ids u64, depths u8)`. Raises `ValueError` if any word is the empty
/// sentinel or carries an invalid prefix.
#[pyfunction]
fn rust_mi_to_nested(py: Python<'_>, morton_array: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let result: Result<(Vec<u64>, Vec<u8>), ()> = py.allow_threads(|| {
        let mut nested = Vec::with_capacity(data.len());
        let mut depths = Vec::with_capacity(data.len());
        for &w in &data {
            match decimal_morton::to_nested(w as u64) {
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

/// Vectorized `coarsen`: coarsen every `morton_index` word (i64) to order `k`.
/// Raises `ValueError` if any word is empty or has an invalid prefix.
#[pyfunction]
fn rust_mi_coarsen(
    py: Python<'_>,
    morton_array: PyReadonlyArray1<i64>,
    k: u8,
) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let result: Result<Vec<i64>, ()> = py.allow_threads(|| {
        data.par_iter()
            .map(|&w| {
                decimal_morton::coarsen(w as u64, k)
                    .map(|c| c as i64)
                    .ok_or(())
            })
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
fn rust_mi_order_of(py: Python<'_>, morton_array: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let orders: Vec<u8> = py.allow_threads(|| {
        data.par_iter()
            .map(|&w| decimal_morton::order_of(w as u64))
            .collect()
    });
    Ok(orders.into_pyarray_bound(py).into_any().unbind())
}

/// Vectorized `base_cell_of`: read the base cell `0..=11` of every word.
/// The empty sentinel / invalid prefix maps to `255` (no valid base cell).
#[pyfunction]
fn rust_mi_base_cell_of(py: Python<'_>, morton_array: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let bases: Vec<u8> = py.allow_threads(|| {
        data.par_iter()
            .map(|&w| decimal_morton::base_cell_of(w as u64).unwrap_or(255))
            .collect()
    });
    Ok(bases.into_pyarray_bound(py).into_any().unbind())
}

/// Vectorized `encode` from base cells, packed tuples and orders.
///
/// `tuples` is a flat `(n, 29)` row-major u8 array; row `i` holds the stored
/// `0..=3` tuples for element `i` (only the first `orders[i]` entries are read).
/// Returns the i64 `morton_index` words.
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
                    decimal_morton::encode(bases[i], row, orders[i]) as i64
                })
                .collect::<Vec<i64>>()
        })
    });
    match result {
        Ok(words) => Ok(words.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(e))),
    }
}

/// Vectorized `decode`: unpack each word into its base cell, order, kind flag
/// (0 = area, 1 = point) and its full tuple row.
///
/// Returns `(base_cells u8, orders u8, kinds u8, tuples (n,29) u8)`; tuple
/// columns past an element's order are zero. Raises `ValueError` on any empty /
/// invalid word.
#[pyfunction]
fn rust_mi_decode(py: Python<'_>, morton_array: PyReadonlyArray1<i64>) -> PyResult<PyObject> {
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
            let dec = decimal_morton::decode(w as u64).map_err(|e| e.to_string())?;
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

/// A Python module implemented in Rust.
#[pymodule]
fn _rustie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_norm2mort, m)?)?;
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
    m.add_function(wrap_pyfunction!(rust_linestring_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_from_nested, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_to_nested, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_coarsen, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_order_of, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_base_cell_of, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_encode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mi_decode, m)?)?;
    Ok(())
}
