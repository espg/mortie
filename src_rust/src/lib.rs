//! Rust-accelerated morton indexing for mortie
//!
//! This module provides Python bindings for fast morton encoding operations,
//! replacing the numba-accelerated functions to eliminate Dask conflicts.

pub mod buffer;
pub mod geo2mort;
pub mod morton;
pub mod prefix_trie;

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

    // Parallel computation using rayon
    let results: Vec<i64> = (0..max_len)
        .into_par_iter()
        .map(|i| {
            let order_val = order_arr[if order_arr.len() == 1 { 0 } else { i }];
            let normed_val = normed_arr[if normed_arr.len() == 1 { 0 } else { i }];
            let parents_val = parents_arr[if parents_arr.len() == 1 { 0 } else { i }];
            morton::fast_norm2mort_scalar(order_val, normed_val, parents_val)
        })
        .collect();

    // Return as NumPy array
    Ok(results.into_pyarray_bound(py).into_any().unbind())
}

/// Build compacted prefix trie over morton indices (Python binding)
///
/// Returns a list of tuples: (characteristic, count, original_indices, child_node_ids, depth)
#[pyfunction]
#[pyo3(signature = (morton_array, max_depth=None))]
fn split_children_rust(
    py: Python<'_>,
    morton_array: PyReadonlyArray1<i64>,
    max_depth: Option<usize>,
) -> PyResult<PyObject> {
    let data = morton_array.to_vec()?;
    let flat = prefix_trie::split_children_flat(&data, max_depth);

    // Convert Vec<FlatNode> to a Python list of tuples
    let py_list = pyo3::types::PyList::empty_bound(py);
    for (characteristic, count, indices, child_ids, depth) in flat {
        let py_indices = pyo3::types::PyList::new_bound(py, &indices);
        let py_child_ids = pyo3::types::PyList::new_bound(py, &child_ids);
        let tuple = pyo3::types::PyTuple::new_bound(py, &[
            characteristic.to_object(py),
            count.to_object(py),
            py_indices.to_object(py),
            py_child_ids.to_object(py),
            depth.to_object(py),
        ]);
        py_list.append(tuple)?;
    }
    Ok(py_list.to_object(py))
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

    let results: Vec<i64> = (0..max_len)
        .into_par_iter()
        .map(|i| {
            let lat = lat_arr[if lat_arr.len() == 1 { 0 } else { i }];
            let lon = lon_arr[if lon_arr.len() == 1 { 0 } else { i }];
            geo2mort::geo2mort_scalar(lat, lon, order)
        })
        .collect();

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

    let results: Vec<i64> = (0..max_len)
        .into_par_iter()
        .map(|i| {
            let lo = lon_arr[if lon_arr.len() == 1 { 0 } else { i }];
            let la = lat_arr[if lat_arr.len() == 1 { 0 } else { i }];
            geo2mort::ang2pix_scalar(depth, lo, la) as i64
        })
        .collect();

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

    let results: Vec<(f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| geo2mort::pix2ang_scalar(depth, pixel_arr[i] as u64))
        .collect();

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
///
/// # Returns
/// For scalar: ndarray shape (3, 4)
/// For array of N pixels: ndarray shape (N, 3, 4)
#[pyfunction]
fn rust_boundaries<'py>(
    py: Python<'py>,
    depth: u8,
    pixel: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let pixel_is_scalar = pixel.extract::<i64>().is_ok();

    if pixel_is_scalar {
        let pix = pixel.extract::<i64>()? as u64;
        let xyz = geo2mort::boundaries_scalar(depth, pix);
        // Return as (3, 4) ndarray
        let arr = numpy::ndarray::Array2::from_shape_fn((3, 4), |(r, c)| xyz[r][c]);
        return Ok(PyArray2::from_owned_array_bound(py, arr).into_any().unbind());
    }

    let pixel_arr = pixel.extract::<PyReadonlyArray1<i64>>()?.to_vec()?;
    let n = pixel_arr.len();

    let results: Vec<[[f64; 4]; 3]> = (0..n)
        .into_par_iter()
        .map(|i| geo2mort::boundaries_scalar(depth, pixel_arr[i] as u64))
        .collect();

    // Shape (N, 3, 4) — matches healpy array output
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

    let results: Vec<(f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let x = data[i * 3];
            let y = data[i * 3 + 1];
            let z = data[i * 3 + 2];
            geo2mort::vec2ang_single(x, y, z)
        })
        .collect();

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

    let result = std::panic::catch_unwind(|| buffer::morton_buffer(&data, k));

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

/// A Python module implemented in Rust.
#[pymodule]
fn _rustie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_norm2mort, m)?)?;
    m.add_function(wrap_pyfunction!(split_children_rust, m)?)?;
    m.add_function(wrap_pyfunction!(rust_geo2mort, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ang2pix, m)?)?;
    m.add_function(wrap_pyfunction!(rust_pix2ang, m)?)?;
    m.add_function(wrap_pyfunction!(rust_boundaries, m)?)?;
    m.add_function(wrap_pyfunction!(rust_vec2ang, m)?)?;
    m.add_function(wrap_pyfunction!(rust_morton_buffer, m)?)?;
    Ok(())
}
