//! Rust-accelerated morton indexing for mortie
//!
//! This module provides Python bindings for fast morton encoding operations,
//! replacing the numba-accelerated functions to eliminate Dask conflicts.

pub mod geo2mort;
pub mod morton;
pub mod prefix_trie;

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
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

/// A Python module implemented in Rust.
#[pymodule]
fn _rustie(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_norm2mort, m)?)?;
    m.add_function(wrap_pyfunction!(split_children_rust, m)?)?;
    m.add_function(wrap_pyfunction!(rust_geo2mort, m)?)?;
    Ok(())
}
