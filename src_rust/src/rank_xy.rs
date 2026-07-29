//! Subtree-local rank <-> face-local (x, y) deinterleave bindings (issue #149).
//!
//! Thin wrappers over the vendored healpix crate's z-order curve
//! (`healpix::zoc` — the same LUT/BMI2 kernels the crate's own `Layer`
//! decode routes through), so no new bit code lives here. Depth and range
//! validation happen in the Python wrappers (`mortie/rank_xy.py`), whose
//! pure-numpy ladder is the executable spec reference
//! (docs/specification.md §8); these bindings are the shipped fast path.

use healpix::zoc::get_zoc;
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::panic_msg;

/// Deinterleave subtree-local ranks into face-local (x, y) (vectorized).
///
/// # Arguments
/// * `rank_array` - Subtree-local ranks (u64 NumPy array), each in `[0, 4^depth)`
/// * `depth` - Subtree depth, `0..=29`
///
/// # Returns
/// Tuple of two u64 NumPy arrays: (x, y), the even- and odd-bit gathers.
#[pyfunction]
pub fn rust_rank_to_xy(
    py: Python<'_>,
    rank_array: PyReadonlyArray1<u64>,
    depth: u8,
) -> PyResult<PyObject> {
    // Borrow the contiguous numpy buffer directly (GIL-held, no copy); fall
    // back to a copy only for the rare non-contiguous input.
    let owned;
    let data: &[u64] = match rank_array.as_slice() {
        Ok(s) => s,
        Err(_) => {
            owned = rank_array.to_vec()?;
            &owned
        }
    };

    let result = std::panic::catch_unwind(|| {
        let zoc = get_zoc(depth);
        let mut xs = Vec::with_capacity(data.len());
        let mut ys = Vec::with_capacity(data.len());
        for &r in data {
            let ij = zoc.h2ij(r);
            xs.push(zoc.ij2i(ij) as u64);
            ys.push(zoc.ij2j(ij) as u64);
        }
        (xs, ys)
    });

    match result {
        Ok((xs, ys)) => {
            let py_x = xs.into_pyarray_bound(py).into_any().unbind();
            let py_y = ys.into_pyarray_bound(py).into_any().unbind();
            let tuple = pyo3::types::PyTuple::new_bound(py, &[py_x, py_y]);
            Ok(tuple.to_object(py))
        }
        Err(e) => Err(PyValueError::new_err(panic_msg(e, "rank_to_xy panicked"))),
    }
}

/// Interleave face-local (x, y) into subtree-local ranks (vectorized).
///
/// # Arguments
/// * `x_array` - Column coordinates (u64 NumPy array), each in `[0, 2^depth)`
/// * `y_array` - Row coordinates (u64 NumPy array), same length
/// * `depth` - Subtree depth, `0..=29`
///
/// # Returns
/// Subtree-local ranks as a u64 NumPy array.
#[pyfunction]
pub fn rust_xy_to_rank(
    py: Python<'_>,
    x_array: PyReadonlyArray1<u64>,
    y_array: PyReadonlyArray1<u64>,
    depth: u8,
) -> PyResult<PyObject> {
    let x_owned;
    let xs: &[u64] = match x_array.as_slice() {
        Ok(s) => s,
        Err(_) => {
            x_owned = x_array.to_vec()?;
            &x_owned
        }
    };
    let y_owned;
    let ys: &[u64] = match y_array.as_slice() {
        Ok(s) => s,
        Err(_) => {
            y_owned = y_array.to_vec()?;
            &y_owned
        }
    };

    if xs.len() != ys.len() {
        return Err(PyValueError::new_err(
            "x and y arrays must have the same length",
        ));
    }

    let result = std::panic::catch_unwind(|| {
        let zoc = get_zoc(depth);
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| zoc.ij2h(x as u32, y as u32))
            .collect::<Vec<u64>>()
    });

    match result {
        Ok(ranks) => Ok(ranks.into_pyarray_bound(py).into_any().unbind()),
        Err(e) => Err(PyValueError::new_err(panic_msg(e, "xy_to_rank panicked"))),
    }
}

#[cfg(test)]
mod tests {
    use healpix::zoc::get_zoc;

    /// Reference even-bit gather (the mask ladder from mortie/rank_xy.py).
    fn compact(mut v: u64) -> u64 {
        v &= 0x5555555555555555;
        v = (v | (v >> 1)) & 0x3333333333333333;
        v = (v | (v >> 2)) & 0x0F0F0F0F0F0F0F0F;
        v = (v | (v >> 4)) & 0x00FF00FF00FF00FF;
        v = (v | (v >> 8)) & 0x0000FFFF0000FFFF;
        (v | (v >> 16)) & 0x00000000FFFFFFFF
    }

    #[test]
    fn zoc_matches_mask_ladder_across_depth_classes() {
        // One depth per zoc size class (small/medium/large) plus the extremes.
        for depth in [1u8, 6, 8, 9, 16, 17, 29] {
            let zoc = get_zoc(depth);
            let n_rank = 4u64.pow(depth as u32);
            // Corners plus a fixed stride through the rank space.
            let probes = [0, 1, 2, 3, n_rank / 3, n_rank / 2, n_rank - 1];
            for &r in probes
                .iter()
                .chain((0..64).map(|k| n_rank * k / 64).collect::<Vec<_>>().iter())
            {
                let r = r.min(n_rank - 1);
                let ij = zoc.h2ij(r);
                let (x, y) = (zoc.ij2i(ij) as u64, zoc.ij2j(ij) as u64);
                assert_eq!(x, compact(r), "x mismatch at depth {depth}, rank {r}");
                assert_eq!(y, compact(r >> 1), "y mismatch at depth {depth}, rank {r}");
                assert_eq!(
                    zoc.ij2h(x as u32, y as u32),
                    r,
                    "round trip at depth {depth}"
                );
            }
        }
    }
}
