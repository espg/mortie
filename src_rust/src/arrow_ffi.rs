//! Arrow C Data Interface (PyCapsule) producer + consumer for `morton_index`
//! (issue #93).
//!
//! Makes the `morton_index` Arrow surface library-agnostic: any Arrow lib that
//! speaks the [PyCapsule C Data Interface] — `arro3-core` (the carrier zagg runs
//! on its Lambda worker, **without pyarrow**), pyarrow, polars — can pull a typed
//! morton column zero-copy, and hand one back. Where `mortie.arrow` (issue #86)
//! builds a pyarrow `ExtensionArray` and is gated on pyarrow, this path builds
//! the raw C structs in Rust via the `arrow` crate, so the runtime stays
//! numpy-only.
//!
//! The exported schema carries the `morton_index` Arrow **extension type** as
//! field metadata (`ARROW:extension:name` = [`EXTENSION_NAME`]), so the dtype
//! survives the boundary. Nulls travel as a real Arrow **validity bitmap** built
//! from the all-zero empty sentinel (`data == 0`), and are mapped back to the
//! sentinel on import — the same null<->sentinel convention as #86, but carried
//! byte-for-byte through any Arrow lib rather than via pyarrow's `fill_null`.
//!
//! [PyCapsule C Data Interface]: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

use std::collections::HashMap;
use std::ffi::CString;

use arrow::array::{Array, UInt64Array};
use arrow::datatypes::{DataType, Field};
use arrow::ffi::{from_ffi, FFI_ArrowArray, FFI_ArrowSchema};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// The Arrow extension name that tags a `morton_index` column. Mirrors
/// `mortie.arrow.EXTENSION_NAME` on the pyarrow side, so a column produced here
/// is recognized by the pandas `__from_arrow__` hook and vice versa.
pub const EXTENSION_NAME: &str = "mortie.morton_index";

/// The all-zero empty word: the missing-value sentinel (prefix 0). A word equal
/// to this maps to an Arrow null on export and comes back from a null on import.
const SENTINEL: u64 = 0;

/// Build the `morton_index` Arrow field: `uint64` storage, nullable, carrying the
/// extension-type metadata so the morton-ness survives the C-Data boundary.
fn morton_index_field() -> Field {
    let mut meta = HashMap::new();
    meta.insert(
        "ARROW:extension:name".to_string(),
        EXTENSION_NAME.to_string(),
    );
    // No parameters to carry; the extension name is the whole identity (matches
    // the pyarrow side's empty `__arrow_ext_serialize__`).
    meta.insert("ARROW:extension:metadata".to_string(), String::new());
    Field::new("item", DataType::UInt64, true).with_metadata(meta)
}

/// Build the C schema capsule (named `"arrow_schema"`) for a `morton_index`
/// column. Shared by the `__arrow_c_schema__` and `__arrow_c_array__` paths.
fn schema_capsule<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
    let ffi_schema = FFI_ArrowSchema::try_from(&morton_index_field())
        .map_err(|e| PyValueError::new_err(format!("failed to export Arrow schema: {e}")))?;
    let name = CString::new("arrow_schema").unwrap();
    PyCapsule::new_bound(py, ffi_schema, Some(name))
}

/// `MortonIndexArray.__arrow_c_schema__` backend: the schema capsule alone.
#[pyfunction]
pub fn rust_mi_export_c_schema(py: Python<'_>) -> PyResult<Bound<'_, PyCapsule>> {
    schema_capsule(py)
}

/// `MortonIndexArray.__arrow_c_array__` backend: `(schema_capsule, array_capsule)`
/// over the packed `uint64` words, with the sentinel mapped to an Arrow null via
/// a real validity bitmap and the `morton_index` extension type on the schema.
#[pyfunction]
pub fn rust_mi_export_c_array(
    py: Python<'_>,
    words: Vec<u64>,
) -> PyResult<(Bound<'_, PyCapsule>, Bound<'_, PyCapsule>)> {
    // Sentinel word -> Arrow null, mirroring #86's `data == _SENTINEL` mask, but
    // as a validity bitmap in the exported buffer rather than a pyarrow `mask=`.
    let arr: UInt64Array = words
        .iter()
        .map(|&w| if w == SENTINEL { None } else { Some(w) })
        .collect();
    let ffi_array = FFI_ArrowArray::new(&arr.into_data());
    let array_name = CString::new("arrow_array").unwrap();
    let array_capsule = PyCapsule::new_bound(py, ffi_array, Some(array_name))?;
    Ok((schema_capsule(py)?, array_capsule))
}

/// Return a capsule's pointer, but only after confirming it carries the expected
/// C Data Interface name. `PyCapsule::pointer()` looks the pointer up by the
/// capsule's *own* name and never checks it, so a mis-ordered or wrong-named pair
/// would otherwise let us reinterpret an `FFI_ArrowSchema` as an `FFI_ArrowArray`
/// (memory corruption reachable from safe Python). Guard the name first.
fn checked_pointer(
    capsule: &Bound<'_, PyCapsule>,
    expected: &str,
) -> PyResult<*mut std::ffi::c_void> {
    let ok = matches!(capsule.name()?, Some(name) if name.to_bytes() == expected.as_bytes());
    if !ok {
        return Err(PyValueError::new_err(format!(
            "expected a PyCapsule named {expected:?}"
        )));
    }
    let ptr = capsule.pointer();
    if ptr.is_null() {
        return Err(PyValueError::new_err(format!(
            "PyCapsule {expected:?} holds a null pointer"
        )));
    }
    Ok(ptr)
}

/// Consume a `(schema_capsule, array_capsule)` pair produced by *any* Arrow lib
/// and return the packed `uint64` words (Arrow nulls -> the empty sentinel).
///
/// This is the import half of the C Data Interface: it moves the `FFI_ArrowArray`
/// out of its capsule (leaving a released stub so the capsule destructor is a
/// no-op — the C-Data ownership-transfer convention) and borrows the schema.
#[pyfunction]
pub fn rust_mi_import_c_array(
    py: Python<'_>,
    schema_capsule: &Bound<'_, PyCapsule>,
    array_capsule: &Bound<'_, PyCapsule>,
) -> PyResult<Py<PyAny>> {
    let schema_ptr = checked_pointer(schema_capsule, "arrow_schema")? as *const FFI_ArrowSchema;
    let array_ptr = checked_pointer(array_capsule, "arrow_array")? as *mut FFI_ArrowArray;

    // Move the array struct out of the capsule and leave an empty (released)
    // struct behind, so the producer's capsule destructor won't double-free.
    // Borrow the schema for the duration of the call (the capsule outlives it).
    let (data, schema_dt) = unsafe {
        let ffi_array = std::ptr::replace(array_ptr, FFI_ArrowArray::empty());
        let data = from_ffi(ffi_array, &*schema_ptr)
            .map_err(|e| PyValueError::new_err(format!("failed to import Arrow array: {e}")))?;
        let dt = data.data_type().clone();
        (data, dt)
    };
    if schema_dt != DataType::UInt64 {
        return Err(PyValueError::new_err(format!(
            "morton_index import expects uint64 storage, got {schema_dt:?}"
        )));
    }

    let arr = UInt64Array::from(data);
    let words: Vec<u64> = (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                SENTINEL
            } else {
                arr.value(i)
            }
        })
        .collect();
    Ok(numpy::PyArray1::from_vec_bound(py, words)
        .into_any()
        .unbind())
}
