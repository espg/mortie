//! The mortie id codec: HEALPix cells as packed-`u64` morton words.
//!
//! This crate is the codec **only** (issue #48, option (c)): the packed-word
//! grammar and its encode/decode, order/truncation/containment arithmetic,
//! the decimal-string grammar, and the `(depth, nested-ipix) ↔ packed-word`
//! pivot primitives. It is contractually dependency-minimal — no non-std
//! dependencies — and it must never grow a moc-crate dependency; set-ops/
//! RangeMOC layers belong elsewhere. `tests/dep_contract.rs` guards the
//! zero-dependency half by failing `cargo test` if this crate's manifest
//! declares any dependency table; no CI job runs `cargo test` today, so the
//! guard fires for whoever runs the suite rather than on every push.
//!
//! # The pivot primitives
//!
//! External composers (e.g. healpix-geo's zuniq ↔ morton conversions, issue
//! #200) hinge on the depth/ipix pivot rather than coordinate-level encode:
//!
//! * [`decimal_morton::from_nested`] — `(nested, depth) → packed word`
//! * [`decimal_morton::from_nested_point`] — order-29 nested id → max-encoded
//!   *point* word (no area claim)
//! * [`decimal_morton::to_nested`] — `packed word → Some((depth, nested))`
//! * [`morton::nested2mort`] / [`morton::mort2nested`] — the same pivot in
//!   `(nested, depth)` tuple order, panicking on the empty sentinel
//!
//! [`from_nested`] and [`to_nested`] are re-exported at the crate root as the
//! canonical pivot pair.

pub mod decimal_morton;
pub mod morton;

pub use decimal_morton::{from_nested, to_nested};
