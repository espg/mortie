//! The external view of `mortie-core`'s public surface (issue #200).
//!
//! An integration test compiles as its own crate, so everything here is reached
//! exactly the way a downstream consumer reaches it — through `mortie_core::`
//! paths, with no `crate::` shortcut and no access to anything `pub(crate)`.
//! That is the point: the inline `#[cfg(test)]` suites in `src/` already pin the
//! codec's *behavior*, while this file pins what is *reachable*, so a future
//! visibility slip breaks the build here instead of silently breaking
//! healpix-geo (issue #203) after publication.
//!
//! The `(depth, nested-ipix) ↔ packed-word` pivot is the contract healpix-geo
//! composes zuniq ↔ morton conversions through, so it gets the sweep; the rest
//! of the documented surface gets a reachability pass.

use mortie_core::decimal_morton as dm;
use mortie_core::morton;

/// The nested index of the depth-`depth` cell under `base` whose order-`n`
/// tuple is `(n + phase) % 4` — consecutive orders walk `0,1,2,3`, so a single
/// path stores all four values at every four consecutive orders, and changing
/// `phase` slides which value lands at which order.
fn cycling_nested(base: u64, depth: u8, phase: u8) -> u64 {
    let mut within = 0u64;
    for n in 1..=depth {
        within |= (((n + phase) % 4) as u64) << (2 * (depth - n) as u32);
    }
    (base << (2 * depth)) | within
}

/// Overwrite the order-28/29 tail of a depth-28/29 nested index. At depth 28 the
/// order-28 tuple is the lowest pair; at depth 29 order 28 is the next pair up
/// and order 29 is the lowest (matching `from_nested`'s tail split).
fn with_tail(nested: u64, depth: u8, t28: u64, t29: u64) -> u64 {
    match depth {
        28 => (nested & !0b11) | t28,
        29 => (nested & !0b1111) | (t28 << 2) | t29,
        _ => unreachable!("only orders 28/29 carry a tail"),
    }
}

/// A deterministic spread of `(nested, depth)` inputs: one cycling path per
/// `(depth, base)` pair over every depth `0..=29` and every base cell, plus an
/// exhaustive enumeration of the order-28/29 tail.
fn nested_sweep() -> Vec<(u64, u8)> {
    let mut out = Vec::new();
    // Body: the phase is the base cell, so at any given order the twelve base
    // cells between them store all four tuple values. Exhaustive per-order
    // behavior belongs to the inline `#[cfg(test)]` suites; one representative
    // path per (depth, base) is what this external view needs.
    for depth in 0..=dm::MAX_ORDER {
        for base in 0..12u64 {
            out.push((cycling_nested(base, depth, base as u8), depth));
        }
    }
    // Tail: orders 28/29 live in the 6-bit suffix, a code path separate from the
    // 27-tuple body, and the cycling paths above only ever reach the diagonal
    // `(t28, t28 + 1)`. Enumerate it instead: all sixteen `(t28, t29)` pairs at
    // depth 29, and all four `t28` values at depth 28 (an order-28 word stores
    // no order-29 tuple).
    for t28 in 0..4u64 {
        out.push((with_tail(cycling_nested(t28, 28, 1), 28, t28, 0), 28));
        for t29 in 0..4u64 {
            let base = (t28 * 4 + t29) % 12;
            out.push((with_tail(cycling_nested(base, 29, 2), 29, t28, t29), 29));
        }
    }
    out
}

#[test]
fn the_crate_root_pivot_round_trips_every_depth_and_base_cell() {
    for (nested, depth) in nested_sweep() {
        // The crate root re-exports the canonical pivot pair; a downstream crate
        // may name either it or the module path, and they must agree.
        let word = mortie_core::from_nested(nested, depth);
        assert_eq!(
            word,
            dm::from_nested(nested, depth),
            "root and module from_nested disagree at depth {depth}, nested {nested}"
        );

        let (d, n) = mortie_core::to_nested(word).expect("an encoded word decodes");
        assert_eq!((d, n), (depth, nested), "pivot round trip");
        assert_eq!(
            dm::to_nested(word),
            Some((d, n)),
            "root and module to_nested disagree at depth {depth}, nested {nested}"
        );

        // The pivot is the area encoder: order and kind are part of the contract.
        assert_eq!(dm::order_of(word), depth);
        assert_eq!(dm::kind_of(word), dm::Kind::Area);
    }
}

#[test]
fn the_morton_bridge_is_the_same_pivot_in_tuple_order() {
    // healpix-geo composes through `morton`'s `(nested, depth)` argument order;
    // it must be the identical mapping, not a parallel implementation.
    for (nested, depth) in nested_sweep() {
        let word = morton::nested2mort(nested, depth);
        assert_eq!(word, dm::from_nested(nested, depth), "nested2mort");
        assert_eq!(morton::mort2nested(word), (nested, depth), "mort2nested");
    }
}

#[test]
fn a_max_encoded_point_pivots_to_the_same_nested_cell_as_its_area_twin() {
    let depth = dm::MAX_ORDER;
    // A point keys its suffix off the `(t28, t29)` pair rather than the area
    // tail's order code, so walk all sixteen pairs on every base cell.
    for base in 0..12u64 {
        for tail in 0..16u64 {
            let nested = with_tail(cycling_nested(base, depth, 3), depth, tail / 4, tail % 4);
            let point = dm::from_nested_point(nested);
            let area = dm::from_nested(nested, depth);

            // Both are order 29 over the same nested cell, and both pivot back
            // to it — the point/area distinction is a packed-word concept the
            // bare nested index does not carry, so it is read off `kind_of`,
            // not `to_nested`.
            assert_ne!(point, area, "a point must not collide with its area twin");
            assert_eq!(dm::to_nested(point), Some((depth, nested)));
            assert_eq!(dm::to_nested(area), Some((depth, nested)));
            assert_eq!(dm::kind_of(point), dm::Kind::Point);
            assert_eq!(dm::kind_of(area), dm::Kind::Area);
            assert_eq!(dm::order_of(point), dm::MAX_ORDER);
        }
    }
}

#[test]
fn the_documented_codec_surface_is_reachable_from_outside() {
    // Every item the crate documents as public, named from a foreign crate. The
    // assertions are deliberately shallow — behavior is pinned by the inline
    // suites; what is being tested here is that these names still resolve.
    assert_eq!(dm::MAX_ORDER, 29);
    assert_eq!(dm::BODY_TUPLES, 27);

    // encode / decode, with the decoded struct's fields read individually.
    let tuples = [0u8, 1, 2, 3, 0];
    let word = dm::encode(4, &tuples, 5);
    let decoded: dm::DecimalMorton = dm::decode(word).expect("a well-formed word decodes");
    assert_eq!(decoded.base_cell, 4);
    assert_eq!(decoded.order, 5);
    assert_eq!(decoded.kind, dm::Kind::Area);
    assert_eq!(decoded.tuples, tuples);
    assert_eq!(dm::base_cell_of(word), Some(4));

    // encode_point takes all 29 tuples and yields the Point kind.
    let point = dm::encode_point(4, &[2u8; 29]);
    assert_eq!(dm::kind_of(point), dm::Kind::Point);

    // The empty sentinel is the documented rejection, carrying DecodeError.
    assert_eq!(dm::decode(0), Err(dm::DecodeError::Empty));
    assert_eq!(dm::base_cell_of(0), None);
    assert_eq!(dm::to_nested(0), None);

    // coarsen: truncation in packed space equals truncation in nested space.
    let (order, nested) = dm::to_nested(word).expect("to_nested");
    let coarse = dm::coarsen(word, 2).expect("coarsen a valid word");
    assert_eq!(
        dm::to_nested(coarse),
        Some((2, nested >> (2 * (order - 2) as u32)))
    );

    // common_ancestor: the deepest enclosing cell of a set, and its error type.
    let sibling = dm::encode(4, &[0u8, 1, 2, 3, 1], 5);
    assert_eq!(
        dm::common_ancestor(&[word, sibling]),
        Ok(dm::encode(4, &tuples, 4))
    );
    assert_eq!(
        dm::common_ancestor(&[]),
        Err(dm::CommonAncestorError::Empty)
    );
    assert_eq!(
        dm::common_ancestor(&[word, dm::encode(5, &tuples, 5)]),
        Err(dm::CommonAncestorError::MixedBaseCell)
    );

    // The decimal-string grammar, both directions, plus its error type.
    let repr = dm::to_decimal_repr(word).expect("a valid word renders");
    assert_eq!(dm::from_decimal_repr(&repr), Ok(word));
    assert_eq!(dm::to_decimal_repr(0), None);
    let err: dm::ParseError = dm::from_decimal_repr("not a morton").unwrap_err();
    assert!(!format!("{err}").is_empty(), "ParseError renders");

    // The one-way legacy converter, keyed off the repr grammar it shares.
    assert_eq!(dm::from_legacy_decimal(112), dm::encode(0, &[0, 1], 2));
}
