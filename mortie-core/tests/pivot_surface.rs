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

/// A deterministic spread of `(nested, depth)` inputs covering every depth
/// `0..=29`, every base cell, and a mix of tuple paths within each.
fn nested_sweep() -> Vec<(u64, u8)> {
    let mut out = Vec::new();
    for depth in 0..=dm::MAX_ORDER {
        for base in 0..12u64 {
            // Paths chosen to exercise all four tuple values and both the body
            // (orders 1..=27) and the 28/29 tail: all-zero, all-three, an
            // alternating pattern, and a pseudo-random one.
            let span = 1u64 << (2 * depth); // cells below `base` at this depth
            for within in [0, span - 1, 0b01_10_11_00_01 % span, 0x2f5b_a731 % span] {
                out.push(((base << (2 * depth)) | within, depth));
            }
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
    for base in 0..12u64 {
        let nested = (base << (2 * depth)) | 0x1234_5678_9abc;
        let point = dm::from_nested_point(nested);
        let area = dm::from_nested(nested, depth);

        // Both are order 29 over the same nested cell, and both pivot back to it
        // — the point/area distinction is a packed-word concept the bare nested
        // index does not carry, so it is read off `kind_of`, not `to_nested`.
        assert_ne!(point, area, "a point must not collide with its area twin");
        assert_eq!(dm::to_nested(point), Some((depth, nested)));
        assert_eq!(dm::to_nested(area), Some((depth, nested)));
        assert_eq!(dm::kind_of(point), dm::Kind::Point);
        assert_eq!(dm::kind_of(area), dm::Kind::Area);
        assert_eq!(dm::order_of(point), dm::MAX_ORDER);
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
