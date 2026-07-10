//! Tests for the `descent-stats` instrumentation (issue #90).
//!
//! An integration test in its own binary (see `[[test]]` in Cargo.toml,
//! `required-features = ["descent-stats"]`): the collector is a process-wide
//! global, and the lib's other unit tests run descents concurrently — a
//! separate process keeps the take → run → take protocol clean.  Runs under
//! `cargo test --features descent-stats`; plain `cargo test` skips it.

use mortie_rustie::coverage::descent_stats::{self as ds, Cause};
use mortie_rustie::coverage::{polygon_to_morton_coverage, polygon_to_morton_moc};
use mortie_rustie::moc;

/// One serial test fn: the collector is global, so the scenarios must not
/// interleave.
#[test]
fn straddle_leaves_are_cause_tagged() {
    // Mid-latitude square (the bench shape): vertex leaves + quad crossings,
    // no exact-incidence touches, no near-pole path.
    let lats = vec![20.0, 20.0, 30.0, 30.0];
    let lons = vec![-125.0, -115.0, -115.0, -125.0];
    let _ = ds::take();
    let flat = polygon_to_morton_coverage(&lats, &lons, 6, true);
    let stats = ds::take();

    // Counter/record consistency and per-leaf sanity.
    assert_eq!(stats.leaf.iter().sum::<u64>() as usize, stats.leaves.len());
    assert!(!stats.leaves.is_empty());
    for l in &stats.leaves {
        assert_eq!(l.depth, 6, "exact descent stops at the target order");
        assert!(
            flat.binary_search(&l.morton).is_ok(),
            "straddle leaf {} not in the flat cover",
            l.morton
        );
        assert!(l.circ > 0.0 && l.circ < 0.1, "order-6 circumradius sane");
        let c = &l.center;
        let norm2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        assert!((norm2 - 1.0).abs() < 1e-9, "centre is a unit vector");
    }
    assert!(stats.leaf[Cause::VertexLeaf as usize] >= 1);
    assert!(stats.leaf[Cause::QuadCross as usize] > 0);
    assert_eq!(stats.leaf[Cause::QuadTouch as usize], 0);
    assert_eq!(stats.leaf[Cause::NearPoleBulge as usize], 0);
    assert!(
        stats.internal.iter().sum::<u64>() > 0,
        "coarse straddle nodes were refined"
    );

    // take() drained the collector.
    let empty = ds::take();
    assert_eq!(empty.leaves.len(), 0);
    assert_eq!(empty.leaf.iter().sum::<u64>(), 0);

    // #103 on-grid box (west edge exactly on lon 0): the closed-set
    // exact-incidence branch fires and is tagged QuadTouch.
    let lats = vec![20.0, 20.0, 25.0, 25.0];
    let lons = vec![0.0, 5.0, 5.0, 0.0];
    let moc_cells = polygon_to_morton_moc(&lats, &lons, 6);
    let stats = ds::take();
    assert!(
        stats.leaf[Cause::QuadTouch as usize] > 0,
        "on-grid edge must tag exact-incidence touches"
    );
    // Leaf records are pre-normalization: flattening both must agree.
    let flat = moc::to_order(&moc_cells, 6);
    for l in &stats.leaves {
        assert!(flat.binary_search(&l.morton).is_ok());
    }
}
