//! Benchmarks comparing the new full-resolution `decimal_morton` 64-bit kernel
//! (issue #35) against the existing order-<=18 decimal-Morton path, for the
//! operations @espg asked about on PR #43: **encoding and decoding lat+lon
//! tuples** at orders the old path supports (6, 12, 17), plus the new kernel's
//! own primitives (`coarsen`, the `from_nested`/`to_nested` healpix bridge).
//!
//! Encode is timed on the *same* inputs through two pipelines:
//!
//! * **old** — `geo2mort_scalar` (healpix hash -> `fast_norm2mort_scalar`,
//!   decimal-packed i64). Capped at order 18.
//! * **new** — healpix hash -> `decimal_morton::from_nested` (the single-pass
//!   packed-word encoder, no intermediate tuple buffer).
//!
//! Both pipelines share the healpix `hash`, so the *delta* the benchmark
//! isolates is the bit-packing layer (decimal string-digit math vs. the packed
//! prefix/body/suffix word), which is the part issue #35 replaces. The new side
//! uses `from_nested` directly — the real encode path a `healpix::hash` feeds —
//! rather than decomposing the nested index back into a tuple buffer and calling
//! `encode` (the tuple path is only hit when a caller already holds per-order
//! tuples; a raw lat/lon never does).
//!
//! `coarsen`, `from_nested` and `to_nested` are timed standalone so CodSpeed
//! tracks the kernel primitives every later skin sits on, not just encode/decode.
//!
//! Coverage/polygon code is unaffected by this PR — nothing calls
//! `decimal_morton` yet — so its benchmarks live unchanged in
//! `coverage_bench.rs`; this file does not duplicate them.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use healpix::coords::Degrees;
use healpix::get;

use mortie_rustie::decimal_morton;
use mortie_rustie::geo2mort::geo2mort_scalar;
use mortie_rustie::morton::mort2nested;

// ---------------------------------------------------------------------------
// Shared input data
// ---------------------------------------------------------------------------

/// A deterministic spread of mid-latitude (lat, lon) pairs covering both
/// hemispheres and a range of base cells, so neither encoder hits a trivial
/// constant-fold.
fn sample_points(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let f = i as f64;
            let lat = -70.0 + (f * 7.3) % 140.0;
            let lon = -180.0 + (f * 13.7) % 360.0;
            (lat, lon)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Encode: lat/lon -> index
// ---------------------------------------------------------------------------

fn bench_encode(c: &mut Criterion) {
    let pts = sample_points(10_000);
    let mut group = c.benchmark_group("encode_latlon");

    for order in [6u8, 12, 17] {
        // old: geo2mort (healpix hash + decimal-pack)
        group.bench_with_input(BenchmarkId::new("old_geo2mort", order), &order, |b, &o| {
            b.iter(|| {
                let mut acc = 0u64;
                for &(lat, lon) in &pts {
                    acc ^= geo2mort_scalar(black_box(lat), black_box(lon), o);
                }
                acc
            })
        });

        // new: healpix hash + decimal_morton::from_nested (single-pass packed
        // word, no intermediate tuple buffer -- the real lat/lon encode path).
        group.bench_with_input(
            BenchmarkId::new("new_from_nested", order),
            &order,
            |b, &o| {
                let layer = get(o);
                b.iter(|| {
                    let mut acc = 0u64;
                    for &(lat, lon) in &pts {
                        let nested = layer.hash(Degrees(black_box(lon), black_box(lat)));
                        acc ^= decimal_morton::from_nested(nested, o);
                    }
                    acc
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Decode: index -> components
// ---------------------------------------------------------------------------

fn bench_decode(c: &mut Criterion) {
    let pts = sample_points(10_000);
    let mut group = c.benchmark_group("decode_index");

    for order in [6u8, 12, 17] {
        // Pre-encode both representations so the loop times only the decode.
        let layer = get(order);
        let old_words: Vec<u64> = pts
            .iter()
            .map(|&(lat, lon)| geo2mort_scalar(lat, lon, order))
            .collect();
        let new_words: Vec<u64> = pts
            .iter()
            .map(|&(lat, lon)| decimal_morton::from_nested(layer.hash(Degrees(lon, lat)), order))
            .collect();

        // old: mort2nested (decimal-digit unpack -> nested + depth)
        group.bench_with_input(
            BenchmarkId::new("old_mort2nested", order),
            &order,
            |b, _| {
                b.iter(|| {
                    let mut acc = 0u64;
                    for &w in &old_words {
                        let (nested, depth) = mort2nested(black_box(w));
                        acc ^= nested ^ depth as u64;
                    }
                    acc
                })
            },
        );

        // new: decimal_morton::decode (packed-word unpack). NB: decode returns an
        // owned `Vec<u8>` of tuples (its API), so the new side alone pays a
        // per-word allocation here that `mort2nested` (scalar return) does not.
        group.bench_with_input(
            BenchmarkId::new("new_decimal_morton", order),
            &order,
            |b, _| {
                b.iter(|| {
                    let mut acc = 0u64;
                    for &w in &new_words {
                        let dec = decimal_morton::decode(black_box(w)).expect("decode");
                        acc ^= dec.base_cell as u64 ^ dec.order as u64;
                    }
                    acc
                })
            },
        );

        // new (alloc-free): to_nested, the healpix-bridge inverse used for
        // center/vertices/UNIQ. Returns a scalar (depth, nested), so unlike
        // `decode` it carries no per-word `Vec` allocation -- the fair twin of
        // `mort2nested` above.
        group.bench_with_input(BenchmarkId::new("new_to_nested", order), &order, |b, _| {
            b.iter(|| {
                let mut acc = 0u64;
                for &w in &new_words {
                    let (depth, nested) =
                        decimal_morton::to_nested(black_box(w)).expect("to_nested");
                    acc ^= nested ^ depth as u64;
                }
                acc
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Coarsen: word at order 29 -> word at coarser order k
// ---------------------------------------------------------------------------

fn bench_coarsen(c: &mut Criterion) {
    let pts = sample_points(10_000);
    let mut group = c.benchmark_group("coarsen");

    // Pre-encode an order-29 corpus (the deepest source, so every target k
    // actually coarsens). from_nested at depth 29 gives canonical area words.
    let layer = get(29u8);
    let words: Vec<u64> = pts
        .iter()
        .map(|&(lat, lon)| decimal_morton::from_nested(layer.hash(Degrees(lon, lat)), 29))
        .collect();

    // Coarsen to a spread of targets: a body order, the 29->28 tail-preserving
    // case, and an order-0 base-cell rollup.
    for k in [6u8, 17, 28, 0] {
        group.bench_with_input(BenchmarkId::new("from_29_to", k), &k, |b, &k| {
            b.iter(|| {
                let mut acc = 0u64;
                for &w in &words {
                    acc ^= decimal_morton::coarsen(black_box(w), k).expect("coarsen");
                }
                acc
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// healpix bridge: from_nested (encode side timed standalone for CodSpeed)
// ---------------------------------------------------------------------------

fn bench_from_nested(c: &mut Criterion) {
    let pts = sample_points(10_000);
    let mut group = c.benchmark_group("from_nested");

    for order in [6u8, 17, 29] {
        let layer = get(order);
        // Pre-hash so the loop times only the bit-reshuffle, not the healpix hash.
        let nested: Vec<u64> = pts
            .iter()
            .map(|&(lat, lon)| layer.hash(Degrees(lon, lat)))
            .collect();
        group.bench_with_input(BenchmarkId::new("depth", order), &order, |b, &o| {
            b.iter(|| {
                let mut acc = 0u64;
                for &n in &nested {
                    acc ^= decimal_morton::from_nested(black_box(n), o);
                }
                acc
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_coarsen,
    bench_from_nested
);
criterion_main!(benches);
